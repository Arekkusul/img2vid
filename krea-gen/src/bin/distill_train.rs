//! Single-stage step-distillation LoRA training for Krea 2 Raw, built on the user's own converted
//! weights (see ../main.rs / img2vid's image_generate.py for the standard t2i/edit CLI).
//!
//! This is NOT the standard `KreaRawTrainer` (mlx-gen-krea/src/training.rs), which regresses a LoRA
//! onto real images via ordinary flow-matching reconstruction -- that teaches subject/style, not
//! fewer sampling steps. This binary implements single-stage progressive distillation (Salimans &
//! Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models"): the SAME frozen base
//! model serves as its own teacher, so no second trainable model copy is resident (avoiding the
//! memory wall that ruled out TDM/Hyper-SD-style methods on this hardware -- see
//! docs/krea-distillation-research.md). Per training example:
//!
//!   1. x0 = a real training photo's VAE latent, noise ~ N(0,I), context = its caption's features
//!      (all frozen/no-grad, cached once -- identical to KreaRawTrainer's caching stage).
//!   2. Pick a random adjacent triple (sigma_i, sigma_mid, sigma_end) from the FULL 52-step Krea
//!      schedule; x_i = (1-sigma_i)*x0 + sigma_i*noise (closed-form flow interpolation, matching
//!      training.rs::build_batch exactly).
//!   3. TEACHER (bare base, adapters cleared, no grad): two small Euler steps sigma_i -> sigma_mid
//!      -> sigma_end, using the crate's own Euler step formula `x + v*(sigma_next-sigma)`
//!      (gen-core/src/sampling/unified.rs -- reused verbatim rather than re-derived, to avoid a
//!      sign-convention bug).
//!   4. STUDENT (LoRA installed, differentiable): ONE big Euler step sigma_i -> sigma_end from the
//!      SAME x_i.
//!   5. Loss = MSE(student's implied velocity, teacher's two-step composite velocity). Backward
//!      flows only through the student's forward + LoRA factors (the teacher's two passes are
//!      plain, un-traced computation, so carry no gradient -- exactly like `noise - x0` in the
//!      existing trainer's loss_fn).
//!
//! Scope, honestly stated (see docs/krea-distillation-research.md): this collapses 2 fine steps
//! into 1 (a single distillation stage, ~2x), not Turbo's 6.5x (52->8) TDM-distilled jump -- Turbo
//! is a separately, expensively trained model, not LoRA-derivable from Raw. Not expected to match
//! Turbo-grade quality; the goal is a real, honestly-scoped, measurably-converging LoRA.

use std::path::{Path, PathBuf};

use clap::Parser;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{build_lora_targets, clear_adapters, LoraParams, TrainAdapter};
use mlx_gen::TrainOptimizer;
use mlx_gen_krea::loader::{load_text_encoder, load_transformer};
use mlx_gen_krea::schedule::{dynamic_mu, krea_sigmas};
use mlx_gen_krea::text_encoder::{KreaTextEncoder, KreaTokenizer};
use mlx_gen_krea::transformer::Krea2Transformer;
use mlx_gen_krea::vae::{load_vae, QwenVae};
use mlx_rs::error::Exception;
use mlx_rs::memory::get_memory_limit;
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

/// Projected DENSE (non-block-checkpointed) first-step peak memory in GB, as a function of unified
/// token count `s`. Duplicated from `mlx-gen-krea/src/training.rs`'s private
/// `projected_dense_peak_gb`/`PREFLIGHT_BF16` (same constants, same model) rather than imported --
/// that fn isn't `pub`. Keep in sync if upstream refits these.
const PREFLIGHT_BF16: (f64, f64, f64) = (24.0, 6.0e-3, 1.5e-7);
const PREFLIGHT_TXT_TOKENS: f64 = 64.0;

fn projected_dense_peak_gb(s: f64) -> f64 {
    PREFLIGHT_BF16.0 + PREFLIGHT_BF16.1 * s + PREFLIGHT_BF16.2 * s * s
}

/// Refuse a non-checkpointed run whose dense first step would exceed this machine's memory budget
/// (mirrors `training.rs::preflight_memory_guard` -- same 0.85 safety margin, same token-count
/// formula). Consulted only when gradient checkpointing is off.
fn preflight_memory_guard(edge: u32) {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = projected_dense_peak_gb(s);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        panic!(
            "distill_train: a dense first training step at resolution {edge} needs ~{projected:.0} \
             GB, exceeding this machine's ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit x \
             0.85). Pass --gradient-checkpointing (recomputes block activations in the backward) or \
             reduce --resolution."
        );
    }
}

/// Per-single-stream-block LOCAL LoRA target paths (e.g. `"attn.to_q"`), in trained-file order --
/// the factors `forward_with_blocks_checkpointed` threads as explicit checkpoint inputs. Mirrors
/// `training.rs`'s private helper of the same shape.
fn block_local_targets(num_blocks: usize, target_paths: &[String]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = vec![Vec::new(); num_blocks];
    for path in target_paths {
        if let Some((idx, local)) = path
            .strip_prefix("transformer_blocks.")
            .and_then(|rest| rest.split_once('.'))
        {
            if let Ok(i) = idx.parse::<usize>() {
                if i < out.len() {
                    out[i].push(local.to_string());
                }
            }
        }
    }
    out
}

/// The standard PEFT attention surface -- matches KreaRawTrainer's default target modules exactly,
/// so a trained adapter here reloads through the same sc-7578 apply path.
const TARGET_MODULES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// Granularity of the reference teacher schedule this distills FROM (the real 52-step Raw
/// schedule) -- the same value `image_generate.py`'s default `--steps` uses, so the distilled
/// student targets exactly the inference path we already run.
const FINE_STEPS: usize = 52;

/// Text encoder loads Q8 (frozen, caption-cache only, then dropped) -- matches KreaRawTrainer.
const TRAINER_ENCODER_BITS: i32 = 8;

#[derive(Parser)]
#[command(about = "Single-stage step-distillation LoRA training for Krea 2 Raw")]
struct Args {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    training_dir: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 500)]
    steps: u32,
    #[arg(long, default_value_t = 16)]
    rank: i32,
    #[arg(long, default_value_t = 16.0)]
    alpha: f32,
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    #[arg(long, default_value_t = 50)]
    save_every: u32,
    #[arg(long, default_value_t = 1024)]
    resolution: u32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Recompute per-block activations in the backward instead of retaining them (trades compute
    /// for memory). Defaults ON: this runs unattended on a single 64GB Mac with no operator
    /// watching for an OOM, so the safer default is the right one; the dense path is opt-in via
    /// --gradient-checkpointing=false for anyone who has profiled real headroom.
    #[arg(long, default_value_t = true)]
    gradient_checkpointing: bool,
}

struct DatasetItem {
    x0: Array,
    context: Array,
}

fn decode_image(path: &Path) -> Image {
    let dynimg = image::open(path).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Image {
        width,
        height,
        pixels: rgb.into_raw(),
    }
}

fn encode_latents(vae: &QwenVae, image: &Image, edge: u32) -> Array {
    let pre = preprocess_init_image(image, edge, edge).expect("preprocess init image");
    let lat = vae.encode(&pre).expect("vae encode");
    lat.squeeze_axes(&[2]).expect("squeeze temporal axis")
}

fn encode_caption(tokenizer: &KreaTokenizer, encoder: &KreaTextEncoder, caption: &str) -> Array {
    let (ids, attn) = tokenizer.encode_prompt(caption).expect("tokenize caption");
    encoder.forward(&ids, &attn).expect("encode caption")
}

/// Load every `img_*.png` + sibling `.txt` caption pair in `dir`, sorted by filename.
fn load_dataset(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut entries: Vec<(PathBuf, String)> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read training dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .map(|img_path| {
            let txt_path = img_path.with_extension("txt");
            let caption = std::fs::read_to_string(&txt_path)
                .unwrap_or_else(|e| panic!("read caption {}: {e}", txt_path.display()))
                .trim()
                .to_string();
            (img_path, caption)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// One frozen forward pass (no LoRA installed -- caller must have cleared adapters first), no grad.
fn teacher_forward(transformer: &Krea2Transformer, x: &Array, sigma: f32, context: &Array) -> Array {
    let t = Array::from_slice(&[sigma], &[1]);
    transformer
        .forward(x, &t, context, None)
        .expect("teacher forward")
}

/// The crate's own Euler step: `x + v*(sigma_next - sigma)` (gen-core sampling/unified.rs::Euler,
/// algebraically identical to the legacy flow-match step this model family is trained/sampled
/// under -- reused verbatim rather than re-derived).
fn euler_step(x: &Array, v: &Array, sigma: f32, sigma_next: f32) -> Array {
    let dt = sigma_next - sigma;
    let scaled = v.multiply(Array::from_slice(&[dt], &[1])).expect("scale v");
    x.add(&scaled).expect("euler step add")
}

/// A tiny deterministic hash, seed -> u32, used only to pick a random schedule index per step --
/// not cryptographic, not MLX's RNG (avoids spending an MLX random `key()` call on a scalar index).
fn rand_u32(seed: u64) -> u32 {
    let mut x = seed ^ 0x9E3779B97F4A7C15;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    (x & 0xFFFF_FFFF) as u32
}

/// One full distillation training step's forward+backward, mirroring
/// `mlx-gen-krea/src/training.rs::compute_loss_grads`'s function-boundary pattern exactly: the
/// `move` closure passed to `keyed_value_and_grad` captures `transformer`/`adapter` by (re-)borrow
/// for the DURATION OF THIS CALL ONLY, releasing them back to the caller when this function
/// returns -- letting the outer training loop reuse the same `transformer`/`adapter` next step.
#[allow(clippy::too_many_arguments)]
fn train_step(
    transformer: &mut Krea2Transformer,
    adapter: &TrainAdapter,
    adapter_paths: &[String],
    params: &LoraParams,
    alpha: f32,
    rank: f32,
    item: &DatasetItem,
    sigmas: &[f32],
    compute_dtype: Dtype,
    lora_dtype: Option<Dtype>,
    seed_step: u64,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> (f32, LoraParams) {
    let max_i = sigmas.len().saturating_sub(3);
    let i = (rand_u32(seed_step) as usize) % (max_i + 1);
    let (sigma_i, sigma_mid, sigma_end) = (sigmas[i], sigmas[i + 1], sigmas[i + 2]);

    let noise = random::normal::<f32>(
        item.x0.shape(),
        None,
        None,
        Some(&random::key(seed_step.wrapping_mul(2) + 1).expect("rng key")),
    )
    .expect("sample noise");
    // x_i = (1-sigma_i)*x0 + sigma_i*noise -- training.rs::build_batch's exact interpolation.
    let one_minus = Array::from_slice(&[1.0 - sigma_i], &[1]);
    let s = Array::from_slice(&[sigma_i], &[1]);
    let x_i = item
        .x0
        .multiply(&one_minus)
        .unwrap()
        .add(&noise.multiply(&s).unwrap())
        .expect("build x_i");

    // --- TEACHER: two small frozen Euler steps, no LoRA installed, no grad ---
    clear_adapters(transformer, adapter_paths);
    let v1 = teacher_forward(transformer, &x_i, sigma_i, &item.context);
    let x_mid = euler_step(&x_i, &v1, sigma_i, sigma_mid);
    let v2 = teacher_forward(transformer, &x_mid, sigma_mid, &item.context);
    let x_end_teacher = euler_step(&x_mid, &v2, sigma_mid, sigma_end);
    // Implied single-big-step velocity a student jumping sigma_i -> sigma_end must match.
    let span = sigma_end - sigma_i;
    let target_v = x_end_teacher
        .subtract(&x_i)
        .expect("teacher delta")
        .multiply(Array::from_slice(&[1.0 / span], &[1]))
        .expect("scale target velocity");
    eval([&target_v]).expect("eval teacher target");

    // --- STUDENT: one big differentiable Euler step, LoRA installed inside the traced loss ---
    let x_i_dtype = x_i.as_dtype(compute_dtype).expect("cast x_i");
    let timestep = Array::from_slice(&[sigma_i], &[1]);
    let context = item.context.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        // Install on `self` unconditionally (mirrors compute_loss_grads): needed for the dense
        // branch; moot-but-harmless for the checkpointed branch, which re-injects factors per
        // block from `p` directly as explicit checkpoint inputs.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, Dtype::Bfloat16)?;
        let v_student = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_blocks_checkpointed(
                    &x_i_dtype, &timestep, &context, None, &p, locals, alpha,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward(&x_i_dtype, &timestep, &context, None)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diff = v_student.as_dtype(Dtype::Float32)?.subtract(&target_v)?;
        let loss = diff.square()?.mean(None)?;
        Ok(vec![loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0).expect("value_and_grad");
    (val[0].item::<f32>(), grads)
}

fn main() {
    let args = Args::parse();

    eprintln!("[distill_train] loading tokenizer + text encoder (Q8, frozen)...");
    let tokenizer = KreaTokenizer::from_snapshot(&args.snapshot).expect("load tokenizer");
    let mut encoder = load_text_encoder(&args.snapshot).expect("load text encoder");
    encoder
        .quantize(TRAINER_ENCODER_BITS)
        .expect("quantize text encoder");

    eprintln!("[distill_train] loading transformer (dense bf16, trainable base)...");
    let mut transformer = load_transformer(&args.snapshot).expect("load transformer");
    let compute_dtype = transformer.compute_dtype();
    let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);

    eprintln!("[distill_train] loading VAE...");
    let vae = load_vae(&args.snapshot).expect("load vae");

    let edge = bucket_resolution(args.resolution);
    let img_seq = (edge as f64 / 16.0).powi(2);
    let sigmas = krea_sigmas(FINE_STEPS, dynamic_mu(img_seq));
    eprintln!(
        "[distill_train] fine reference schedule: {FINE_STEPS} steps, edge={edge}, {} sigma nodes",
        sigmas.len()
    );

    // --- cache dataset: VAE latents + caption features (frozen text encoder still resident) ---
    let raw_items = load_dataset(&args.training_dir);
    if raw_items.is_empty() {
        panic!(
            "no img_*.png + .txt pairs found under {}",
            args.training_dir.display()
        );
    }
    eprintln!("[distill_train] caching {} training items...", raw_items.len());
    let cache: Vec<DatasetItem> = raw_items
        .iter()
        .map(|(img_path, caption)| {
            let image = center_crop_square(&decode_image(img_path));
            let x0 = encode_latents(&vae, &image, edge);
            let context = encode_caption(&tokenizer, &encoder, caption);
            eval([&x0, &context]).expect("eval cached item");
            DatasetItem { x0, context }
        })
        .collect();

    // Every caption is cached now -- free the encoder (multi-GB resident, idle for the rest of the
    // run), matching KreaRawTrainer's memory-hardening pattern exactly.
    drop(encoder);
    mlx_rs::memory::clear_cache();

    // --- LoRA targets: default single-stream block attention (to_q/to_k/to_v/to_out.0) ---
    let target_paths: Vec<String> = transformer
        .adaptable_paths()
        .into_iter()
        .filter(|path| {
            path.starts_with("transformer_blocks.")
                && TARGET_MODULES
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect();
    if target_paths.is_empty() {
        panic!("no LoRA target modules resolved on the transformer");
    }
    eprintln!("[distill_train] {} LoRA target modules", target_paths.len());
    let (targets, mut params) =
        build_lora_targets(&mut transformer, &target_paths, args.rank, args.seed)
            .expect("build lora targets");
    let adapter = TrainAdapter::Lora { targets };
    let adapter_paths = adapter.paths();

    // sc-7577-style safety (mirrors training.rs): checkpointing ON recomputes each block's
    // activations in the backward instead of retaining them; SDPA-segment checkpointing covers
    // attention in that case, so the per-block flag goes the opposite way of whole-block
    // checkpointing. OFF is refused up front if the dense first step would exceed this machine's
    // budget, rather than risking an uncatchable SIGKILL mid-run.
    let checkpoint_targets = block_local_targets(transformer.num_blocks(), &target_paths);
    transformer.set_sdpa_checkpoint(!args.gradient_checkpointing);
    if !args.gradient_checkpointing {
        preflight_memory_guard(edge);
    }
    let checkpoint_blocks: Option<&[Vec<String>]> =
        args.gradient_checkpointing.then_some(checkpoint_targets.as_slice());

    let mut opt = TrainOptimizer::from_config("adamw", args.lr, 0.0).expect("build optimizer");
    let stem = "krea2_distill_x2";
    let mut last_loss = 0.0f32;

    for step in 1..=args.steps {
        let item = &cache[(step as usize - 1) % cache.len()];
        let seed_step = args.seed.wrapping_add(step as u64);

        let (loss, grads) = train_step(
            &mut transformer,
            &adapter,
            &adapter_paths,
            &params,
            args.alpha,
            args.rank as f32,
            item,
            &sigmas,
            compute_dtype,
            lora_dtype,
            seed_step,
            checkpoint_blocks,
        );
        last_loss = loss;

        let (clipped, _norm) = clip_grad_norm(&grads, 1.0).expect("clip grad norm");
        let clipped: LoraParams = clipped.into_iter().map(|(k, v)| (k, v.into_owned())).collect();
        opt.step(&mut params, &clipped).expect("optimizer step");
        eval(params.values()).expect("eval params");

        if step % 10 == 0 || step == args.steps {
            eprintln!("[distill_train] step {step}/{}: loss={loss:.6}", args.steps);
        }
        if args.save_every > 0 && step % args.save_every == 0 && step != args.steps {
            let ckpt = args.output.join(checkpoint_filename(stem, step));
            adapter
                .save(&params, args.alpha, args.rank as f32, 0, "", &ckpt)
                .expect("save checkpoint");
            eprintln!("[distill_train] checkpoint saved: {}", ckpt.display());
        }
    }

    std::fs::create_dir_all(&args.output).expect("create output dir");
    let final_path = args.output.join(format!("{stem}.safetensors"));
    adapter
        .save(&params, args.alpha, args.rank as f32, 0, "", &final_path)
        .expect("save final adapter");
    eprintln!(
        "[distill_train] DONE. steps={} final_loss={last_loss:.6} adapter={}",
        args.steps,
        final_path.display()
    );
}
