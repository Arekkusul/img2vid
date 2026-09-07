use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use mlx_gen::gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_krea::model::{load_edit, load_raw, load_turbo_edit};

/// Generate an image with Krea 2, or edit one given a reference image + the identity-edit LoRA.
/// Thin CLI wrapper around mlx-gen-krea, mirroring the shape of img2vid's existing `img2vid`
/// video CLI: real flags in, a PNG out, clean errors to stderr with a non-zero exit on failure.
#[derive(Parser)]
#[command(name = "krea-gen")]
struct Args {
    /// Path to the assembled Krea 2 snapshot dir (transformer/ + text_encoder/ + vae/ + ...).
    #[arg(long)]
    snapshot: PathBuf,

    /// Prompt (text-to-image) or edit instruction (edit mode).
    #[arg(long)]
    prompt: String,

    #[arg(long)]
    negative_prompt: Option<String>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 1024)]
    width: u32,

    #[arg(long, default_value_t = 1024)]
    height: u32,

    #[arg(long, default_value_t = 30)]
    steps: u32,

    #[arg(long, default_value_t = 4.0)]
    guidance: f32,

    #[arg(long)]
    seed: Option<u64>,

    /// If set, run image-edit mode: apply --prompt as an edit instruction to this source image.
    #[arg(long)]
    edit_source: Option<PathBuf>,

    /// Optional identity-edit LoRA safetensors path (edit mode only).
    #[arg(long)]
    lora: Option<PathBuf>,

    /// Use the distilled, CFG-free Turbo edit path instead of the full-CFG Raw edit.
    #[arg(long)]
    turbo_edit: bool,
}

fn load_rgb(path: &PathBuf) -> Result<Image, String> {
    let img = image::open(path)
        .map_err(|e| format!("failed to open source image {}: {e}", path.display()))?
        .to_rgb8();
    let (width, height) = img.dimensions();
    Ok(Image {
        width,
        height,
        pixels: img.into_raw(),
    })
}

fn save_png(img: &Image, path: &PathBuf) -> Result<(), String> {
    let buf: image::RgbImage = image::ImageBuffer::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or_else(|| "generated pixel buffer did not match its own width/height".to_string())?;
    buf.save(path)
        .map_err(|e| format!("failed to save output image {}: {e}", path.display()))
}

fn run(args: Args) -> Result<(), String> {
    let base_spec = LoadSpec::new(WeightsSource::Dir(args.snapshot.clone()));

    let (generator, conditioning) = if let Some(source_path) = &args.edit_source {
        let spec = match &args.lora {
            Some(lora_path) => base_spec.with_adapters(vec![AdapterSpec::new(
                lora_path.clone(),
                1.0,
                AdapterKind::Lora,
            )]),
            None => base_spec,
        };
        let generator = if args.turbo_edit {
            load_turbo_edit(&spec).map_err(|e| format!("load krea_2_turbo_edit: {e}"))?
        } else {
            load_edit(&spec).map_err(|e| format!("load krea_2_edit: {e}"))?
        };
        let source = load_rgb(source_path)?;
        (
            generator,
            vec![Conditioning::Reference {
                image: source,
                strength: None,
            }],
        )
    } else {
        let generator = load_raw(&base_spec).map_err(|e| format!("load krea_2_raw: {e}"))?;
        (generator, Vec::new())
    };

    let turbo_edit = args.edit_source.is_some() && args.turbo_edit;
    let request = GenerationRequest {
        prompt: args.prompt.clone(),
        negative_prompt: if turbo_edit {
            None
        } else {
            Some(args.negative_prompt.clone().unwrap_or_default())
        },
        width: args.width,
        height: args.height,
        count: 1,
        seed: args.seed,
        steps: Some(args.steps),
        guidance: if turbo_edit { None } else { Some(args.guidance) },
        conditioning,
        cancel: CancelFlag::new(),
        ..Default::default()
    };

    let output = generator
        .generate(&request, &mut |_| {})
        .map_err(|e| format!("generate: {e}"))?;

    let image = match output {
        GenerationOutput::Images(mut images) => images
            .pop()
            .ok_or_else(|| "generator produced zero images".to_string())?,
        _ => return Err("generator returned a non-image output".to_string()),
    };

    save_png(&image, &args.output)?;
    eprintln!("wrote {} ({}x{})", args.output.display(), image.width, image.height);
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("krea-gen: {msg}");
            ExitCode::FAILURE
        }
    }
}
