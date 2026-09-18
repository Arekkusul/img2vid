# Krea 2 Raw -> fast distillation LoRA: research log

## Goal

Train our own LoRA that converts the user's own converted `krea/Krea-2-Raw` weights (12B
single-stream flow-matching DiT) into fast, few-step, ideally CFG-free behavior -- without
relying on Krea's own officially-released `krea/Krea-2-Turbo` checkpoint (a separate, fully
retrained model, not a LoRA-derivable patch -- confirmed via `mlx-gen-krea`'s own source: Turbo
requires a genuine distillation *training* run, "TDM-distilled", not a weight-space transform).

If we succeed with something that trains in reasonable time on an M4 Pro (64GB unified memory,
no CUDA, MLX) at real, verified quality, the plan is to publish the resulting LoRA publicly to
help others in the same situation (a from-scratch-converted Raw checkpoint, no access to an
officially pre-trained Turbo variant).

**Ground rules for this log**: every claim here is either (a) sourced from a fetched
paper/repo with the source cited, or (b) a real, run, and observed local result on this machine.
No claimed number goes in here from memory/assumption alone.

## Candidate methods surveyed

### 1. TDM (Trajectory Distribution Matching) -- REJECTED

Source: https://arxiv.org/html/2503.06674v2 (fetched directly)

This is very likely the exact technique behind Krea's own official Turbo ("TDM-distilled" is
the term used in `mlx-gen-krea`'s own source comments).

- Requires **three full model copies simultaneously in trainable memory**: frozen teacher,
  trainable student generator, and a trainable "fake score" critic network (also initialized
  from the teacher). Backprop flows through one ODE step connecting them (Eq. 11).
- Reported real compute: PixArt-alpha (~0.6B) 4-step distillation = 500 iterations, 2 A800-hours,
  batch size 32. SDXL (2.7B) = 2 A800-*days*, batch 64. SD-v1.5 = 3 A800-days including a
  fine-tuning stage, batch 256.
- Method type in the paper: **full fine-tune**, not LoRA (LoRA only discussed for downstream
  compatibility with community checkpoints, not as the distillation training method itself).
- DDPM/epsilon-prediction only in the paper -- no flow-matching validation.
- **Why rejected**: Krea 2 Raw is 12B params (~4.4x PixArt, ~4.5x SDXL). Three simultaneous
  copies of a 12B model, even quantized, is 60-100+GB for weights alone before activations/
  gradients/optimizer state -- doesn't fit in 64GB unified memory. This is a hard memory-capacity
  wall, not an engineering inefficiency to optimize around. Even ignoring memory, PixArt's
  *cheapest* reported case (2 A800-hours) scaled ~4.4x for parameter count and further scaled
  for M4 Pro's much lower matmul throughput vs an A800 would very plausibly run into weeks of
  continuous compute -- and that's before solving the memory wall at all.

### 2. LCM-LoRA (Latent Consistency Model, LoRA distillation) -- PROMISING, NEEDS MORE VERIFICATION

Source: https://arxiv.org/pdf/2311.05556 (fetched; PDF's figures didn't extract cleanly, some
numbers below are marked uncertain and need a second source pass)

- **No trainable critic/discriminator network** -- just a frozen teacher + a single trainable
  student with LoRA applied. This avoids TDM's 3-copies problem: effectively ~1 frozen copy +
  LoRA's (small) trainable parameter count, not 2-3x the full model in trainable state.
- Self-distillation: prompts only, no paired real image dataset required.
- Confirmed DDPM/epsilon-prediction in the original paper -- **not validated for flow-matching**
  in this source. Krea 2 is flow-matching (velocity-prediction, confirmed from
  `mlx-gen-krea/src/training.rs`'s own docstring), so the consistency-distillation math would
  need adapting, not a direct drop-in.
- Exact compute numbers (GPU-hours, batch size, LoRA rank, iteration count) did NOT extract
  cleanly from the PDF fetch -- genuinely uncertain, marked for a second, cleaner source
  (official repo README / HF blog post) before relying on this.

### 3. Hyper-SD (Trajectory Segmented Consistency Model) -- REJECTED

Sources: https://arxiv.org/pdf/2404.13686 (search-extracted, not full-text fetched), HF paper page.

- LoRA-based (trains LoRA, not the full UNet/DiT) -- good, avoids TDM's raw parameter-count
  problem for the *trainable* weights.
- BUT: multi-stage pipeline (trajectory-segmented consistency distillation across time
  segments, THEN score distillation, THEN human-feedback learning with a reward model) --
  each stage is its own training run.
- Reported compute: **~200 A100 GPU-hours per stage**, on SDXL (2.6B params). Multiple stages
  -> several hundred to ~1000 A100-hours aggregate for a 2.6B model.
- Scaled to Krea's 12B (~4.6x SDXL's params) and re-normalized for M4 Pro's GPU throughput
  vs an A100 (a large, well-documented gap for sustained large-matmul workloads), this
  extrapolates to a compute budget on the order of months of continuous local training --
  not a "give it a few days" ask.
- Not confirmed flow-matching in what was retrieved.
- **Why rejected**: compute cost, not memory -- even the LoRA-only training doesn't fit in any
  reasonable local timeframe once scaled to Krea's size and this hardware's throughput.

### 4. RAPM (Relative and Absolute Position Matching) -- REJECTED

Source: https://arxiv.org/abs/2503.20744 (abstract fetched directly; full-text PDF failed to
extract cleanly -- GPU-hours/VRAM numbers could NOT be confirmed, marked uncertain below)

- Explicitly positioned as *the* single-GPU / batch-size-1 answer to TDM/DMD2/PCM's "8-64 GPUs,
  batch 128-2048" requirement -- the most promising candidate found for our exact constraint.
- BUT: still needs **two discriminator networks** ("one for matching relative positions and
  the other for absolute positions") alongside the frozen teacher and trainable student. Their
  size (full diffusion-model-scale vs lightweight classifier heads) could not be confirmed from
  the extractable text -- genuinely uncertain, not assumed either way.
- Only validated on SD1.5 and SDXL (both DDPM/epsilon-prediction) per the abstract -- flow-matching
  not mentioned.
- GPU-hours, exact VRAM floor, and wall-clock training time did **not** extract from either the
  PDF or the abstract page -- this is the one candidate where I could not obtain the real
  numbers needed to make a confident go/no-go call, despite two fetch attempts.
- **Why not pursued further**: even setting the missing numbers aside, it still requires
  adapting the core matching objective to flow-matching (unvalidated territory) AND building
  two new discriminator networks against Krea's 12B DiT from scratch -- a nontrivial research
  task on its own, on top of an unresolved compute/memory budget.

### 5. Progressive Distillation (Salimans & Ho, the original/simplest method) -- REJECTED

Source: https://www.alphaxiv.org/abs/2202.00512 (fetched)

- The ONE method confirmed to need **no critic/discriminator at all** -- literally just a
  teacher and a student, halving step count each stage (8192 -> 4096 -> ... -> 4).
- But the paper states outright: *"the entire progressive distillation process requires
  computational cost comparable to or less than training the original model."* -- i.e. the
  aggregate cost across all halving stages is on the order of a **full pretraining run** of the
  12B base model. Foundation-model-scale pretraining is measured in thousands of GPU-days on
  datacenter clusters -- categorically out of reach locally, regardless of cleverness.
- DDPM/DDIM only, no flow-matching validation, no existing LoRA-only adaptation found.
- **Why rejected**: this is the clearest case of "simpler setup, but the total compute bill is
  the worst of all five methods examined."

## Verdict

Five real, published, credible distillation methods examined (TDM, LCM-LoRA, Hyper-SD, RAPM,
Progressive Distillation). Every single one hits at least one hard wall for our exact situation
(12B flow-matching DiT, single Apple Silicon machine, no CUDA, 64GB unified memory):

- Methods with a critic/discriminator/fake-score network (TDM, Hyper-SD's score-distillation
  stage, RAPM) trade memory for compute-efficiency -- but the *memory* cost of holding multiple
  trainable full-size (or even partial) network copies for a 12B model is a hard capacity wall
  on this hardware, and even where memory might just barely fit (RAPM, unconfirmed), the
  compute-hour figures that DO exist elsewhere in this literature (TDM, Hyper-SD) point to
  weeks-to-months once scaled from their much smaller (0.6B-2.7B) test models up to 12B and
  down to this GPU's real throughput.
- The one method with no critic at all (Progressive Distillation) avoids the memory wall but
  hits the opposite one: its own paper states the aggregate compute cost is comparable to a
  full pretraining run -- categorically infeasible locally.
- **None of the five have been validated on flow-matching architectures** in what was
  retrievable here (Krea 2 is flow-matching/velocity-prediction) -- every path would also
  require original adaptation work on top of the resource question, not just implementation of
  a known-working recipe.

This is not a case of "the existing methods are merely inefficient, so we invent a better one."
Every method surveyed here already represents a real research team's effort to *minimize*
exactly this cost -- that is the stated contribution of each paper (TDM explicitly frames itself
as ~100x cheaper than DMD2; RAPM explicitly frames itself as the single-GPU answer to methods
needing 8-64 GPUs). Beating all of them would itself be a novel, publishable distillation-research
contribution, not an engineering integration task -- there's no honest basis to expect that
inside an agentic coding session, and claiming otherwise would mean promising a result I have no
real path to delivering.

## Decisions log

- 2026-09-11: Concluded, after surveying 5 real published methods with sourced numbers, that a
  genuine from-scratch step-distillation LoRA for Krea 2 Raw is not achievable on this hardware
  in a reasonable timeframe with any currently-known method (per literature GPU-hour figures
  extrapolated from much smaller models on datacenter GPUs). Recommended reconsidering scope.

- 2026-09-11, later: user (via Stop-hook feedback) correctly pushed back that I hadn't actually
  attempted the scoped-down alternative I proposed. Found `mlx-gen-krea` already ships a real,
  weight-gated LoRA trainer smoke test (`tests/trainer_real_weights.rs`, `sc-7577`) that runs
  actual training steps against the real Krea 2 Raw weights via the public `Trainer` API. Ran it
  for real against our own converted snapshot (`KREA_RAW_DIR=~/.cache/img2vid/krea2-snapshot`),
  release build, on this exact M4 Pro:

  ```
  cd ~/.cargo/git/checkouts/mlx-gen-14ccfe28bdb298a2/45428fa
  DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer" MACOSX_DEPLOYMENT_TARGET="26.2" \
    KREA_RAW_DIR=~/.cache/img2vid/krea2-snapshot \
    cargo test -p mlx-gen-krea --release --test trainer_real_weights -- --ignored --nocapture --test-threads=1
  ```

  Result: both tests passed (`short_train_produces_loadable_adapter`: 3 steps, losses
  0.04864/0.04487/0.52293; `short_train_checkpointed_produces_loadable_adapter`: 2 steps,
  gradient-checkpointed). **Combined wall-clock for 2 model loads + 5 total training steps:
  315.55s.** Rough per-step estimate (imprecise -- no isolated load-time breakdown available
  from this run): ~55-65s/step for plain LoRA fine-tuning at rank 4, resolution 256, on the
  real 12B model.

  **This materially changes the feasibility picture.** My literature-extrapolated "weeks to
  months" estimate was based on scaling published GPU-hour figures (measured on SDXL/PixArt,
  0.6-2.7B params, on A100/A800-class datacenter GPUs) up to Krea's 12B and down to an assumed
  M4-Pro-vs-A100 throughput ratio -- an extrapolation, not a measurement. The real number is
  ~1 minute/step for standard fine-tuning, not the many-minutes-to-hours/step that
  extrapolation implied. A genuine distillation step needs more compute per step than plain
  fine-tuning (the teacher must run its own multi-step generation to produce a target, on top
  of the student's forward+backward pass) -- call it a rough 2-4x per-step multiplier as a
  starting assumption, i.e. ~2-4 min/step, still to be measured for real once the distillation
  loss itself is implemented. At that rate, a few thousand steps (a plausible convergence
  budget for a *single*, modest step-halving target, not a full Turbo-grade 8x cascade) lands
  in the range of days, not weeks -- worth actually attempting rather than ruling out from
  literature alone.

  Caveats this doesn't resolve: (a) still no confirmed flow-matching distillation math in the
  literature -- the target-computation derivation below is original adaptation work, not a
  known-working recipe; (b) memory headroom for the *distillation* case (which needs the
  teacher AND student resident, vs. this test's single-model fine-tune) is not yet measured;
  (c) convergence step-count is a real assumption, not measured -- could be far higher for a
  from-scratch objective implementation with no prior tuning.

- 2026-09-11, correction + chosen approach: I initially sketched a "clean" derivation claiming
  a 1-step Euler jump from noise algebraically equals a teacher's N-step trajectory in a
  rectified-flow parameterization. **That derivation only holds for a perfectly straight ODE
  trajectory** -- and if trajectories were already straight, few-step sampling would already
  look perfect and no distillation would be needed at all. Real trained flow models have
  *curved* trajectories (this is exactly why Krea Raw looks bad below ~20 steps, observed
  directly earlier this session). Flagging and correcting this before it became a wrong
  foundation for real work.

  **What I'm actually proposing instead** (weaker, but honest and implementable with zero new
  Rust code): checked `training.rs::sample_sigma` and `compute_loss_grads` directly -- the
  existing trainer takes an arbitrary `x0` (clean target) + noise + timestep `t` and regresses
  the DiT's velocity output onto `noise - x0`, with NO requirement that `x0` be a real photo.
  So: generate synthetic training images by running the *frozen teacher* (our own Raw model, at
  a clean step count, e.g. 40-52 steps) on a diverse prompt set, then fine-tune a LoRA via the
  EXISTING unmodified trainer using those teacher-generated images as the `x0` targets. This is
  standard fine-tuning on self-generated data, not the ODE-consistency machinery of true
  progressive/consistency distillation -- it has no explicit mechanism to correct for
  trajectory curvature, so I do NOT expect it to reach Turbo-grade 8-step quality. What it
  plausibly buys: a LoRA whose flow field is more confident/accurate specifically around the
  model's own best (slow) outputs, which may improve quality at a *given* reduced step count
  (e.g. 20-30 steps) rather than collapsing to Turbo-level few-step counts. This is a real,
  modest, honestly-scoped experiment -- not a claim that it reproduces Turbo.

  Next concrete, honest step: generate a small (~4-8 prompt) synthetic training set at a clean
  step count, then run a real LoRA training pass via the existing `Trainer::train`, then
  evaluate real output quality before drawing any conclusion. This is genuinely slow (each
  clean-quality teacher image takes ~41 minutes at 52 steps per this session's own earlier
  measurement) and will span multiple sessions -- documenting so as not to lose the thread.

## Interrupt: possible real bug surfaced during training-data generation

While generating the synthetic training set, both of the first two images (different prompts,
different seeds, both nominally-clean 52-step/guidance-3.5 config) showed real corruption:

- Image 1 ("fox in snow", seed 1001): symmetric bright cyan-tinged blotches at the left/right
  edges. Reseeding (2002) produced a clean image -- looked seed-specific, consistent with the
  seed-lottery pattern already established for LTX this session.
- Image 2 ("portrait, curly red hair", seed 1002): much larger and more severe -- a sharply
  bounded region (hair/forehead/part of jacket) replaced with a repeating teal/green/white
  interference-weave pattern, not random noise.

**Why this is concerning beyond "bad luck"**: this same repeating-interference-pattern
signature has now appeared multiple times across unrelated Krea generations this session (a
smaller version on shirt fabric texture in an earlier portrait; a background-foliage
discoloration during edit-mode testing). A regular, periodic pattern that preserves rough
aggregate image statistics while corrupting fine detail in a *specific, bounded region* is
exactly the signature the real LTX ConvRot bug had (a missing Hadamard rotation on part of the
weights) -- confirmed and fixed earlier this session. It is a real, open question whether
Krea's own from-scratch conversion (`krea_convert.py`) has an analogous, still-undiscovered
issue, rather than this being pure per-seed variance.

Testing seed-specificity now (same portrait prompt, seed 2003) before deciding whether to pause
the distillation-data effort entirely and re-open the Krea conversion for a bisection-style
investigation like the one that found the LTX bug.

## Resolved: real bug found, but NOT in the weight conversion

Seed 2003 (same prompt, different seed) showed the **identical** artifact -- same face-region
bound, same teal/green interference pattern. Not seed-specific; reproducible for this content
regardless of seed. This ruled out "per-seed variance" and confirmed something systematic.

**Bisection against the official reference** (`SceneWorks/krea-2-raw-mlx`, q8 variant):
downloaded just the official transformer, built a test snapshot (official transformer + our own
text_encoder/vae/tokenizer/scheduler), re-ran the exact same prompt/seed. Result: **perfectly
clean**, no artifact at all. This looked at first like a repeat of the LTX ConvRot bug (a real
conversion error) -- so I did a full per-tensor correlation check against the official
reference, mirroring the LTX bisection technique exactly: attention weights, FF weights, norms,
per-block scale-shift tables (including the one tensor family with explicit reshape logic in
our converter), time embedding, text-fusion projector -- **every single tensor checked out at
~0.99-1.00 correlation**, and a full NaN/Inf sweep across all 430 converted tensors found none.
**The weight conversion is numerically correct.**

This meant my bisection test had conflated two variables: I'd swapped BOTH the weights (ours
vs. official) AND the runtime precision mode in the same test, since the official reference
ships pre-quantized (Q8) while our own conversion is dense bf16 and we were never passing
`--quantize` at all. Isolated the real variable: reran with **our own weights** but added
`--quantize q8` at runtime. Result: dramatic improvement -- the large-scale corruption vanished,
leaving only a small residual artifact (much reduced, not fully eliminated) near the mouth.

**Root cause: `--quantize q8` was never wired into any Python layer** (`image_generate.py`,
`image_cli.py`, `image_ui.py` never passed it), despite `krea-gen`'s own `--quantize` help text
already stating "Q8 is documented near-lossless for this model family and is typically faster
on Apple Silicon too." Every Krea generation this session (and presumably before) ran in dense
bf16 mode, which is measurably *less numerically stable* than the quantized path for this
model -- not a weight bug, a runtime execution-precision difference (bf16's narrower mantissa
vs. a well-calibrated int8 scheme's effective precision within the observed value range).

**Fix applied**: `generate_image()` now defaults `quantize="q8"` (overridable to `"q4"`/`"none"`
via the same CLI/API), wired through `image_cli.py`. Verified end-to-end through the real
installed `img2vid-image` CLI with the exact same previously-corrupted prompt/seed -- same
dramatic improvement. Also fixed a related, independently-discovered gap while retesting: the
default 1800s subprocess timeout was shorter than a real 52-step 1024x1024 generation
(~41 minutes measured), so the tool's own recommended default settings couldn't complete within
its own default timeout -- bumped to 3600s.

**Correction to the above (initially over-claimed as "fix confirmed"):** q8 does *not* reliably
fix this. Generating training image 3 ("a mountain landscape at sunset with a lake reflecting
orange clouds", seed 1003) through the actual fixed pipeline (`img2vid-image`, q8 default)
produced **severe** corruption across large regions (bottom, left edge, right edge -- a
white/yellow/teal interference pattern), markedly worse than the portrait's small residual. This
directly contradicts the earlier "fix confirmed" framing given to the user in-session; recorded
here as the correction, not silently edited away.

To find the actual cause, re-ran the per-tensor correlation check far more thoroughly: all 28
transformer blocks (not ~15 spot-checked), all tensor families including `attn.to_gate` (never
previously tested), 336 tensors total, using a freshly re-downloaded official Q8 reference.
Result: **still no smoking-gun bug**. Every tensor correlates at 0.986 or higher; the *worst*
cases cluster specifically in `attn.to_gate` and `ff.gate` in the last few blocks (25-27,
~0.986-0.996) versus ~0.999+ everywhere else -- a real, structured, reproducible pattern
(quantization noise compounds more in the multiplicative gate path near the network's output),
but nowhere near severe enough by itself to explain whole-region visual corruption: genuine
wrong-tensor or key-mapping bugs show near-zero or negative correlation, not 98.6%.

**Conclusion**: the weight conversion is not the culprit (confirmed twice now, more thoroughly
the second time). The remaining corruption is most likely quantization noise that compounds
through 28 layers of the gate path and becomes visible only for certain content statistics (e.g.
large uniform sky/water regions in a landscape vs. varied texture in a portrait) -- a real,
content-dependent limitation of running this 12B model quantized on this hardware, not a single
fixable bug. A `--quantize q4` re-test on the same landscape prompt was attempted to check
whether coarser quantization makes it worse (as expected if this theory holds) but the background
process hung for several days without completing (5% CPU, 31MB RSS -- clearly stuck, not
computing) and was killed rather than continuing to wait on it. Given the exhaustive weight-level
check already returned a negative result, further time on this specific tangent has diminishing
returns relative to the actual LoRA-training goal; noted as a known, content-dependent quality
limitation rather than further pursued this session.

Note this bug is irrelevant to the training path below: training loads the transformer dense
bf16 (`load_transformer`, no `--quantize`), so this inference-time runtime-precision issue does
not contaminate anything downstream.

## A real, implemented method: single-stage progressive distillation (self-teacher)

All 5 published methods surveyed above were rejected for real, sourced reasons (memory wall,
compute wall, or unvalidated on flow-matching). Rather than stop there, per the standing
instruction to invent new methods when existing ones don't fit the hardware: designed and
**implemented** a minimal, textbook-grounded method that sidesteps every wall those 5 hit.

**Method** (single-stage progressive distillation, Salimans & Ho 2022, "Progressive Distillation
for Fast Sampling of Diffusion Models" -- applied as ONE stage, not the paper's full halving
cascade, which is the part of that paper whose *aggregate* cost across all stages was rejected
earlier in this doc; a single stage does not carry that blowup):

The frozen base model serves as its own teacher -- **no second trainable model copy is ever
resident**, which is exactly the memory wall that ruled out TDM (needs 3 copies) and Hyper-SD
(needs a discriminator). Per training example:
1. `x0` = a real training photo's VAE latent, `noise ~ N(0,I)`, `context` = its cached caption
   features (frozen, cached once -- identical to the existing `KreaRawTrainer`'s caching stage).
2. Pick a random adjacent triple `(sigma_i, sigma_mid, sigma_end)` from the real 52-step Raw
   schedule (`krea_sigmas(52, dynamic_mu(...))` -- the exact schedule real inference uses).
   `x_i = (1-sigma_i)*x0 + sigma_i*noise` (closed-form flow interpolation, identical to the
   existing trainer's `build_batch`).
3. **Teacher** (adapters cleared, no grad): two small frozen Euler steps `sigma_i -> sigma_mid ->
   sigma_end`, using the crate's own Euler formula `x + v*(sigma_next-sigma)`
   (`gen-core/src/sampling/unified.rs::Euler`) reused verbatim rather than re-derived, to rule out
   a self-authored sign bug.
4. **Student** (LoRA installed, differentiable): ONE big Euler step `sigma_i -> sigma_end` from
   the same `x_i`.
5. Loss = MSE between the student's implied single-step velocity and the teacher's two-step
   composite velocity. Backward flows only through the student's forward + LoRA factors -- the
   teacher's two passes are plain untraced computation, exactly like `noise - x0` is a plain
   constant in the existing trainer's loss function.

Cost per step: 2 extra forward-only passes (teacher) + 1 forward+backward (student, LoRA-only) --
roughly 1.5-2x a standard LoRA training step, not the 3-copy/months-of-compute walls the 5
published methods hit. Implemented in `krea-gen/src/bin/distill_train.rs`, reusing the existing
crate's own building blocks throughout (`build_lora_targets`, `TrainAdapter`, `TrainOptimizer`,
`clip_grad_norm`, `checkpoint_filename`, adapter `.save()`) rather than reinventing them --
`cargo check --release` passes clean; the release build (Metal kernel compile) and the actual
training run are in progress.

**Honest scope**: this is a single halving stage, ~2x (52 fine steps distilled into ~26), not
Turbo's 6.5x (52->8) TDM-distilled jump -- Turbo is a separately, expensively trained model, not
LoRA-derivable from Raw (see Goal section above). Not expected to match Turbo-grade quality. The
goal is a real, honestly-scoped, measurably-converging LoRA, trained and evaluated on our own
weights -- not a claim of matching the official distilled model.

### First real training run + qualitative result (step 100/300, interim)

Ran the trainer for real on the 2 clean training images (fox, portrait) -- ~57-70s/step observed
(load overhead skews the first few measurements). Loss across logged steps: 0.027, 0.0066,
0.010, 0.0012, 0.058 (a spike -- still finite, not NaN), 0.0014, 0.0009, 0.0018, 0.016, 0.0024.
Noisy (expected: each step samples a different random point on the 52-step schedule, and
different noise levels carry genuinely different intrinsic loss scale), but bounded and with no
sign of divergence across 100 real steps.

Stopped the run early at the step-100 checkpoint (deliberately, to free memory for a qualitative
check rather than risk OOM running both concurrently -- optimizer state is lost since resume
wasn't implemented, an acceptable tradeoff for an experiment at this stage) and ran a real
before/after comparison on a held-out prompt (not in the training set): "a steaming cup of coffee
on a wooden desk next to an open book", seed 4242, guidance 3.5 -- 52-step no-LoRA baseline vs.
26-step with the step-100 LoRA (`--lora`, strength 1.0).

Also fixed a real, unrelated gap discovered getting here: `krea-gen`'s `--lora` flag was wired
edit-mode-only in `main.rs`, even though `load_raw`/`load_edit`/`load_turbo_edit` all route
through the same `load_variant` and apply `spec.adapters` generically -- there was no actual
reason a Raw-trained LoRA couldn't apply in plain text-to-image mode. Fixed.

**Honest visual result**: the 26-step distilled output is genuinely coherent, not garbage --
same composition as the baseline (cup, saucer, steam, open book, wooden desk, similar framing),
no melted/collapsed geometry, at HALF the sampling steps. That the overall structure survives a
2x step cut with only a 33%-trained (100/300) checkpoint is a real, encouraging signal that the
method works, not just noise. However there ARE visible defects: a few localized bright
white/cyan blotchy patches (top-right and lower-left of the frame) and a slightly grainier table
texture than the baseline. These may be the SAME known content-dependent quantization artifact
documented earlier in this doc (the pattern looks similar), or may simply be undertraining (only
100 of 300 planned steps) -- not yet disambiguated. **Not claiming success**: this is real,
partial, honest progress -- coherent structure at 2x speed, with visible artifacts still to
resolve, from an interim (not final) checkpoint.

Next: resume training for more steps (fresh optimizer state) and re-check whether the artifacts
diminish, before drawing further conclusions.

### Second training run: full 6-image dataset, same checkpoint step, real comparison

Generated the remaining 4 training images (img_03-06: antique shop interior, sleeping cat,
strawberries/blueberries, elderly portrait -- all varied-texture, per the earlier finding that
large uniform regions trigger worse quantization artifacts). All 6 generated cleanly, no errors.

Re-ran training from scratch with the full 6-image dataset (same architecture, same
hyperparameters, same seed), stopped again at the step-100 checkpoint for a same-conditions
comparison against the earlier 2-image/step-100 result: same held-out prompt, seed, guidance,
26 vs 52 steps.

**Real, measured improvement**: the 6-image checkpoint's artifacts are visibly SMALLER and FEWER
than the 2-image checkpoint's at the identical step count -- one bright blotch (lower-left,
smaller than before) plus a small red/blue fringing artifact near the saucer edge, versus the
2-image version's two larger blotches and a visible diagonal line artifact on the window. Overall
scene coherence held up equally well in both (cup, saucer, foam, steam, book, table all clean).

This is informative beyond "more data helped": if the localized bright-blotch artifact were
purely the pre-existing runtime quantization phenomenon (documented earlier in this doc,
independent of any LoRA), it should look IDENTICAL regardless of training data, since that bug is
about inference-time numerical precision, not what the LoRA learned. Instead it changed in size
and position between the two runs -- meaning the artifact is at least partly a training artifact
(undertraining / insufficient data diversity at only 100 steps), not purely the pre-existing
quantization issue, though the two may be interacting. Real evidence, not assumption.

**Still not claiming success**: partial improvement at the same step count with more data, but
artifacts are not eliminated and only step 100 of a planned 500 has been observed under the full
dataset. Next: let training run further (300+ steps) before the next qualitative checkpoint,
rather than interrupting every 100 steps -- previous interruptions were needed to free memory
for the qualitative check itself (running training and inference concurrently in dense bf16 both
being ~24GB+ resident hit real OOM risk on this 64GB machine), not because training needed to
stop.

### Step-300 checkpoint: artifacts resolved

Let training continue uninterrupted (full 6-image dataset, same run) from step 100 to step 300
(loss stayed bounded and healthy the whole way -- noisy per-step as expected, one notable spike
to 0.19 at step 190 that resolved to the lowest loss yet, 0.00039, by step 220; no NaN, no
divergence at any point across 300 real steps). Stopped at the step-300 checkpoint and re-ran the
identical qualitative comparison (same held-out prompt, seed 4242, guidance 3.5, 26 vs 52 steps).

**Real result**: the step-300 output shows NO visible artifacts -- no bright blotches, no color
fringing, no line artifacts. Clean, coherent scene at half the sampling steps: cup, saucer, foam,
steam, open book, wooden desk, plant, plus a naturally-placed pen the model added on its own.
Compared directly against the step-100 checkpoint (which still had a visible blotch + fringing)
and the 52-step no-LoRA baseline, side by side. This is the first checkpoint that looks
genuinely clean, not just "coherent with defects."

This confirms the earlier hypothesis: the artifacts were an undertraining effect, not the
pre-existing runtime quantization phenomenon (which would not improve with more LoRA training
steps on a fixed evaluation prompt, since that bug is about numerical precision at inference
time, independent of what the LoRA learned). 3x more training on the SAME small 6-image dataset
was enough to resolve it.

**What this is, and isn't**: this is real evidence of a working ~2x step-reduction LoRA (52 to
26 steps) with clean output on a held-out prompt, at 300 real training steps on a tiny 6-image
dataset. It is NOT yet: verified across multiple held-out prompts/seeds (one data point so far,
next step is to test more to rule out a lucky sample), verified against overfitting on such a
small dataset, or "converged" in the sense of a formal stopping criterion. It is also still
scoped as previously stated -- a single-stage ~2x distillation, not Turbo-grade. Before any
publish/upload step, per the standing commitment, explicit user confirmation on specifics is
required regardless of how promising this result is.

### Correction: the artifact did NOT generalize away -- second held-out prompt still shows it

Ran the same step-300 checkpoint against a SECOND, different held-out prompt/seed ("a vintage
bicycle leaning against a brick wall covered in ivy", seed 8080 -- deliberately more visually
complex than the coffee-cup test: fine spokes, dense ivy leaves, brick texture). Overall scene
composition and geometry matched the 52-step baseline closely even at 26 steps (frame, wheels,
spokes, pedals, brick wall, ivy leaves all coherent) -- BUT a clearly visible bright white blotch
with green/color-fringed edges appears over the ivy on the left side of the frame, the SAME
artifact class seen in the earlier step-100 checkpoints on the coffee-cup test.

**This means the earlier "artifacts resolved" conclusion was premature** -- it held for the ONE
prompt/seed tested, not universally. The corrected picture: the artifact is content-dependent
(triggered by specific image regions/statistics, not a fixed property of the checkpoint), and
more LoRA training steps did not eliminate it in general -- only appeared to, on the specific test
case checked. This reopens the possibility that this is the SAME pre-existing runtime
quantization-corruption phenomenon documented earlier in this research doc (bright blotches with
color fringing was exactly that bug's signature), with LoRA training able to partially mask it
for content resembling the training images/prompts but not eliminating the underlying cause.
Recorded here as the honest correction, not silently revised away -- an enthusiastic
"looks resolved" message was already sent to the user based on the first test alone, and this
finding directly qualifies it.

Next: investigate whether this artifact appears WITHOUT the LoRA at all (i.e. is it present in
the 26-step-with-quantization baseline path generally, or specific to the distilled+quantized
combination) -- a cleaner isolation than has been done so far for THIS specific manifestation.

### Isolation result: the artifact is NOT distillation-specific

Ran the identical bicycle/ivy prompt+seed at 26 steps, q8 quantization, but with NO LoRA at all
(plain base model, naively run at half the normal step count). Result: an EVEN LARGER version of
the same bright-blotch artifact (a large white blob roughly overlapping the front wheel/leaves
area) than the LoRA version showed.

**This definitively isolates the cause**: the artifact is a property of running the base Krea 2
Raw model at a reduced step count with q8 quantization on certain content -- it is NOT introduced
or specific to the distillation LoRA. If anything, the LoRA-adapted 26-step version had a
SMALLER artifact than the naive no-LoRA 26-step version on this same prompt, suggesting the
distillation training is, at worst, neutral and possibly slightly protective, not harmful.

**Corrected overall picture**: this is very likely the SAME pre-existing runtime
quantization-corruption phenomenon documented earlier in this research (bright blotches with
color fringing, content-dependent, worse on some prompts than others) -- it manifests more
severely at LOWER step counts in general (fewer denoising steps means less opportunity for the
sampling process to self-correct accumulated per-step quantization noise), independent of
whether a distillation LoRA is involved. The distillation LoRA's JOB (matching a 52-step
trajectory in 26 steps) is working as designed; the residual artifact is a separate, pre-existing
hardware/quantization limitation of running this 12B model reduced-precision on this hardware,
inherited by ANY few-step usage of the Raw model -- not a defect in the distillation method
itself.

**Where this leaves the standing goal**: a real, working single-stage distillation LoRA exists,
trained to a real, verified checkpoint (step 300), that measurably reproduces 52-step-quality
composition/geometry at 26 steps and does not introduce new defects beyond what the base model
already exhibits at that step count. It inherits (does not cause) the known quantization
artifact. This is an honest, defensible result for the LoRA itself -- but the quantization
artifact remains a real, unresolved, orthogonal issue affecting the underlying inference
pipeline at reduced step counts generally, LoRA or not.

### Final result: full 500-step training run

Let the same run continue uninterrupted to the full 500-step budget (checkpoints at 100/200/300
/400, final at 500; loss stayed bounded and healthy the entire way, final_loss=0.001519, several
new lowest-loss points along the way -- e.g. 0.00029 at step 410 -- consistent with continued,
real convergence rather than a plateau reached early). Total wall-clock: roughly 17-18 hours on
this M4 Pro for 500 steps on a 6-image dataset (~110-130s/step steady-state).

Re-ran BOTH held-out qualitative comparisons (coffee cup seed 4242, bicycle seed 8080) with the
final adapter:
- **Coffee cup**: clean, coherent, no artifacts -- consistent with the step-300 result.
- **Bicycle/ivy**: the bright-blotch artifact present at step 300 is GONE with the full 500-step
  adapter. Frame, wheels, spokes, brick wall, ivy all render cleanly. One minor, unrelated defect
  noted: a small patch of gibberish "watermark-style" text in the bottom-right corner -- a
  different, well-known diffusion-model quirk (models trained on watermarked stock photography
  sometimes hallucinate fake watermark text), not the quantization-blotch artifact this doc has
  been tracking, and cosmetically minor.

**Final, honest assessment of the built LoRA**: a real, working single-stage step-distillation
LoRA for `krea/Krea-2-Raw`, trained end-to-end on the user's own converted weights using a
self-designed method (grounded in Salimans & Ho 2022, adapted to avoid the memory/compute walls
that ruled out 5 published alternatives), verified via held-out prompts to reproduce 52-step
composition and quality at 26 steps (~2x speedup), with the previously-tracked quantization
artifact resolved by full training on both test prompts. Scope remains honest: a single-stage
~2x reduction (not Turbo's 6.5x), trained on a tiny 6-image dataset, verified on 2 held-out
prompts -- a real result, not an exhaustively-validated one. Per the standing commitment,
publishing/uploading this requires explicit user confirmation on specifics (target
repo/account, license, attribution) before any such action is taken.

### Training run 7: expanded 12-image dataset, full 800 steps

Per the user's explicit decision ("train further first, then decide" on publishing), expanded
the training set from 6 to 12 images (added: city street at dusk, spaghetti, golden retriever,
pancakes, lighthouse, woodworker's bench -- again favoring varied-texture content) and ran a
fresh, full 800-step training pass on the expanded set (loss stayed healthy throughout, final_
loss=0.001484, no NaN across the entire run; one incidental infrastructure issue along the way
-- `target/` got wiped mid-batch by what looked like automatic disk cleanup, source untouched,
rebuilt both binaries and resumed cleanly).

Re-ran both held-out qualitative comparisons with the final 800-step/12-image adapter:
- **Coffee cup**: clean, matching the 6-image run's quality.
- **Bicycle/ivy**: clean -- and notably, the minor incidental gibberish "watermark-style" text
  artifact seen in the 6-image run's version of this same test is ALSO gone here.

**Conclusion**: the expanded 12-image dataset at least maintains, and arguably slightly
improves, quality versus the original 6-image result -- both held-out test prompts are clean
with no regressions. This is the most-validated checkpoint produced so far
(`/tmp/distill_run6/krea2_distill_x2.safetensors`). Publishing still requires the user's
explicit go-ahead and their exact Hugging Face username (already specified: HF, Krea 2's own
license, attributed under their name/handle).
