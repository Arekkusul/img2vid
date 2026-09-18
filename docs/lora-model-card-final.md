---
license: other
license_name: krea-2-community
license_link: https://huggingface.co/Arekkusul/krea-2-raw-distill-lora/blob/main/LICENSE.pdf
base_model: krea/Krea-2-Raw
tags:
  - lora
  - text-to-image
  - krea-2
  - distillation
---

# Krea 2 Raw — Single-Stage Step-Distillation LoRA

A LoRA adapter for [`krea/Krea-2-Raw`](https://huggingface.co/krea/Krea-2-Raw) (the undistilled
12B single-stream flow-matching DiT behind Krea 2) that halves the sampling step count —
**52 steps → 26 steps** — while reproducing the full-quality composition and detail of the
52-step baseline, verified on held-out prompts never seen during training.

This was trained on a from-scratch conversion of a Krea 2 Raw checkpoint, for anyone in the same
situation: **you have a converted Raw checkpoint but not the official, separately and
expensively trained Turbo checkpoint**, and want a faster path without needing that separate
model.

## What this is NOT

- **Not Krea 2 Turbo.** Turbo is a genuinely separate, TDM-distilled model (8 steps, CFG-free) —
  not something a LoRA can reach from Raw. This LoRA is a single-stage, ~2x step reduction, not
  Turbo's ~6.5x jump. Don't expect Turbo-grade speed or quality.
- **Not exhaustively validated.** Verified on 2 held-out prompts (a coffee-cup still life, a
  detailed bicycle/brick/ivy scene) across two training runs (6-image and 12-image datasets).
  That's real evidence the method works, not a comprehensive quality guarantee across all
  content types, styles, or resolutions.

## Method

Single-stage progressive distillation (Salimans & Ho 2022, *"Progressive Distillation for Fast
Sampling of Diffusion Models"*), adapted to run on a single Apple Silicon Mac with no second
trainable model copy resident in memory:

1. A real training photo's VAE latent (`x0`) + random noise define a point `x_i` on the
   flow-matching trajectory at a random noise level.
2. The **frozen base model itself** (no LoRA) takes two small reference Euler steps from `x_i`
   forward — this is the "teacher" signal, computed fresh every training step, no separate
   teacher model needed.
3. The **LoRA-adapted model** takes one big Euler step covering the same span — the "student."
4. Loss = MSE between the two, backprop only through the student's LoRA factors.

This method was designed after five published few-step-distillation methods (TDM, Hyper-SD,
RAPM, LCM-LoRA, Progressive Distillation's full cascade) were researched and ruled out as
infeasible on consumer Apple Silicon hardware — each hits a memory wall (needing 2-3 resident
model copies), a compute wall (hundreds of GPU-hours), or is unvalidated for flow-matching
architectures.

## Training details

- **Base model**: a `krea/Krea-2-Raw` conversion (12B, hidden=6144, 28 single-stream blocks,
  GQA 48Q/12KV heads).
- **LoRA target modules**: `to_q`, `to_k`, `to_v`, `to_out.0` on all 28 transformer blocks
  (112 targets total) — the standard PEFT attention surface.
- **Rank**: 16, **alpha**: 16.
- **Optimizer**: AdamW, learning rate 1e-4, gradient norm clipped to 1.0.
- **Training data**: 12 self-generated 1024x1024 images spanning varied subject matter and
  texture (portrait, animal, food, landscape, interior, still life) — no external/copyrighted
  training images.
- **Steps**: 800, batch size 1, ~110-130s/step on an Apple M4 Pro (64GB unified memory).
  Final training loss: 0.0015 (bounded, noisy-but-healthy throughout — flow-matching loss
  varies naturally by sampled noise level, not a sign of instability).
- **Hardware**: entirely trained on a single Mac mini M4 Pro, no cloud GPU.

## Usage

Diffusers/PEFT-format LoRA (standard `{path}.lora_a`/`{path}.lora_b` naming, `alpha`/`rank`
metadata) — load with any Krea 2 Raw-compatible LoRA loader. Recommended: 26 steps, guidance
scale 3.5 (matches training conditions) — other step counts/guidance are untested.

## License

Inherits the Krea 2 Community License from the base model (`krea/Krea-2-Raw`) this LoRA is a
derivative of — free for personal/non-commercial use, with the base license's content-filtering
obligation for deployers. Full text: [`LICENSE.pdf`](LICENSE.pdf) in this repo (copied verbatim
from the base model).

## Known limitations

- Trained on a small (12-image) dataset — may not generalize to all content types equally well.
- Verified on 2 held-out prompts only.
- The base `krea/Krea-2-Raw` model has a separate, pre-existing content-dependent artifact at
  reduced step counts (unrelated to this LoRA — confirmed via isolation testing to occur, and be
  worse, even WITHOUT this LoRA). This LoRA does not cause it and did not worsen it in testing,
  but it's a base-model characteristic worth knowing about.
