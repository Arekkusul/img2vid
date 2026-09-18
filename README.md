# img2vid

Two independent local generation tools, both on-device via [MLX](https://github.com/ml-explore/mlx)
— no cloud API, no CUDA required:
- **Text-to-video / image-to-video**: give it a prompt (and optionally a starting image), get
  back a generated video clip.
- **Text-to-image / image-edit** (standalone, see below): give it a prompt, get back an image;
  optionally give it a reference image too for editing.

## Combined web UI

`scripts/run_studio_ui.sh` (or `img2vid-studio-ui`) launches both tools in one tabbed Gradio app
— a "Video" tab and an "Image" tab, so you can switch between them without running two separate
servers. It's a thin composition (`gr.TabbedInterface`) of the two standalone UIs described
below, which still work independently if you only need one.

## Video: Model

[LTX-2.5](https://github.com/Lightricks/LTX-2) (Lightricks), a 19B-parameter joint audio+video
diffusion transformer, converted from the user's own ComfyUI-exported fp8 checkpoint to MLX via
a from-scratch conversion (`src/img2vid/ltx_convert.py`) and run through
[`ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) (a separate, `uv`-managed MLX-native
inference runtime cloned into `ltx-2-mlx-upstream/`).

**Unlike a pre-packaged model download, this project's video weights are converted locally from a
user-supplied checkpoint** — there's no `download_model.sh` step. The source file uses a mixed
w4a8-codebook + int8-tensorwise quantization scheme (ComfyUI/comfy-kitchen's own format); the
conversion logic (dequantization, key remapping, MLX re-quantization) was derived by reading the
real Apache-2.0-licensed `comfy-kitchen` source, not guessed — see `WORKING-CONTEXT.md` for the
one real dequantization bug this surfaced (a missing Hadamard rotation on ~57% of the quantized
tensors) and how it was found.

Why MLX: PyTorch+MPS is currently broken/unusably slow for this class of video model on Apple
Silicon. MLX is the only practical local path on Mac hardware today.

### Two separate venvs

`ltx-2-mlx-upstream/` is its own `uv` workspace with its own venv — a different dependency set
than this project's own `.venv` (pure MLX, no torch; the conversion script is the only thing that
needs torch, to read the ComfyUI-format source file). It's gitignored; if missing:

```sh
cd ltx-2-mlx-upstream && uv sync
```

### Setup

```sh
scripts/setup.sh                                    # this project's own .venv (Krea + conversion tooling)
cd ltx-2-mlx-upstream && uv sync && cd ..            # the LTX runtime's own venv
img2vid-ltx-convert --source ~/Downloads/your-checkpoint.safetensors \
  --output-dir ~/.cache/img2vid/ltx23-model          # one-time conversion of your own checkpoint
```

## Video: Usage

Activate the environment first (needed once per shell session):
```sh
source scripts/env.sh && source .venv/bin/activate
```

CLI — text-to-video:
```sh
img2vid --prompt "a heavy wooden door creaks slowly open" --output outputs/clip.mp4
```

CLI — image-to-video (add `--image`):
```sh
img2vid --image photo.jpg --prompt "the person waves at the camera" --output outputs/clip.mp4
```

Web UI:
```sh
scripts/run_ui.sh
# open the printed local URL (http://127.0.0.1:7860); leave the image blank for text-to-video,
# or upload one for image-to-video, enter a prompt, click Generate
```

### Verified performance (measured on this machine: M4 Pro, 64GB unified memory)

Only the dev transformer + CFG one-stage pipeline is available (this checkpoint has no distilled
LoRA fused in): `--one-stage`, always used by `generate_video()`.
- T2V, 256x256, 9 frames, 8 steps: 74s.
- I2V, 320x320, 25 frames, 16 steps: 232s (~3m52s) — real, coherent motion (verified: a
  head-turn matching the prompt, identity preserved throughout the clip).
- `--frames` must satisfy `(frames - 1) % 8 == 0` (LTX's latent frame grid).

## Image: Model

[Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw) (Krea.ai, 12B-parameter dense single-stream
text-to-image DiT), run through [`mlx-gen-krea`](https://github.com/SceneWorks/mlx-gen) (Apache 2.0),
a Rust-native MLX inference library — this project builds a small Rust CLI (`krea-gen/`) against it,
since it's a library, not a standalone tool.

**Unlike the video pipeline, this doesn't use a pre-packaged model download.** The transformer
weights are converted locally from a user-supplied fp8 checkpoint (ComfyUI-style, from an
unverified third-party source) via `scripts/convert_krea_model.sh` — see `src/img2vid/krea_convert.py`
for the exact key-remapping/dequantization logic, derived directly from `mlx-gen-krea`'s Rust
source, not guessed. The text encoder + VAE come from the official (gated) HF repo separately via
`scripts/download_krea_components.sh`.

**License**: [Krea 2 Community License](https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf)
— free for personal/non-commercial use. It contractually requires anyone *deploying* the model to
implement content-filtering to prevent illegal/NCII/CSAM generation. This project is a plain
pass-through wrapper with no such filtering built in — appropriate for personal local use, not for
redistribution or hosting to others without adding that layer yourself.

### Setup

Requires Xcode (not just Command Line Tools — `mlx-gen-krea`'s Metal kernels compile from source)
and Rust (`rustup`). Both one-time system setup, not scripted here.

```sh
scripts/build_krea_gen.sh              # cargo build --release (first build compiles MLX's C++ core)
scripts/convert_krea_model.sh          # converts ~/Downloads/imagemodelfp8.safetensors by default
scripts/download_krea_components.sh    # text encoder + VAE from the gated krea/Krea-2-Raw repo
                                        # (needs `hf auth login` + accepting the license on the model page first)
scripts/download_distill_lora.sh       # optional: the published 2x-speedup LoRA, for --distilled
```

### Usage

```sh
img2vid-image --prompt "a red fox sitting in a snowy forest, photorealistic, soft morning light" \
  --output outputs/fox.png --steps 52 --guidance 3.5
```

Image-to-image editing (optional identity-preserving LoRA):
```sh
img2vid-image --prompt "change the background to a snowy mountain" \
  --edit-source photo.jpg --output outputs/edited.png
```

Web UI: `scripts/run_image_ui.sh` — upload a reference image to switch to edit mode, or leave it
blank for text-to-image.

### Verified performance (measured on this machine: M4 Pro, 64GB unified memory)

Krea 2 Raw is a **true classifier-free-guidance model, not distilled for few-step inference** —
at only 8 steps, output was structurally correct (right composition/pose) but had visible color
corruption and banding artifacts. At the model's documented **52 steps, guidance 3.5**: a real,
clean 1024x1024 photorealistic image in **~41 minutes**, GPU at 99% utilization throughout. Use a
low step count (e.g. 8-20) only for fast pipeline smoke-tests, not for real output.

### Distillation: a 2x-faster LoRA, trained on this hardware

Krea only ships an official few-step model (Turbo) as a *separately, expensively trained*
checkpoint — not something derivable from Raw via a LoRA. Rather than wait on that, this project
includes a self-designed single-stage step-distillation trainer
(`krea-gen/src/bin/distill_train.rs`, progressive distillation per Salimans & Ho 2022, adapted
to need no second resident model copy) that trains a real LoRA cutting the 52-step baseline down
to **26 steps at matching quality**, entirely on a single Apple Silicon Mac.

The trained LoRA is published at
**[huggingface.co/Arekkusul/krea-2-raw-distill-lora](https://huggingface.co/Arekkusul/krea-2-raw-distill-lora)**
— use it directly via `img2vid-image --distilled` (auto-selects 26 steps + the LoRA), or train
your own on your own weights following `docs/krea-distillation-research.md` (full method,
research into 5 rejected published alternatives, and honest results/limitations).

## Development

```sh
scripts/setup.sh
source scripts/env.sh
pytest -q
ruff check .
mypy src
```

Tests mock the `ltx-2-mlx`/`krea-gen` subprocesses entirely — no GPU/model weights needed to run
the suite. End-to-end verification (real model, real generation) is a separate manual step; see
`scripts/verify_e2e.sh`.
