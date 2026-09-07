# img2vid

Two independent local generation tools, both on-device via [MLX](https://github.com/ml-explore/mlx)
— no cloud API, no CUDA required:
- **Image-to-video**: give it an image + a text prompt, get back a generated video clip.
- **Text-to-image / image-edit** (standalone, see below): give it a prompt, get back an image;
  optionally give it a reference image too for editing.

## Video: Model

[Wan2.2 TI2V-5B](https://huggingface.co/AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit)
(Alibaba, Apache 2.0), a dense 5B-parameter unified text+image-to-video diffusion model, run
through [`mlx-gen`](https://github.com/lpalbou/mlx-gen) (MIT license), an MLX-native inference
runtime. This is the standard open-weight release, unmodified.

Why this model/runtime combo: PyTorch+MPS is currently broken for Wan2.2-class video models on
Apple Silicon (measured: 82 minutes for a 2-second clip via GGUF+ComfyUI+MPS). MLX is the only
practical local path on Mac hardware today.

### Disk & memory

- Model weights (q8 package): ~16.9GB, cached at `~/.cache/huggingface/hub` (shared across
  projects, not project-local).
- Quantization saves disk only — `mlx-gen` dequantizes to BF16 at load time, so RAM usage is not
  reduced by using q8 over bf16.
- Default generation settings (`832x480`, 81 frames, `--low-ram`) are chosen to stay well within
  64GB unified memory. Larger canvases (up to the model's native `1280x704`) are possible with
  more headroom but are not the default — see `docs/upgrading.md`-style notes below.

### Upgrading to a larger model

Wan2.2 I2V-A14B (MoE, ~39.5GB via the same `mlx-gen` runtime) gives higher quality but consumes
nearly all free disk on a machine with ~43GB free. Not installed by default. To use it, download
via `mlxgen download --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` and pass
`--model` to override the default in `img2vid generate` / the UI.

## Video: Setup

```sh
scripts/setup.sh          # creates .venv, installs deps (handles a local Homebrew pyexpat bug)
scripts/download_model.sh # downloads the ~17GB model weights (checks free disk first)
```

## Video: Usage

Activate the environment first (needed once per shell session):
```sh
source scripts/env.sh && source .venv/bin/activate
```

CLI:
```sh
img2vid --image photo.jpg --prompt "the person waves at the camera" --output outputs/clip.mp4
```

Web UI:
```sh
scripts/run_ui.sh
# open the printed local URL (http://127.0.0.1:7860), upload an image, enter a prompt, click Generate
```

### Verified performance (measured on this machine: M4 Pro, 64GB unified memory)

Default settings (`832x480`, 81 frames, 25 steps, `--low-ram`): **~17.5 minutes** wall clock,
GPU (Metal via MLX) at 98-100% utilization throughout. Output: 4.05s clip at 20fps. A minimal
smoke-test run (9 frames, 4 steps) took ~79s — useful for quickly checking the pipeline works
before committing to a full-length generation.

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

## Development

```sh
scripts/setup.sh
source scripts/env.sh
pytest -q
ruff check .
mypy src
```

Tests mock the `mlxgen` subprocess entirely — no GPU/model weights needed to run the suite.
End-to-end verification (real model, real generation) is a separate manual step; see
`scripts/verify_e2e.sh`.
