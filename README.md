# img2vid

Local image-to-video generation: give it an image + a text prompt, get back a generated
video clip. Runs entirely on-device via [MLX](https://github.com/ml-explore/mlx) — no cloud
API, no CUDA required.

## Model

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

## Setup

```sh
scripts/setup.sh          # creates .venv, installs deps (handles a local Homebrew pyexpat bug)
scripts/download_model.sh # downloads the ~17GB model weights (checks free disk first)
```

## Usage

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
