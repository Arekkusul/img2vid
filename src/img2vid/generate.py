import subprocess
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".cache" / "img2vid" / "ltx23-model"


def _default_ltx_bin(repo_root: Path | None = None) -> str:
    """Prefer the ltx-2-mlx binary built into ltx-2-mlx-upstream's own uv venv.

    ltx-2-mlx-upstream is a separate `uv` workspace (its own venv, own dependency set --
    it needs a much newer/different MLX stack than this project's Krea/Wan tooling), not a
    console-script installed into this project's own venv like mlxgen was. Its console
    script's shebang points directly at that venv's python, so invoking it needs no extra
    env setup (no DYLD_LIBRARY_PATH, no `uv run`) -- just the absolute path.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    candidate = root / "ltx-2-mlx-upstream" / ".venv" / "bin" / "ltx-2-mlx"
    return str(candidate) if candidate.is_file() else "ltx-2-mlx"


DEFAULT_LTX_BIN = _default_ltx_bin()


class GenerationError(RuntimeError):
    """Raised when ltx-2-mlx fails to produce a usable video file."""


def generate_video(
    prompt: str,
    *,
    image_path: Path | None = None,
    output_path: Path | None = None,
    width: int = 704,
    height: int = 480,
    frames: int = 97,
    steps: int = 30,
    cfg_scale: float = 3.0,
    frame_rate: float = 24.0,
    seed: int | None = None,
    model: Path = DEFAULT_MODEL,
    dev: bool = False,
    low_ram: bool = True,
    ltx_bin: str = DEFAULT_LTX_BIN,
    timeout: float = 1800,
) -> Path:
    """Generate a video from a text prompt (T2V), or a prompt + image (I2V), via LTX-2.5.

    Text-to-video when `image_path` is None; image-to-video (single-anchor, frame 0) when
    it's set -- mirrors `image_generate.py`'s optional-edit-image pattern.

    Defaults to the fast, CFG-free distilled pipeline (`--distilled`, needs
    `transformer-distilled.safetensors` -- see `scripts/fuse_distilled_lora.py`, which fuses
    the community distilled LoRA onto this project's own converted dev weights, not someone
    else's checkpoint). Measured ~6x faster than `dev=True` at equal quality (36s vs 232s at
    320x320x25 frames). `dev=True` uses the slower dev transformer + CFG one-stage pipeline
    (`--one-stage`, needs only `transformer-dev.safetensors`) -- the only path available
    before the LoRA was fused, kept as a fallback.

    Raises FileNotFoundError/ValueError for bad inputs (fail fast, before spawning a
    subprocess), or GenerationError if ltx-2-mlx fails, hangs past `timeout` seconds, or
    produces no usable output file. Default timeout is generous (30min) since real
    generations legitimately take minutes.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if (frames - 1) % 8 != 0:
        raise ValueError(f"frames must satisfy (frames - 1) % 8 == 0, got {frames}")

    model = Path(model)
    if not model.is_dir():
        raise FileNotFoundError(
            f"LTX model directory not found: {model} (run img2vid-ltx-convert first)"
        )

    required_transformer = "transformer-dev.safetensors" if dev else "transformer-distilled.safetensors"
    if not (model / required_transformer).is_file():
        if dev:
            raise FileNotFoundError(
                f"{model / required_transformer} not found (run img2vid-ltx-convert first)"
            )
        raise FileNotFoundError(
            f"{model / required_transformer} not found -- the fast default path needs the "
            "distilled LoRA fused onto your dev weights first: run "
            "scripts/fuse_distilled_lora.py, or pass dev=True to use the slower --one-stage "
            "path (needs only transformer-dev.safetensors)"
        )

    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path = Path(output_path) if output_path is not None else Path("outputs/generated.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        ltx_bin, "generate",
        "--model", str(model),
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "-f", str(frames),
        "--frame-rate", str(frame_rate),
        "--output", str(output_path),
    ]
    if dev:
        argv += ["--one-stage", "--steps", str(steps), "--cfg-scale", str(cfg_scale)]
    else:
        argv.append("--distilled")
    if image_path is not None:
        argv += ["--image", str(image_path)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if low_ram:
        argv.append("--low-ram")

    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise GenerationError(
            f"'{ltx_bin}' is not installed; build ltx-2-mlx-upstream's venv "
            "(cd ltx-2-mlx-upstream && uv sync)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(f"ltx-2-mlx did not finish within {timeout}s and was killed") from exc

    if result.returncode != 0:
        raise GenerationError(f"ltx-2-mlx exited with code {result.returncode}: {result.stderr.strip()}")

    if not output_path.exists():
        raise GenerationError(
            f"ltx-2-mlx reported success but the output file was never created: {output_path}"
        )
    if output_path.stat().st_size == 0:
        raise GenerationError(f"ltx-2-mlx produced an empty output file: {output_path}")

    return output_path
