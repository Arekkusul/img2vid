import os
import subprocess
import sys
from pathlib import Path

DYLD_LIBRARY_PATH = "/opt/homebrew/opt/expat/lib"
DEFAULT_MODEL = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"


def _default_mlxgen_bin(executable: str | None = None) -> str:
    """Prefer the mlxgen console script installed alongside this interpreter.

    A bare "mlxgen" only resolves via $PATH, which isn't guaranteed to include the
    venv's bin dir unless the venv was `source`-activated — e.g. invoking
    `.venv/bin/img2vid` directly (as scripts/run_ui.sh and a packaged install both do)
    does not activate the venv. sys.executable is always the venv's python in that case,
    so its sibling `mlxgen` is the reliable target.
    """
    candidate = Path(executable or sys.executable).parent / "mlxgen"
    return str(candidate) if candidate.is_file() else "mlxgen"


DEFAULT_MLXGEN_BIN = _default_mlxgen_bin()


class GenerationError(RuntimeError):
    """Raised when mlxgen fails to produce a usable video file."""


def generate_video(
    image_path: Path,
    prompt: str,
    *,
    output_path: Path | None = None,
    width: int = 832,
    height: int = 480,
    frames: int = 81,
    steps: int = 25,
    guidance: float = 5.0,
    flow_shift: float = 3.0,
    fps: int = 20,
    seed: int | None = None,
    model: str = DEFAULT_MODEL,
    low_ram: bool = True,
    mlxgen_bin: str = DEFAULT_MLXGEN_BIN,
    timeout: float = 1800,
) -> Path:
    """Generate a video from an image + prompt via the mlxgen CLI.

    Raises FileNotFoundError/ValueError for bad inputs (fail fast, before spawning a
    subprocess), or GenerationError if mlxgen fails, hangs past `timeout` seconds, or
    produces no usable output file. Default timeout is generous (30min) since real
    generations legitimately take minutes.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    output_path = Path(output_path) if output_path is not None else Path("outputs/generated.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        mlxgen_bin, "generate",
        "--model", model,
        "--image", str(image_path),
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--flow-shift", str(flow_shift),
        "--fps", str(fps),
        "--output", str(output_path),
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if low_ram:
        argv.append("--low-ram")

    env = {**os.environ, "DYLD_LIBRARY_PATH": DYLD_LIBRARY_PATH}

    try:
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=timeout)
    except FileNotFoundError as exc:
        raise GenerationError(
            f"'{mlxgen_bin}' is not installed or not on PATH; run scripts/setup.sh"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GenerationError(f"mlxgen did not finish within {timeout}s and was killed") from exc

    if result.returncode != 0:
        raise GenerationError(f"mlxgen exited with code {result.returncode}: {result.stderr.strip()}")

    if not output_path.exists():
        raise GenerationError(
            f"mlxgen reported success but the output file was never created: {output_path}"
        )
    if output_path.stat().st_size == 0:
        raise GenerationError(f"mlxgen produced an empty output file: {output_path}")

    return output_path
