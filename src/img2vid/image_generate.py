import subprocess
from pathlib import Path

DEFAULT_SNAPSHOT_DIR = Path.home() / ".cache" / "img2vid" / "krea2-snapshot"
DEFAULT_DISTILL_LORA = Path.home() / ".cache" / "img2vid" / "krea2-distill-lora.safetensors"
DISTILLED_STEPS = 26


class ImageGenerationError(RuntimeError):
    """Raised when krea-gen fails to produce a usable image file."""


def _default_krea_bin(repo_root: Path | None = None) -> str:
    """Prefer the krea-gen binary built into this repo's krea-gen/ crate.

    krea-gen isn't a pip console-script (it's a separately-built Rust binary), so unlike
    mlxgen there's no venv sibling to resolve against — fall back to bare "krea-gen" (a $PATH
    lookup) if the crate hasn't been built yet.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    candidate = root / "krea-gen" / "target" / "release" / "krea-gen"
    return str(candidate) if candidate.is_file() else "krea-gen"


DEFAULT_KREA_BIN = _default_krea_bin()


def generate_image(
    prompt: str,
    *,
    output_path: Path | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 52,
    guidance: float = 3.5,
    seed: int | None = None,
    negative_prompt: str | None = None,
    edit_image_path: Path | None = None,
    lora_path: Path | None = None,
    turbo_edit: bool = False,
    distilled: bool = False,
    quantize: str = "q8",
    snapshot: Path = DEFAULT_SNAPSHOT_DIR,
    krea_bin: str = DEFAULT_KREA_BIN,
    timeout: float = 3600,
) -> Path:
    """Generate (or edit) an image with Krea 2 via the krea-gen CLI.

    Text-to-image when `edit_image_path` is None; image-edit mode (optionally with an identity
    LoRA) when it's set. `lora_path` applies in EITHER mode (Raw-trained LoRAs, e.g. the
    distillation adapter, are exactly as valid for plain text-to-image as the identity-edit LoRA
    is for edit mode -- `krea-gen`'s `--lora` flag isn't edit-specific). `turbo_edit` remains
    edit-mode only (it selects the CFG-free Turbo edit schedule, which has no t2i counterpart).

    `distilled=True` is a t2i-only convenience for the self-trained step-distillation LoRA
    (`docs/krea-distillation-research.md`): defaults `lora_path` to `DEFAULT_DISTILL_LORA` (if
    not otherwise given) and `steps` to `DISTILLED_STEPS` (26) *when the caller left `steps` at
    its plain default of 52* -- an explicit `steps=N` from the caller is always respected.
    Raises FileNotFoundError with an actionable message if the distilled LoRA hasn't been
    trained yet.

    Raises FileNotFoundError/ValueError for bad inputs (fail fast, before spawning a
    subprocess), or ImageGenerationError if krea-gen fails, hangs past `timeout` seconds, or
    produces no usable output file.

    `quantize` defaults to "q8" ("none" or "q4" also accepted, matching krea-gen's own
    `--quantize` values -- "none" omits the flag, using krea-gen's own dense default). Measured
    directly: dense (unquantized) execution produced a real face-region corruption artifact
    that `--quantize q8` eliminated, even though the underlying converted weight VALUES already
    matched the official reference checkpoint at ~0.999 correlation -- this is a runtime
    numerical-stability difference (bf16 matmul precision vs. a well-calibrated int8 path), not
    a weight-conversion bug. krea-gen's own `--quantize` help text independently documents Q8 as
    near-lossless and typically faster on Apple Silicon for this model family.
    """
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if edit_image_path is None and turbo_edit:
        raise ValueError("turbo_edit requires edit_image_path (edit mode only)")
    if distilled and edit_image_path is not None:
        raise ValueError("distilled is a text-to-image convenience; it doesn't apply in edit mode")

    if distilled:
        if lora_path is None:
            lora_path = DEFAULT_DISTILL_LORA
        if steps == 52:
            steps = DISTILLED_STEPS
        if not Path(lora_path).is_file():
            raise FileNotFoundError(
                f"Distilled LoRA not found: {lora_path} "
                "(run scripts/download_distill_lora.sh to fetch the published adapter, "
                "or train your own -- see docs/krea-distillation-research.md)"
            )

    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Krea 2 snapshot not found: {snapshot} "
            "(run scripts/convert_krea_model.sh and scripts/download_krea_components.sh)"
        )

    if edit_image_path is not None:
        edit_image_path = Path(edit_image_path)
        if not edit_image_path.is_file():
            raise FileNotFoundError(f"Edit source image not found: {edit_image_path}")

    output_path = Path(output_path) if output_path is not None else Path("outputs/krea_generated.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        krea_bin,
        "--snapshot", str(snapshot),
        "--prompt", prompt,
        "--output", str(output_path),
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--guidance", str(guidance),
    ]
    if quantize != "none":
        argv += ["--quantize", quantize]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if negative_prompt:
        argv += ["--negative-prompt", negative_prompt]
    if lora_path is not None:
        argv += ["--lora", str(lora_path)]
    if edit_image_path is not None:
        argv += ["--edit-source", str(edit_image_path)]
        if turbo_edit:
            argv.append("--turbo-edit")

    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ImageGenerationError(
            f"'{krea_bin}' is not installed; run scripts/build_krea_gen.sh"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ImageGenerationError(f"krea-gen did not finish within {timeout}s and was killed") from exc

    if result.returncode != 0:
        raise ImageGenerationError(f"krea-gen exited with code {result.returncode}: {result.stderr.strip()}")

    if not output_path.exists():
        raise ImageGenerationError(
            f"krea-gen reported success but the output file was never created: {output_path}"
        )
    if output_path.stat().st_size == 0:
        raise ImageGenerationError(f"krea-gen produced an empty output file: {output_path}")

    return output_path
