"""Fuse the LTX-2.5 community distilled LoRA onto this project's own converted dev
transformer, producing transformer-distilled.safetensors -- the file generate_video()'s
fast default path (see src/img2vid/generate.py) needs.

Built entirely from the user's own converted weights (transformer-dev.safetensors, from
img2vid-ltx-convert) + the distilled LoRA's delta -- not a copy of anyone else's pre-fused
checkpoint.

Must run under ltx-2-mlx-upstream's own venv (needs ltx_core_mlx, a separate uv workspace
not installed into this project's own .venv) -- use fuse_distilled_lora.sh, or:
    cd ltx-2-mlx-upstream && uv run python ../scripts/fuse_distilled_lora.py --lora <path>

Reuses ltx_core_mlx.loader.fuse_loras.apply_loras (the same function the real
--two-stage/--distilled-lora runtime path uses for on-the-fly fusion) and
LTXV_LORA_COMFY_RENAMING_MAP (the same key remapping the runtime applies to community
LoRAs) rather than reimplementing the dequant/add/requant math or key-renaming rules.
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import mlx.core as mx

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "img2vid" / "ltx23-model"


def fuse(model_dir: Path, lora_path: Path, strength: float, output_name: str) -> Path:
    upstream_src = Path(__file__).resolve().parent.parent / "ltx-2-mlx-upstream" / "packages" / "ltx-core-mlx" / "src"
    sys.path.insert(0, str(upstream_src))
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

    dev_path = model_dir / "transformer-dev.safetensors"
    if not dev_path.is_file():
        raise FileNotFoundError(f"{dev_path} not found -- run img2vid-ltx-convert first")
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA not found: {lora_path}")

    print(f"loading {dev_path} ...", flush=True)
    raw_model = cast("dict[str, mx.array]", mx.load(str(dev_path)))
    flat_model = {k.removeprefix("transformer."): v for k, v in raw_model.items()}
    print(f"  {len(flat_model)} tensors", flush=True)

    print(f"loading LoRA {lora_path} ...", flush=True)
    raw_lora = cast("dict[str, mx.array]", mx.load(str(lora_path)))
    lora_remapped = {}
    for k, v in raw_lora.items():
        new_key = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(k)
        if new_key is not None:
            lora_remapped[new_key] = v
    print(f"  {len(raw_lora)} raw keys -> {len(lora_remapped)} remapped", flush=True)

    model_sd = StateDict(sd=flat_model, size=0, dtype=set())
    lora_sd = StateDict(sd=lora_remapped, size=0, dtype=set())
    lora_with_strength = LoraStateDictWithStrength(lora_sd, strength)

    print("fusing (dequant -> add delta -> requant per quantized tensor) ...", flush=True)
    fused = apply_loras(model_sd, [lora_with_strength])
    print(f"  fused {len(fused.sd)} tensors", flush=True)

    print("evaluating in chunks (avoid one giant lazy graph) ...", flush=True)
    items = list(fused.sd.items())
    chunk = 64
    for i in range(0, len(items), chunk):
        batch = items[i : i + chunk]
        mx.eval(*(v for _, v in batch))

    prefixed = {f"transformer.{k}": v for k, v in fused.sd.items()}

    output_path = model_dir / output_name
    print(f"saving {output_path} ...", flush=True)
    mx.save_safetensors(str(output_path), prefixed)
    print("done", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fuse a distilled LoRA onto this project's own converted LTX dev transformer."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path)
    parser.add_argument("--lora", required=True, type=Path, help="Path to the distilled LoRA safetensors file")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output-name", default="transformer-distilled.safetensors")
    args = parser.parse_args(argv)

    try:
        output_path = fuse(args.model_dir, args.lora, args.strength, args.output_name)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
