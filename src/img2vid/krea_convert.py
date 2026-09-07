import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SOURCE_PREFIX = "model.diffusion_model."


class ConversionError(RuntimeError):
    """Raised when a source tensor key can't be mapped, or the converted checkpoint's key set
    doesn't match the architecture mlx-gen-krea expects."""


@dataclass(frozen=True)
class Krea2Config:
    """Mirrors mlx-gen-krea's Rust `Krea2Config::turbo()` (verified against its source) — the
    dense single-stream Krea 2 DiT shared by Turbo and Raw."""

    hidden_size: int = 6144
    num_attention_heads: int = 48
    num_kv_heads: int = 12
    attention_head_dim: int = 128
    num_layers: int = 28
    intermediate_size: int = 16384
    num_layerwise_text_blocks: int = 2
    num_refiner_text_blocks: int = 2
    text_hidden_dim: int = 2560
    text_intermediate_size: int = 6912
    in_channels: int = 64

    @classmethod
    def turbo(cls) -> "Krea2Config":
        return cls()


_BLOCK_SUFFIX_MAP = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
}

_TOP_LEVEL_MAP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_TEXT_FUSION_BLOCK_RE = re.compile(r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$")


def remap_key(source_key: str) -> str | None:
    """Map a ComfyUI-style source tensor key to its diffusers-style mlx-gen-krea target key.

    Returns None for keys outside our source namespace or quantization-only metadata
    (`.weight_scale`) that carries no target of its own. Raises ConversionError for a key that
    is in our namespace but doesn't match any known naming pattern — a sign the source checkpoint
    isn't the architecture this mapping was derived from.
    """
    if not source_key.startswith(SOURCE_PREFIX):
        return None
    key = source_key[len(SOURCE_PREFIX) :]
    if key.endswith(".weight_scale"):
        return None

    if key in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[key]

    m = _BLOCK_RE.match(key)
    if m:
        idx, suffix = m.groups()
        if suffix == "mod.lin":
            return f"transformer_blocks.{idx}.scale_shift_table"
        mapped = _BLOCK_SUFFIX_MAP.get(suffix)
        if mapped is None:
            raise ConversionError(f"unrecognized block tensor suffix {suffix!r} (from {source_key})")
        return f"transformer_blocks.{idx}.{mapped}"

    m = _TEXT_FUSION_BLOCK_RE.match(key)
    if m:
        family, idx, suffix = m.groups()
        mapped = _BLOCK_SUFFIX_MAP.get(suffix)
        if mapped is None:
            raise ConversionError(
                f"unrecognized text-fusion tensor suffix {suffix!r} (from {source_key})"
            )
        return f"text_fusion.{family}.{idx}.{mapped}"

    raise ConversionError(f"unrecognized source tensor key: {source_key!r}")


def _attn_keys(prefix: str) -> list[str]:
    return [f"{prefix}.{name}.weight" for name in ("norm_q", "norm_k", "to_q", "to_k", "to_v", "to_gate", "to_out.0")]


def _ff_keys(prefix: str) -> list[str]:
    return [f"{prefix}.{name}.weight" for name in ("gate", "up", "down")]


def _text_block_keys(prefix: str) -> list[str]:
    norms = [f"{prefix}.norm1.weight", f"{prefix}.norm2.weight"]
    return _attn_keys(f"{prefix}.attn") + _ff_keys(f"{prefix}.ff") + norms


def _single_block_keys(prefix: str) -> list[str]:
    return _text_block_keys(prefix) + [f"{prefix}.scale_shift_table"]


def expected_transformer_keys(cfg: Krea2Config) -> set[str]:
    """Python reimplementation of mlx-gen-krea's Rust `expected_transformer_keys()` — the exact
    key set its `validate_transformer()` requires, used here as our own pre-flight check."""
    keys: list[str] = [
        "img_in.weight",
        "img_in.bias",
        "txt_in.norm.weight",
        "txt_in.linear_1.weight",
        "txt_in.linear_1.bias",
        "txt_in.linear_2.weight",
        "txt_in.linear_2.bias",
        "time_embed.linear_1.weight",
        "time_embed.linear_1.bias",
        "time_embed.linear_2.weight",
        "time_embed.linear_2.bias",
        "time_mod_proj.weight",
        "time_mod_proj.bias",
    ]
    for i in range(cfg.num_layerwise_text_blocks):
        keys += _text_block_keys(f"text_fusion.layerwise_blocks.{i}")
    keys.append("text_fusion.projector.weight")
    for i in range(cfg.num_refiner_text_blocks):
        keys += _text_block_keys(f"text_fusion.refiner_blocks.{i}")
    for i in range(cfg.num_layers):
        keys += _single_block_keys(f"transformer_blocks.{i}")
    keys += [
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.norm.weight",
        "final_layer.scale_shift_table",
    ]
    return set(keys)


_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def dequantize(tensor: torch.Tensor, scale: torch.Tensor | None, *, key: str = "") -> torch.Tensor:
    if tensor.dtype in _FP8_DTYPES and scale is None:
        raise ConversionError(f"{key} is fp8-quantized but has no companion weight_scale tensor")
    if scale is None:
        return tensor.to(torch.bfloat16)
    return (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)


def convert_transformer(source_path: Path, output_dir: Path, cfg: Krea2Config | None = None) -> Path:
    """Convert a ComfyUI-style Krea 2 transformer checkpoint into the diffusers-style
    `transformer/` snapshot dir mlx-gen-krea's `Weights::from_dir` expects.

    Raises FileNotFoundError if `source_path` doesn't exist, ConversionError if the file isn't a
    valid safetensors checkpoint, any source key doesn't match the expected naming, or the
    converted key set doesn't exactly match the architecture's expected keys (missing or extra).
    """
    cfg = cfg or Krea2Config.turbo()
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    try:
        raw = load_file(str(source_path))
    except Exception as exc:
        raise ConversionError(f"failed to read {source_path} as a safetensors checkpoint: {exc}") from exc

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale"):
            continue
        target = remap_key(key)
        if target is None:
            continue
        scale = raw.get(f"{key}_scale") if key.endswith(".weight") else None
        converted[target] = dequantize(tensor, scale, key=key)

    expected = expected_transformer_keys(cfg)
    actual = set(converted.keys())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ConversionError(
            f"converted key set does not match expected architecture: "
            f"{len(missing)} missing (e.g. {sorted(missing)[:5]}), "
            f"{len(extra)} extra (e.g. {sorted(extra)[:5]})"
        )

    output_dir = Path(output_dir)
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(transformer_dir / "diffusion_pytorch_model.safetensors"))
    # Every field here equals Krea2Config::turbo()'s Rust defaults, which is what the loader
    # falls back to for any key missing from this file — so `{}` would work too, but writing the
    # values explicitly makes the produced snapshot self-describing and inspectable.
    (transformer_dir / "config.json").write_text(json.dumps(_config_json(cfg), indent=2))
    return transformer_dir


def _config_json(cfg: Krea2Config) -> dict:
    return {
        "in_channels": cfg.in_channels,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "num_layers": cfg.num_layers,
        "intermediate_size": cfg.intermediate_size,
        "num_layerwise_text_blocks": cfg.num_layerwise_text_blocks,
        "num_refiner_text_blocks": cfg.num_refiner_text_blocks,
        "text_hidden_dim": cfg.text_hidden_dim,
        "text_intermediate_size": cfg.text_intermediate_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="img2vid-krea-convert",
        description="Convert a ComfyUI-style Krea 2 transformer checkpoint into the diffusers-style "
        "snapshot layout mlx-gen-krea expects.",
    )
    parser.add_argument("--source", required=True, help="Path to the source .safetensors file")
    parser.add_argument("--output-dir", required=True, help="Snapshot directory to write transformer/ into")
    args = parser.parse_args(argv)

    try:
        transformer_dir = convert_transformer(Path(args.source), Path(args.output_dir))
    except (FileNotFoundError, ConversionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(str(transformer_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
