import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open
from safetensors.torch import load_file

SOURCE_PREFIX = "model.diffusion_model."
CONNECTOR_PREFIXES = ("video_embeddings_connector.", "audio_embeddings_connector.")

# Quantization-metadata suffixes that carry no target key of their own -- consumed by the
# dequant step, then dropped.
_QUANT_METADATA_SUFFIXES = (
    ".weight_codebook",
    ".weight_s_channel",
    ".weight_s_rel",
    ".comfy_quant",
    ".weight_scale",
    ".correction",
)

# Base-transformer-only renames (ComfyUI/diffusers naming -> this project's MLX naming),
# derived directly from dgrauet/ltx-2-mlx's own LTXV_LORA_COMFY_RENAMING_MAP and confirmed
# against the real LTXModel class. The embeddings_connector keeps diffusers-native names
# (its own docstring: "matching ff.net.0.proj + ff.net.2 keys") so these must NOT apply there.
_BASE_RENAMES = (
    (".to_out.0.", ".to_out."),
    (".ff.net.0.proj.", ".ff.proj_in."),
    (".ff.net.2.", ".ff.proj_out."),
    ("audio_ff.net.0.proj.", "audio_ff.proj_in."),
    ("audio_ff.net.2.", "audio_ff.proj_out."),
    (".linear_1.", ".linear1."),
    (".linear_2.", ".linear2."),
)

DEFAULT_GROUP_SIZE = 16
DEFAULT_CONVROT_GROUPSIZE = 256


class ConversionError(RuntimeError):
    """Raised when the source checkpoint doesn't match the expected LTX-2.3 architecture,
    or a quantized tensor's shapes are inconsistent with the assumed quantization params."""


def remap_key(source_key: str) -> tuple[str, str] | None:
    """Map a source (ComfyUI-style) LTX-2.3 tensor key to (target_key, destination), where
    destination is "transformer" or "connector".

    Returns None for keys outside our source namespace, or quantization-metadata-only
    suffixes that carry no target of their own.
    """
    if not source_key.startswith(SOURCE_PREFIX):
        return None
    key = source_key[len(SOURCE_PREFIX) :]
    if key.endswith(_QUANT_METADATA_SUFFIXES):
        return None
    if key.startswith(CONNECTOR_PREFIXES):
        return key, "connector"
    for find, replacement in _BASE_RENAMES:
        key = key.replace(find, replacement)
    return key, "transformer"


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Plain per-channel int8 dequantization: value = q * scale (no codebook/rotation)."""
    return (q.float() * scale.float()).to(torch.bfloat16)


_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _build_hadamard(size: int) -> torch.Tensor:
    """Normalized regular Hadamard matrix via recursive Kronecker product, matching
    comfy-kitchen's ConvRot construction exactly (power-of-4 size)."""
    if size in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[size]
    if size < 4 or (size & (size - 1)) != 0:
        raise ConversionError(f"ConvRot size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[size] = h
    return h


def _rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """W_rot = W @ H_block^T, applied per contiguous group of `group_size` columns."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ConversionError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype)
    rotated = torch.matmul(grouped, h_t)
    return rotated.reshape(out_f, in_f)


def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor, s_rel: torch.Tensor, codebook: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Decode nibble-packed 4-bit codebook indices to the intermediate INT8 grid."""
    n, k_half = qdata.shape
    k = k_half * 2
    groups = k // group_size
    if tuple(s_rel.shape) != (n, groups):
        raise ConversionError(f"s_rel must have shape {(n, groups)}, got {tuple(s_rel.shape)}")
    packed = qdata.to(torch.int32) & 0xFF
    quantized = torch.empty(n, k, dtype=torch.int32)
    quantized[:, 0::2] = packed & 0xF
    quantized[:, 1::2] = (packed >> 4) & 0xF
    values = codebook.to(dtype=torch.float32)[quantized]
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp(-127, 127).to(torch.int8)


def _dequantize_from_int8(
    int8_weight: torch.Tensor,
    s_channel: torch.Tensor,
    correction: torch.Tensor | None,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    n, k = int8_weight.shape
    groups = k // group_size
    weight = int8_weight.float().view(n, groups, group_size)
    weight = weight * s_channel.float().view(n, 1, 1)
    if correction is not None:
        weight = weight + correction.t().unsqueeze(-1).float()
    return weight.view(n, k).to(output_dtype)


def dequantize_w4a8_int8_weight(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor,
    correction: torch.Tensor | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    convrot_groupsize: int = DEFAULT_CONVROT_GROUPSIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode ComfyUI's asym_w4a8_int8 storage into a dense floating weight: nibble-unpack ->
    codebook lookup -> per-group relative scale -> per-channel scale -> inverse ConvRot
    (Hadamard) rotation. Ported directly from Comfy-Org/comfy-kitchen's eager reference
    implementation (Apache 2.0), not reverse-engineered from a description.
    """
    int8_weight = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
    weight_rotated = _dequantize_from_int8(int8_weight, s_channel, correction, group_size, output_dtype)
    h = _build_hadamard(convrot_groupsize)
    return _rotate_weight(weight_rotated.float(), h, convrot_groupsize).to(output_dtype)


def _dequantize_tensor(raw: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Dequantize `raw[key]` using whichever companion metadata is present, or pass it
    through as bf16 if it was never quantized."""
    tensor = raw[key]
    codebook = raw.get(f"{key}_codebook")
    if codebook is not None:
        return dequantize_w4a8_int8_weight(
            tensor,
            raw[f"{key}_s_rel"],
            raw[f"{key}_s_channel"],
            codebook,
            correction=raw.get(f"{key}_correction"),
        )
    scale = raw.get(f"{key}_scale")
    if scale is not None:
        return dequantize_int8_simple(tensor, scale)
    return tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor


# Tensors that were originally quantized in the source AND are 2D Linear weights get
# MLX-native-quantized in our output. Everything else (biases, norms, gate tables,
# scale-shift tables, and the whole connector) stays dense -- matching the official
# recipe's "kept high precision" set for exactly this kind of small/structural tensor.
def _should_quantize_output(key: str, was_source_quantized: bool, value: torch.Tensor) -> bool:
    return was_source_quantized and value.dim() == 2 and key.endswith(".weight")


def _to_mx(tensor: torch.Tensor) -> mx.array:
    """torch -> mx.array. numpy has no bfloat16, so bf16 tensors go through float32."""
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.float().numpy()).astype(mx.bfloat16)
    return mx.array(tensor.numpy())


def convert_checkpoint(source_path: Path, output_dir: Path) -> Path:
    """Convert a ComfyUI-style w4a8-quantized LTX-2.3 checkpoint into the split
    transformer-dev.safetensors + connector.safetensors layout dgrauet/ltx-2-mlx expects.

    The transformer's originally-quantized Linear weights are re-quantized to MLX-native Q8
    (via mx.quantize) to fit the converted output on disk; the connector and all non-Linear
    transformer tensors (biases, norms, tables) stay dense bf16.

    Raises FileNotFoundError if `source_path` doesn't exist, ConversionError for malformed
    quantization metadata.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    try:
        raw = load_file(str(source_path))
    except Exception as exc:
        raise ConversionError(f"failed to read {source_path} as a safetensors checkpoint: {exc}") from exc

    transformer: dict[str, mx.array] = {}
    connector: dict[str, mx.array] = {}

    for key in raw:
        if key.endswith(_QUANT_METADATA_SUFFIXES):
            continue
        mapped = remap_key(key)
        if mapped is None:
            continue
        target_key, destination = mapped

        was_quantized = f"{key}_codebook" in raw or f"{key}_scale" in raw
        value = _dequantize_tensor(raw, key)

        if destination == "connector":
            connector[target_key] = _to_mx(value)
            continue

        if _should_quantize_output(target_key, was_quantized, value):
            wq, scales, biases = mx.quantize(_to_mx(value), group_size=64, bits=8)
            transformer[target_key] = wq
            transformer[f"{target_key.removesuffix('.weight')}.scales"] = scales
            transformer[f"{target_key.removesuffix('.weight')}.biases"] = biases
        else:
            transformer[target_key] = _to_mx(value)

    if not transformer:
        raise ConversionError(f"no {SOURCE_PREFIX}* tensors found in {source_path} -- wrong checkpoint?")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "transformer-dev.safetensors"), transformer)
    mx.save_safetensors(str(output_dir / "connector.safetensors"), connector)

    with safe_open(str(source_path), framework="numpy") as f:
        meta = f.metadata() or {}
    if "config" in meta:
        (output_dir / "embedded_config.json").write_text(
            json.dumps(json.loads(meta["config"]), indent=2)
        )

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="img2vid-ltx-convert",
        description="Convert a ComfyUI-style w4a8-quantized LTX-2.3 checkpoint into the split "
        "transformer-dev.safetensors + connector.safetensors layout dgrauet/ltx-2-mlx expects.",
    )
    parser.add_argument("--source", required=True, help="Path to the source .safetensors file")
    parser.add_argument("--output-dir", required=True, help="Model directory to write into")
    args = parser.parse_args(argv)

    try:
        result_dir = convert_checkpoint(Path(args.source), Path(args.output_dir))
    except (FileNotFoundError, ConversionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(str(result_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
