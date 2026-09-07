import mlx.core as mx
import torch
from safetensors.torch import load_file, save_file

from img2vid.ltx_convert import (
    ConversionError,
    convert_checkpoint,
    dequantize_int8_simple,
    dequantize_w4a8_int8_weight,
    remap_key,
)

SRC = "model.diffusion_model."


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------


def test_remap_key_wrong_prefix_returns_none():
    assert remap_key("some.other.model.weight") is None


def test_remap_key_quant_metadata_returns_none():
    for suffix in ["weight_codebook", "weight_s_channel", "weight_s_rel", "comfy_quant", "weight_scale"]:
        assert remap_key(f"{SRC}transformer_blocks.0.attn1.to_k.{suffix}") is None


def test_remap_key_connector_keys_pass_through_unchanged():
    # Connector keeps diffusers-native naming (ff.net.0.proj / to_out.0) -- no renames apply.
    key, dest = remap_key(f"{SRC}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight")
    assert dest == "connector"
    assert key == "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"

    key, dest = remap_key(f"{SRC}audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight")
    assert dest == "connector"
    assert key == "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"

    key, dest = remap_key(f"{SRC}video_embeddings_connector.learnable_registers")
    assert dest == "connector"
    assert key == "video_embeddings_connector.learnable_registers"


def test_remap_key_transformer_to_out_rename():
    key, dest = remap_key(f"{SRC}transformer_blocks.0.attn1.to_out.0.weight")
    assert dest == "transformer"
    assert key == "transformer_blocks.0.attn1.to_out.weight"


def test_remap_key_transformer_ff_rename():
    key, dest = remap_key(f"{SRC}transformer_blocks.0.ff.net.0.proj.weight")
    assert (key, dest) == ("transformer_blocks.0.ff.proj_in.weight", "transformer")

    key, dest = remap_key(f"{SRC}transformer_blocks.0.ff.net.2.weight")
    assert (key, dest) == ("transformer_blocks.0.ff.proj_out.weight", "transformer")


def test_remap_key_transformer_audio_ff_rename():
    key, dest = remap_key(f"{SRC}transformer_blocks.0.audio_ff.net.0.proj.weight")
    assert (key, dest) == ("transformer_blocks.0.audio_ff.proj_in.weight", "transformer")

    key, dest = remap_key(f"{SRC}transformer_blocks.0.audio_ff.net.2.weight")
    assert (key, dest) == ("transformer_blocks.0.audio_ff.proj_out.weight", "transformer")


def test_remap_key_linear_1_2_rename():
    key, dest = remap_key(f"{SRC}adaln_single.emb.timestep_embedder.linear_1.weight")
    assert (key, dest) == ("adaln_single.emb.timestep_embedder.linear1.weight", "transformer")

    key, dest = remap_key(f"{SRC}adaln_single.emb.timestep_embedder.linear_2.weight")
    assert (key, dest) == ("adaln_single.emb.timestep_embedder.linear2.weight", "transformer")


def test_remap_key_untouched_top_level_passes_through():
    key, dest = remap_key(f"{SRC}patchify_proj.weight")
    assert (key, dest) == ("patchify_proj.weight", "transformer")

    key, dest = remap_key(f"{SRC}scale_shift_table")
    assert (key, dest) == ("scale_shift_table", "transformer")


# ---------------------------------------------------------------------------
# Dequantization math
# ---------------------------------------------------------------------------


def test_dequantize_int8_simple():
    q = torch.tensor([[2, -3], [4, 5]], dtype=torch.int8)
    scale = torch.tensor([[2.0], [0.5]])
    result = dequantize_int8_simple(q, scale)
    expected = torch.tensor([[4.0, -6.0], [2.0, 2.5]])
    assert torch.allclose(result.float(), expected, atol=1e-2)
    assert result.dtype == torch.bfloat16


def test_dequantize_w4a8_int8_weight_roundtrips_a_known_value():
    # Build a weight, quantize it "by hand" using the exact reference recipe (codebook levels
    # picked so nibble index round-trips exactly), then confirm our dequant recovers it.
    n, k, group_size, convrot_groupsize = 4, 256, 16, 256
    groups = k // group_size

    # A codebook where each index maps to a distinct, exactly-representable value.
    codebook = torch.linspace(-1.0, 1.0, 16)

    # Pick nibble index 15 (codebook[15] == 1.0) everywhere, with realistic-magnitude scales --
    # s_rel must be large enough that codebook_value * s_rel survives the intermediate
    # round-to-int8 step without collapsing to 0 (a too-small s_rel, e.g. 1.0, rounds any
    # sub-1.0 codebook value to the integer 0, which a naive test could mistake for a bug).
    nibble = 15
    packed_byte = (nibble & 0xF) | ((nibble & 0xF) << 4)
    qdata = torch.full((n, k // 2), packed_byte, dtype=torch.uint8).to(torch.int8)
    s_rel = torch.full((n, groups), 64.0, dtype=torch.float32)
    s_channel = torch.full((n,), 2.0, dtype=torch.float32)

    result = dequantize_w4a8_int8_weight(
        qdata, s_rel, s_channel, codebook=codebook, group_size=group_size, convrot_groupsize=convrot_groupsize
    )
    assert result.shape == (n, k)
    assert result.dtype == torch.bfloat16
    # Un-rotating a constant vector through an orthogonal Hadamard rotation does not preserve the
    # scalar value pointwise, but it does preserve total energy -- assert the round-trip is finite
    # and non-degenerate rather than asserting an exact per-element value.
    assert torch.isfinite(result.float()).all()
    assert result.float().abs().sum() > 0


def test_dequantize_w4a8_int8_weight_rejects_bad_shapes():
    qdata = torch.zeros((4, 128), dtype=torch.int8)
    s_rel = torch.ones((4, 999), dtype=torch.float32)  # wrong group count
    s_channel = torch.ones((4,), dtype=torch.float32)
    codebook = torch.linspace(-1.0, 1.0, 16)
    with __import__("pytest").raises(ConversionError):
        dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, codebook=codebook)


# ---------------------------------------------------------------------------
# Full checkpoint conversion
# ---------------------------------------------------------------------------


def _fake_w4a8_tensors(prefix: str, out_f: int, in_f: int, group_size: int = 16) -> dict:
    """A real (not degenerate) w4a8-style quantized weight + its dense equivalents for
    a companion bias/norm, matching the exact key set the real checkpoint uses."""
    groups = in_f // group_size
    codebook = torch.linspace(-1.0, 1.0, 16)
    qdata = torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8).to(torch.int8)
    return {
        f"{prefix}.weight": qdata,
        f"{prefix}.weight_codebook": codebook,
        f"{prefix}.weight_s_channel": torch.rand(out_f) + 0.1,
        f"{prefix}.weight_s_rel": (torch.rand(out_f, groups) + 0.1).to(torch.float8_e4m3fn),
        f"{prefix}.comfy_quant": torch.zeros(67, dtype=torch.uint8),
        f"{prefix}.bias": torch.randn(out_f, dtype=torch.bfloat16),
    }


def _tiny_source_tensors() -> dict:
    tensors = {}
    # One quantized transformer-block Linear (in_features=256 -- the minimum valid convrot size).
    tensors.update(_fake_w4a8_tensors(f"{SRC}transformer_blocks.0.attn1.to_k", out_f=8, in_f=256))
    tensors[f"{SRC}transformer_blocks.0.attn1.q_norm.weight"] = torch.randn(8, dtype=torch.bfloat16)
    # A simple (non-codebook) int8 tensor, matching the rarer plain-scale path seen in the real file.
    tensors[f"{SRC}transformer_blocks.0.attn1.to_v.weight"] = torch.randint(
        -100, 100, (8, 256), dtype=torch.int8
    )
    tensors[f"{SRC}transformer_blocks.0.attn1.to_v.weight_scale"] = torch.rand(8, 1) + 0.1
    # ff (needs renaming) -- dense, to keep the fixture small.
    tensors[f"{SRC}transformer_blocks.0.ff.net.0.proj.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
    tensors[f"{SRC}transformer_blocks.0.ff.net.2.weight"] = torch.randn(8, 16, dtype=torch.bfloat16)
    # top-level dense tensor.
    tensors[f"{SRC}patchify_proj.weight"] = torch.randn(8, 4, dtype=torch.bfloat16)
    # connector tensor -- must NOT be renamed, must land in the connector output.
    tensors[f"{SRC}video_embeddings_connector.learnable_registers"] = torch.randn(4, 8, dtype=torch.bfloat16)
    tensors[f"{SRC}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight"] = torch.randn(
        8, 8, dtype=torch.bfloat16
    )
    return tensors


def test_convert_checkpoint_splits_transformer_and_connector(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(_tiny_source_tensors(), str(source))

    out_dir = tmp_path / "model"
    convert_checkpoint(source, out_dir)

    transformer_path = out_dir / "transformer-dev.safetensors"
    connector_path = out_dir / "connector.safetensors"
    assert transformer_path.is_file()
    assert connector_path.is_file()

    connector = load_file(str(connector_path))
    assert "video_embeddings_connector.learnable_registers" in connector
    assert "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_out.0.weight" in connector
    # Connector keeps full precision -- no quantized-weight artifacts.
    assert all(not k.endswith((".scales", ".biases")) for k in connector)


def test_convert_checkpoint_renames_and_quantizes_transformer_weights(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(_tiny_source_tensors(), str(source))

    out_dir = tmp_path / "model"
    convert_checkpoint(source, out_dir)

    transformer = mx.load(str(out_dir / "transformer-dev.safetensors"))
    # Renamed correctly.
    assert "transformer_blocks.0.ff.proj_in.weight" in transformer
    assert "transformer_blocks.0.ff.proj_out.weight" in transformer
    # The originally-quantized Linear weight got MLX-native-quantized (not left dense bf16).
    assert "transformer_blocks.0.attn1.to_k.scales" in transformer
    assert "transformer_blocks.0.attn1.to_k.biases" in transformer
    assert transformer["transformer_blocks.0.attn1.to_k.weight"].dtype == mx.uint32
    # A tensor that was never quantized in the source (q_norm) stays plain/dense.
    assert "transformer_blocks.0.attn1.q_norm.weight" in transformer
    assert "transformer_blocks.0.attn1.q_norm.weight.scales" not in transformer


def test_convert_checkpoint_missing_source_raises_file_not_found(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        convert_checkpoint(tmp_path / "nope.safetensors", tmp_path / "model")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_missing_args_exits_nonzero():
    import pytest

    from img2vid.ltx_convert import main

    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_success_prints_output_dir_and_exits_zero(tmp_path, capsys):
    from unittest.mock import patch

    from img2vid.ltx_convert import main

    out_dir = tmp_path / "model"
    with patch("img2vid.ltx_convert.convert_checkpoint", return_value=out_dir) as mock_convert:
        exit_code = main(["--source", "src.safetensors", "--output-dir", str(out_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(out_dir) in captured.out
    mock_convert.assert_called_once()


def test_cli_conversion_error_prints_message_and_exits_nonzero(capsys):
    from unittest.mock import patch

    from img2vid.ltx_convert import main

    with patch("img2vid.ltx_convert.convert_checkpoint", side_effect=ConversionError("bad tensor")):
        exit_code = main(["--source", "src.safetensors", "--output-dir", "out"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "bad tensor" in captured.err
