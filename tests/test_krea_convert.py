import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from img2vid.krea_convert import (
    ConversionError,
    Krea2Config,
    convert_transformer,
    expected_transformer_keys,
    main,
    remap_key,
)

SRC = "model.diffusion_model."


@pytest.mark.parametrize(
    "source,target",
    [
        (f"{SRC}first.weight", "img_in.weight"),
        (f"{SRC}first.bias", "img_in.bias"),
        (f"{SRC}txtmlp.0.scale", "txt_in.norm.weight"),
        (f"{SRC}txtmlp.1.weight", "txt_in.linear_1.weight"),
        (f"{SRC}txtmlp.1.bias", "txt_in.linear_1.bias"),
        (f"{SRC}txtmlp.3.weight", "txt_in.linear_2.weight"),
        (f"{SRC}txtmlp.3.bias", "txt_in.linear_2.bias"),
        (f"{SRC}tmlp.0.weight", "time_embed.linear_1.weight"),
        (f"{SRC}tmlp.0.bias", "time_embed.linear_1.bias"),
        (f"{SRC}tmlp.2.weight", "time_embed.linear_2.weight"),
        (f"{SRC}tmlp.2.bias", "time_embed.linear_2.bias"),
        (f"{SRC}tproj.1.weight", "time_mod_proj.weight"),
        (f"{SRC}tproj.1.bias", "time_mod_proj.bias"),
        (f"{SRC}txtfusion.projector.weight", "text_fusion.projector.weight"),
        (f"{SRC}last.linear.weight", "final_layer.linear.weight"),
        (f"{SRC}last.linear.bias", "final_layer.linear.bias"),
        (f"{SRC}last.norm.scale", "final_layer.norm.weight"),
        (f"{SRC}last.modulation.lin", "final_layer.scale_shift_table"),
    ],
)
def test_remap_key_top_level(source, target):
    assert remap_key(source) == target


@pytest.mark.parametrize(
    "suffix,target_suffix",
    [
        ("attn.wq.weight", "attn.to_q.weight"),
        ("attn.wk.weight", "attn.to_k.weight"),
        ("attn.wv.weight", "attn.to_v.weight"),
        ("attn.wo.weight", "attn.to_out.0.weight"),
        ("attn.gate.weight", "attn.to_gate.weight"),
        ("attn.qknorm.qnorm.scale", "attn.norm_q.weight"),
        ("attn.qknorm.knorm.scale", "attn.norm_k.weight"),
        ("mlp.gate.weight", "ff.gate.weight"),
        ("mlp.up.weight", "ff.up.weight"),
        ("mlp.down.weight", "ff.down.weight"),
        ("prenorm.scale", "norm1.weight"),
        ("postnorm.scale", "norm2.weight"),
    ],
)
def test_remap_key_single_stream_block(suffix, target_suffix):
    assert remap_key(f"{SRC}blocks.5.{suffix}") == f"transformer_blocks.5.{target_suffix}"


def test_remap_key_single_stream_block_mod_lin_is_scale_shift_table():
    assert remap_key(f"{SRC}blocks.5.mod.lin") == "transformer_blocks.5.scale_shift_table"


@pytest.mark.parametrize("family", ["layerwise_blocks", "refiner_blocks"])
def test_remap_key_text_fusion_block(family):
    assert (
        remap_key(f"{SRC}txtfusion.{family}.1.attn.wq.weight")
        == f"text_fusion.{family}.1.attn.to_q.weight"
    )
    assert (
        remap_key(f"{SRC}txtfusion.{family}.1.prenorm.scale")
        == f"text_fusion.{family}.1.norm1.weight"
    )


def test_remap_key_weight_scale_returns_none():
    assert remap_key(f"{SRC}blocks.0.attn.wq.weight_scale") is None


def test_remap_key_wrong_prefix_returns_none():
    assert remap_key("some.other.model.weight") is None


def test_remap_key_unrecognized_block_suffix_raises():
    with pytest.raises(ConversionError, match="unrecognized"):
        remap_key(f"{SRC}blocks.0.attn.totally_unknown.weight")


def test_remap_key_unrecognized_top_level_raises():
    with pytest.raises(ConversionError, match="unrecognized"):
        remap_key(f"{SRC}some_new_top_level_thing.weight")


def test_expected_transformer_keys_scales_with_layer_counts():
    small = Krea2Config(num_layers=1, num_layerwise_text_blocks=1, num_refiner_text_blocks=1)
    big = Krea2Config(num_layers=2, num_layerwise_text_blocks=1, num_refiner_text_blocks=1)
    assert len(expected_transformer_keys(big)) > len(expected_transformer_keys(small))
    assert "transformer_blocks.1.scale_shift_table" in expected_transformer_keys(big)
    assert "transformer_blocks.1.scale_shift_table" not in expected_transformer_keys(small)


def test_expected_transformer_keys_text_blocks_have_no_scale_shift_table():
    cfg = Krea2Config(num_layers=1, num_layerwise_text_blocks=1, num_refiner_text_blocks=1)
    keys = expected_transformer_keys(cfg)
    assert "text_fusion.layerwise_blocks.0.scale_shift_table" not in keys
    assert "text_fusion.layerwise_blocks.0.norm1.weight" in keys


TINY_CFG = Krea2Config(
    hidden_size=8,
    num_attention_heads=2,
    num_kv_heads=1,
    attention_head_dim=4,
    num_layers=1,
    intermediate_size=16,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=0,
    text_hidden_dim=6,
    text_intermediate_size=12,
    in_channels=8,
)


def _tiny_source_tensors():
    """A minimal but complete source-format checkpoint matching TINY_CFG, including one
    quantized (fp8 + weight_scale) tensor and several plain dense bf16 tensors."""
    h, ic = TINY_CFG.hidden_size, TINY_CFG.in_channels
    th = TINY_CFG.text_hidden_dim
    tensors = {
        f"{SRC}first.weight": torch.randn(h, ic, dtype=torch.bfloat16),
        f"{SRC}first.bias": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}txtmlp.0.scale": torch.randn(th, dtype=torch.bfloat16),
        f"{SRC}txtmlp.1.weight": torch.randn(h, th, dtype=torch.bfloat16),
        f"{SRC}txtmlp.1.bias": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}txtmlp.3.weight": torch.randn(h, h, dtype=torch.bfloat16),
        f"{SRC}txtmlp.3.bias": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}tmlp.0.weight": torch.randn(h, 256, dtype=torch.bfloat16),
        f"{SRC}tmlp.0.bias": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}tmlp.2.weight": torch.randn(h, h, dtype=torch.bfloat16),
        f"{SRC}tmlp.2.bias": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}tproj.1.weight": torch.randn(6 * h, h, dtype=torch.bfloat16),
        f"{SRC}tproj.1.bias": torch.randn(6 * h, dtype=torch.bfloat16),
        f"{SRC}txtfusion.projector.weight": torch.randn(1, 12, dtype=torch.bfloat16),
        f"{SRC}last.linear.weight": torch.randn(ic, h, dtype=torch.bfloat16),
        f"{SRC}last.linear.bias": torch.randn(ic, dtype=torch.bfloat16),
        f"{SRC}last.norm.scale": torch.randn(h, dtype=torch.bfloat16),
        f"{SRC}last.modulation.lin": torch.randn(2, h, dtype=torch.bfloat16),
    }
    # one single-stream block (fp8-quantized attn.wq, everything else dense)
    tensors[f"{SRC}blocks.0.attn.wq.weight"] = torch.full((h, h), 2.0, dtype=torch.float8_e4m3fn)
    tensors[f"{SRC}blocks.0.attn.wq.weight_scale"] = torch.tensor(3.5, dtype=torch.bfloat16)
    for suffix, shape in [
        ("attn.wk.weight", (4, h)),
        ("attn.wv.weight", (4, h)),
        ("attn.wo.weight", (h, h)),
        ("attn.gate.weight", (h, h)),
        ("attn.qknorm.qnorm.scale", (4,)),
        ("attn.qknorm.knorm.scale", (4,)),
        ("mlp.gate.weight", (16, h)),
        ("mlp.up.weight", (16, h)),
        ("mlp.down.weight", (h, 16)),
        ("prenorm.scale", (h,)),
        ("postnorm.scale", (h,)),
        ("mod.lin", (6 * h,)),
    ]:
        tensors[f"{SRC}blocks.0.{suffix}"] = torch.randn(*shape, dtype=torch.bfloat16)
    # one text-fusion (layerwise) block
    for suffix, shape in [
        ("attn.wq.weight", (th, th)),
        ("attn.wk.weight", (th, th)),
        ("attn.wv.weight", (th, th)),
        ("attn.wo.weight", (th, th)),
        ("attn.gate.weight", (th, th)),
        ("attn.qknorm.qnorm.scale", (2,)),
        ("attn.qknorm.knorm.scale", (2,)),
        ("mlp.gate.weight", (12, th)),
        ("mlp.up.weight", (12, th)),
        ("mlp.down.weight", (th, 12)),
        ("prenorm.scale", (th,)),
        ("postnorm.scale", (th,)),
    ]:
        tensors[f"{SRC}txtfusion.layerwise_blocks.0.{suffix}"] = torch.randn(*shape, dtype=torch.bfloat16)
    return tensors


def test_convert_transformer_produces_exact_expected_key_set(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(_tiny_source_tensors(), str(source))

    out_dir = convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)

    assert out_dir == tmp_path / "snapshot" / "transformer"
    weights_path = out_dir / "diffusion_pytorch_model.safetensors"
    assert weights_path.is_file()
    converted = load_file(str(weights_path))
    assert set(converted.keys()) == expected_transformer_keys(TINY_CFG)


def test_convert_transformer_writes_valid_config_json(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(_tiny_source_tensors(), str(source))

    out_dir = convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)

    config = json.loads((out_dir / "config.json").read_text())
    assert isinstance(config, dict)


def test_convert_transformer_dequantizes_fp8_correctly(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(_tiny_source_tensors(), str(source))

    out_dir = convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)
    converted = load_file(str(out_dir / "diffusion_pytorch_model.safetensors"))

    wq = converted["transformer_blocks.0.attn.to_q.weight"]
    assert wq.dtype == torch.bfloat16
    # source was a constant 2.0 fp8 tensor with scale 3.5 -> dequantized should be ~7.0
    assert torch.allclose(wq.float(), torch.full_like(wq.float(), 7.0), atol=0.1)


def test_convert_transformer_passes_through_dense_tensors_as_bf16(tmp_path):
    source_tensors = _tiny_source_tensors()
    source = tmp_path / "source.safetensors"
    save_file(source_tensors, str(source))

    out_dir = convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)
    converted = load_file(str(out_dir / "diffusion_pytorch_model.safetensors"))

    original = source_tensors[f"{SRC}last.norm.scale"]
    result = converted["final_layer.norm.weight"]
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, original)


def test_convert_transformer_missing_source_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_transformer(tmp_path / "nope.safetensors", tmp_path / "snapshot", cfg=TINY_CFG)


def test_convert_transformer_incomplete_source_raises_conversion_error(tmp_path):
    tensors = _tiny_source_tensors()
    del tensors[f"{SRC}blocks.0.attn.wo.weight"]
    source = tmp_path / "source.safetensors"
    save_file(tensors, str(source))

    with pytest.raises(ConversionError, match="missing"):
        convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)


def test_cli_missing_args_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_success_prints_transformer_dir_and_exits_zero(tmp_path, capsys):
    out_dir = tmp_path / "snapshot" / "transformer"
    with patch("img2vid.krea_convert.convert_transformer", return_value=out_dir) as mock_convert:
        exit_code = main(["--source", "src.safetensors", "--output-dir", str(tmp_path / "snapshot")])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(out_dir) in captured.out
    mock_convert.assert_called_once()


def test_cli_conversion_error_prints_message_and_exits_nonzero(capsys):
    with patch("img2vid.krea_convert.convert_transformer", side_effect=ConversionError("bad mapping")):
        exit_code = main(["--source", "src.safetensors", "--output-dir", "out"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "bad mapping" in captured.err


def test_cli_missing_source_prints_message_and_exits_nonzero(capsys):
    with patch("img2vid.krea_convert.convert_transformer", side_effect=FileNotFoundError("nope")):
        exit_code = main(["--source", "src.safetensors", "--output-dir", "out"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "nope" in captured.err


def test_convert_transformer_raises_on_fp8_tensor_missing_scale(tmp_path):
    tensors = _tiny_source_tensors()
    del tensors[f"{SRC}blocks.0.attn.wq.weight_scale"]
    source = tmp_path / "source.safetensors"
    save_file(tensors, str(source))

    with pytest.raises(ConversionError, match="weight_scale"):
        convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)


def test_convert_transformer_raises_conversion_error_on_unreadable_source(tmp_path):
    source = tmp_path / "not_really_safetensors.safetensors"
    source.write_bytes(b"this is not a valid safetensors file")

    with pytest.raises(ConversionError, match="safetensors"):
        convert_transformer(source, tmp_path / "snapshot", cfg=TINY_CFG)
