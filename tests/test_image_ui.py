from unittest.mock import patch

import gradio as gr
import pytest

from img2vid.image_generate import ImageGenerationError
from img2vid.image_ui import _run


@pytest.mark.parametrize("prompt", ["", "   "])
def test_run_raises_gr_error_when_prompt_empty(prompt):
    with pytest.raises(gr.Error):
        _run(None, prompt, 1024, 1024, 30, 4.0, None, None)


def test_run_allows_missing_reference_image_for_t2i(tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_ui.generate_image", return_value=output_path) as mock_gen:
        result = _run(None, "a prompt", 1024, 1024, 30, 4.0, None, None)
    assert result == str(output_path)
    assert mock_gen.call_args.kwargs["edit_image_path"] is None


def test_run_forwards_reference_image_for_edit_mode(tmp_path):
    output_path = tmp_path / "out.png"
    source = tmp_path / "src.png"
    source.write_bytes(b"fake")
    with patch("img2vid.image_ui.generate_image", return_value=output_path) as mock_gen:
        _run(str(source), "an edit", 1024, 1024, 30, 4.0, None, None)
    assert mock_gen.call_args.kwargs["edit_image_path"] == str(source)


def test_run_forwards_field_values(tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_ui.generate_image", return_value=output_path) as mock_gen:
        _run(None, "a prompt", 640, 480, 20, 3.5, 7, "blurry")
    kwargs = mock_gen.call_args.kwargs
    assert kwargs["width"] == 640
    assert kwargs["height"] == 480
    assert kwargs["steps"] == 20
    assert kwargs["guidance"] == 3.5
    assert kwargs["seed"] == 7
    assert kwargs["negative_prompt"] == "blurry"


def test_run_treats_blank_seed_as_none(tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_ui.generate_image", return_value=output_path) as mock_gen:
        _run(None, "a prompt", 1024, 1024, 30, 4.0, None, None)
    assert mock_gen.call_args.kwargs["seed"] is None


def test_run_preserves_explicit_seed_zero(tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_ui.generate_image", return_value=output_path) as mock_gen:
        _run(None, "a prompt", 1024, 1024, 30, 4.0, 0, None)
    assert mock_gen.call_args.kwargs["seed"] == 0


def test_run_reraises_image_generation_error_as_gr_error():
    with patch("img2vid.image_ui.generate_image", side_effect=ImageGenerationError("krea-gen boom")):
        with pytest.raises(gr.Error, match="krea-gen boom"):
            _run(None, "a prompt", 1024, 1024, 30, 4.0, None, None)
