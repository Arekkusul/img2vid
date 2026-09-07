from unittest.mock import patch

import gradio as gr
import pytest

from img2vid.generate import GenerationError
from img2vid.ui import _run


def test_run_raises_gr_error_when_image_missing():
    with pytest.raises(gr.Error):
        _run(None, "a prompt", 832, 480, 81, 25, None)


@pytest.mark.parametrize("prompt", ["", "   "])
def test_run_raises_gr_error_when_prompt_empty(tmp_path, prompt):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    with pytest.raises(gr.Error):
        _run(str(image_path), prompt, 832, 480, 81, 25, None)


def test_run_forwards_field_values_to_generate_video(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        result = _run(str(image_path), "a prompt", 640, 360, 49, 20, 7)

    assert result == str(output_path)
    _, kwargs = mock_gen.call_args
    assert kwargs["width"] == 640
    assert kwargs["height"] == 360
    assert kwargs["frames"] == 49
    assert kwargs["steps"] == 20
    assert kwargs["seed"] == 7


def test_run_reraises_generation_error_as_gr_error(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    with patch("img2vid.ui.generate_video", side_effect=GenerationError("mlxgen boom")):
        with pytest.raises(gr.Error, match="mlxgen boom"):
            _run(str(image_path), "a prompt", 832, 480, 81, 25, None)


def test_run_treats_blank_seed_as_none(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        _run(str(image_path), "a prompt", 832, 480, 81, 25, None)

    assert mock_gen.call_args.kwargs["seed"] is None


def test_run_preserves_explicit_seed_zero(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        _run(str(image_path), "a prompt", 832, 480, 81, 25, 0)

    assert mock_gen.call_args.kwargs["seed"] == 0
