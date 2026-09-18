from unittest.mock import patch

import gradio as gr
import pytest

from img2vid.generate import GenerationError
from img2vid.ui import _run


@pytest.mark.parametrize("prompt", ["", "   "])
def test_run_raises_gr_error_when_prompt_empty(tmp_path, prompt):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    with pytest.raises(gr.Error):
        _run(str(image_path), prompt, 704, 480, 97, 30, 3.0, 24.0, False, None)


def test_run_allows_missing_image_for_t2v(tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        result = _run(None, "a prompt", 704, 480, 97, 30, 3.0, 24.0, False, None)
    assert result == str(output_path)
    assert mock_gen.call_args.kwargs["image_path"] is None


def test_run_forwards_field_values_to_generate_video(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        result = _run(str(image_path), "a prompt", 640, 360, 49, 20, 4.0, 30.0, True, 7)

    assert result == str(output_path)
    kwargs = mock_gen.call_args.kwargs
    assert kwargs["image_path"] == str(image_path)
    assert kwargs["width"] == 640
    assert kwargs["height"] == 360
    assert kwargs["frames"] == 49
    assert kwargs["steps"] == 20
    assert kwargs["cfg_scale"] == 4.0
    assert kwargs["frame_rate"] == 30.0
    assert kwargs["dev"] is True
    assert kwargs["seed"] == 7


def test_run_defaults_dev_to_false(tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        _run(None, "a prompt", 704, 480, 97, 30, 3.0, 24.0, False, None)
    assert mock_gen.call_args.kwargs["dev"] is False


def test_run_reraises_generation_error_as_gr_error(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    with patch("img2vid.ui.generate_video", side_effect=GenerationError("ltx-2-mlx boom")):
        with pytest.raises(gr.Error, match="ltx-2-mlx boom"):
            _run(str(image_path), "a prompt", 704, 480, 97, 30, 3.0, 24.0, False, None)


def test_run_treats_blank_seed_as_none(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        _run(str(image_path), "a prompt", 704, 480, 97, 30, 3.0, 24.0, False, None)

    assert mock_gen.call_args.kwargs["seed"] is None


def test_run_preserves_explicit_seed_zero(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.ui.generate_video", return_value=output_path) as mock_gen:
        _run(str(image_path), "a prompt", 704, 480, 97, 30, 3.0, 24.0, False, 0)

    assert mock_gen.call_args.kwargs["seed"] == 0
