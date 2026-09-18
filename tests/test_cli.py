from unittest.mock import patch

import pytest

from img2vid.cli import main
from img2vid.generate import GenerationError


def test_missing_prompt_arg_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--image", "photo.jpg"])
    assert exc_info.value.code != 0


def test_prompt_only_is_valid_t2v_invocation(tmp_path, capsys):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.cli.generate_video", return_value=output_path) as mock_gen:
        exit_code = main(["--prompt", "a prompt", "--output", str(output_path)])
    assert exit_code == 0
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs["image_path"] is None


def test_nonexistent_image_file_prints_error_and_exits_nonzero(tmp_path, capsys):
    missing = tmp_path / "nope.jpg"
    with patch("img2vid.cli.generate_video", side_effect=FileNotFoundError(f"Input image not found: {missing}")):
        exit_code = main(["--image", str(missing), "--prompt", "a prompt"])
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_success_prints_output_path_and_exits_zero(tmp_path, capsys):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    output_path = tmp_path / "out.mp4"

    with patch("img2vid.cli.generate_video", return_value=output_path) as mock_gen:
        exit_code = main(["--image", str(image_path), "--prompt", "a prompt", "--output", str(output_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(output_path) in captured.out
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs["image_path"] == image_path


def test_generation_error_prints_message_and_exits_nonzero(tmp_path, capsys):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    with patch("img2vid.cli.generate_video", side_effect=GenerationError("ltx-2-mlx boom")):
        exit_code = main(["--image", str(image_path), "--prompt", "a prompt"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "ltx-2-mlx boom" in captured.err


def test_frame_rate_and_cfg_scale_forwarded(tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.cli.generate_video", return_value=output_path) as mock_gen:
        main(["--prompt", "a prompt", "--frame-rate", "30", "--cfg-scale", "4.5", "--output", str(output_path)])
    assert mock_gen.call_args.kwargs["frame_rate"] == 30.0
    assert mock_gen.call_args.kwargs["cfg_scale"] == 4.5


def test_dev_flag_defaults_to_false(tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.cli.generate_video", return_value=output_path) as mock_gen:
        main(["--prompt", "a prompt", "--output", str(output_path)])
    assert mock_gen.call_args.kwargs["dev"] is False


def test_dev_flag_forwarded_when_set(tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.cli.generate_video", return_value=output_path) as mock_gen:
        main(["--prompt", "a prompt", "--dev", "--output", str(output_path)])
    assert mock_gen.call_args.kwargs["dev"] is True
