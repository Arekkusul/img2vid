from unittest.mock import patch

import pytest

from img2vid.cli import main
from img2vid.generate import GenerationError


def test_missing_image_arg_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--prompt", "a prompt"])
    assert exc_info.value.code != 0


def test_missing_prompt_arg_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--image", "photo.jpg"])
    assert exc_info.value.code != 0


def test_nonexistent_image_file_prints_error_and_exits_nonzero(tmp_path, capsys):
    missing = tmp_path / "nope.jpg"
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


def test_generation_error_prints_message_and_exits_nonzero(tmp_path, capsys):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    with patch("img2vid.cli.generate_video", side_effect=GenerationError("mlxgen boom")):
        exit_code = main(["--image", str(image_path), "--prompt", "a prompt"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "mlxgen boom" in captured.err
