from unittest.mock import patch

import pytest

from img2vid.image_cli import main
from img2vid.image_generate import ImageGenerationError


def test_missing_prompt_arg_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_success_prints_output_path_and_exits_zero(tmp_path, capsys):
    output_path = tmp_path / "out.png"

    with patch("img2vid.image_cli.generate_image", return_value=output_path) as mock_gen:
        exit_code = main(["--prompt", "a prompt", "--output", str(output_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(output_path) in captured.out
    mock_gen.assert_called_once()


def test_image_generation_error_prints_message_and_exits_nonzero(capsys):
    with patch("img2vid.image_cli.generate_image", side_effect=ImageGenerationError("krea-gen boom")):
        exit_code = main(["--prompt", "a prompt"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "krea-gen boom" in captured.err


def test_value_error_prints_message_and_exits_nonzero(capsys):
    with patch("img2vid.image_cli.generate_image", side_effect=ValueError("bad input")):
        exit_code = main(["--prompt", "a prompt"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "bad input" in captured.err


def test_edit_source_forwarded_to_generate_image(tmp_path):
    output_path = tmp_path / "out.png"
    source = tmp_path / "src.png"

    with patch("img2vid.image_cli.generate_image", return_value=output_path) as mock_gen:
        main(["--prompt", "an edit", "--edit-source", str(source), "--lora", "/x/lora.safetensors"])

    _, kwargs = mock_gen.call_args
    assert kwargs["edit_image_path"] == source
    assert str(kwargs["lora_path"]) == "/x/lora.safetensors"
