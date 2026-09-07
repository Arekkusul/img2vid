from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from img2vid.generate import GenerationError, generate_video


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "input.jpg"
    p.write_bytes(b"fake-image-bytes")
    return p


def _mock_success(output_path: Path, stdout: str = "done"):
    def _run(argv, **kwargs):
        output_path.write_bytes(b"fake-video-bytes")
        return MagicMock(returncode=0, stdout=stdout, stderr="")
    return _run


def test_rejects_missing_image_before_invoking_subprocess(tmp_path):
    missing = tmp_path / "nope.jpg"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError):
            generate_video(missing, "a prompt", output_path=tmp_path / "out.mp4")
    mock_run.assert_not_called()


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_rejects_empty_prompt_before_invoking_subprocess(image_path, tmp_path, prompt):
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError):
            generate_video(image_path, prompt, output_path=tmp_path / "out.mp4")
    mock_run.assert_not_called()


def test_raises_generation_error_with_stderr_on_nonzero_exit(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="mlxgen: boom")
        with pytest.raises(GenerationError, match="boom"):
            generate_video(image_path, "a prompt", output_path=output_path)


def test_raises_when_exit_zero_but_output_file_missing(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(GenerationError, match="output file"):
            generate_video(image_path, "a prompt", output_path=output_path)


def test_raises_when_output_file_exists_but_empty(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"")
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(GenerationError, match="empty"):
            generate_video(image_path, "a prompt", output_path=output_path)


def test_returns_output_path_on_success(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)):
        result = generate_video(image_path, "a prompt", output_path=output_path)
    assert result == output_path
    assert result.stat().st_size > 0


def test_seed_omitted_from_argv_when_none(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path, seed=None)
    argv = mock_run.call_args.args[0]
    assert "--seed" not in argv


def test_seed_included_when_set(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path, seed=42)
    argv = mock_run.call_args.args[0]
    assert "--seed" in argv
    assert "42" in argv


def test_low_ram_present_by_default(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path)
    argv = mock_run.call_args.args[0]
    assert "--low-ram" in argv


def test_low_ram_absent_when_disabled(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path, low_ram=False)
    argv = mock_run.call_args.args[0]
    assert "--low-ram" not in argv


def test_env_contains_dyld_library_path(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path)
    env = mock_run.call_args.kwargs["env"]
    assert env["DYLD_LIBRARY_PATH"] == "/opt/homebrew/opt/expat/lib"


def test_timeout_raises_generation_error(image_path, tmp_path):
    import subprocess as sp

    output_path = tmp_path / "out.mp4"
    with patch(
        "img2vid.generate.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="mlxgen", timeout=5),
    ):
        with pytest.raises(GenerationError, match="did not finish"):
            generate_video(image_path, "a prompt", output_path=output_path, timeout=5)


def test_timeout_forwarded_to_subprocess_run(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(image_path, "a prompt", output_path=output_path, timeout=42)
    assert mock_run.call_args.kwargs["timeout"] == 42


def test_missing_mlxgen_binary_gives_actionable_message(image_path, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(GenerationError, match="scripts/setup.sh"):
            generate_video(image_path, "a prompt", output_path=output_path)


def test_default_output_path_derived_when_not_given(image_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _run(argv, **kwargs):
        out_idx = argv.index("--output")
        Path(argv[out_idx + 1]).write_bytes(b"fake-video-bytes")
        return MagicMock(returncode=0, stdout="done", stderr="")

    with patch("img2vid.generate.subprocess.run", side_effect=_run):
        result = generate_video(image_path, "a prompt")
    assert result.exists()
    assert result.suffix == ".mp4"
