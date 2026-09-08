from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from img2vid.generate import GenerationError, _default_ltx_bin, generate_video


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "input.jpg"
    p.write_bytes(b"fake-image-bytes")
    return p


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "ltx-model"
    d.mkdir()
    return d


def _mock_success(output_path: Path, stdout: str = "done"):
    def _run(argv, **kwargs):
        output_path.write_bytes(b"fake-video-bytes")
        return MagicMock(returncode=0, stdout=stdout, stderr="")

    return _run


def test_rejects_empty_prompt_before_invoking_subprocess(model_dir, tmp_path):
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError):
            generate_video("   ", output_path=tmp_path / "out.mp4", model=model_dir)
    mock_run.assert_not_called()


def test_rejects_missing_model_dir_before_invoking_subprocess(tmp_path):
    missing = tmp_path / "no-such-model"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError, match="img2vid-ltx-convert"):
            generate_video("a prompt", output_path=tmp_path / "out.mp4", model=missing)
    mock_run.assert_not_called()


def test_rejects_missing_image_before_invoking_subprocess(model_dir, tmp_path):
    missing = tmp_path / "nope.jpg"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError):
            generate_video(
                "a prompt", image_path=missing, output_path=tmp_path / "out.mp4", model=model_dir
            )
    mock_run.assert_not_called()


def test_rejects_frame_count_not_matching_grid(model_dir, tmp_path):
    with patch("img2vid.generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError, match="frames"):
            generate_video("a prompt", output_path=tmp_path / "out.mp4", model=model_dir, frames=50)
    mock_run.assert_not_called()


def test_t2v_omits_image_flag(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir)
    argv = mock_run.call_args.args[0]
    assert "--image" not in argv
    assert "--one-stage" in argv


def test_i2v_includes_image_flag(image_path, model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", image_path=image_path, output_path=output_path, model=model_dir)
    argv = mock_run.call_args.args[0]
    assert "--image" in argv
    assert str(image_path) in argv


def test_raises_generation_error_with_stderr_on_nonzero_exit(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ltx-2-mlx: boom")
        with pytest.raises(GenerationError, match="boom"):
            generate_video("a prompt", output_path=output_path, model=model_dir)


def test_raises_when_exit_zero_but_output_file_missing(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(GenerationError, match="output file"):
            generate_video("a prompt", output_path=output_path, model=model_dir)


def test_raises_when_output_file_exists_but_empty(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"")
    with patch("img2vid.generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(GenerationError, match="empty"):
            generate_video("a prompt", output_path=output_path, model=model_dir)


def test_returns_output_path_on_success(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)):
        result = generate_video("a prompt", output_path=output_path, model=model_dir)
    assert result == output_path
    assert result.stat().st_size > 0


def test_seed_omitted_from_argv_when_none(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir, seed=None)
    argv = mock_run.call_args.args[0]
    assert "--seed" not in argv


def test_seed_included_when_set(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir, seed=42)
    argv = mock_run.call_args.args[0]
    assert "--seed" in argv
    assert "42" in argv


def test_low_ram_present_by_default(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir)
    argv = mock_run.call_args.args[0]
    assert "--low-ram" in argv


def test_low_ram_absent_when_disabled(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir, low_ram=False)
    argv = mock_run.call_args.args[0]
    assert "--low-ram" not in argv


def test_frames_height_width_steps_cfg_frame_rate_forwarded(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video(
            "a prompt",
            output_path=output_path,
            model=model_dir,
            width=320,
            height=320,
            frames=25,
            steps=16,
            cfg_scale=4.0,
            frame_rate=30.0,
        )
    argv = mock_run.call_args.args[0]
    assert "320" in argv
    assert "25" in argv
    assert "16" in argv
    assert "4.0" in argv
    assert "30.0" in argv


def test_timeout_raises_generation_error(model_dir, tmp_path):
    import subprocess as sp

    output_path = tmp_path / "out.mp4"
    with patch(
        "img2vid.generate.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="ltx-2-mlx", timeout=5),
    ):
        with pytest.raises(GenerationError, match="did not finish"):
            generate_video("a prompt", output_path=output_path, model=model_dir, timeout=5)


def test_timeout_forwarded_to_subprocess_run(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_video("a prompt", output_path=output_path, model=model_dir, timeout=42)
    assert mock_run.call_args.kwargs["timeout"] == 42


def test_missing_ltx_binary_gives_actionable_message(model_dir, tmp_path):
    output_path = tmp_path / "out.mp4"
    with patch("img2vid.generate.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(GenerationError, match="ltx-2-mlx-upstream"):
            generate_video("a prompt", output_path=output_path, model=model_dir)


def test_default_output_path_derived_when_not_given(model_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _run(argv, **kwargs):
        out_idx = argv.index("--output")
        Path(argv[out_idx + 1]).write_bytes(b"fake-video-bytes")
        return MagicMock(returncode=0, stdout="done", stderr="")

    with patch("img2vid.generate.subprocess.run", side_effect=_run):
        result = generate_video("a prompt", model=model_dir)
    assert result.exists()


def test_default_ltx_bin_prefers_upstream_venv(tmp_path):
    repo_root = tmp_path
    bin_dir = repo_root / "ltx-2-mlx-upstream" / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    fake_bin = bin_dir / "ltx-2-mlx"
    fake_bin.write_text("")

    assert _default_ltx_bin(repo_root) == str(fake_bin)


def test_default_ltx_bin_falls_back_to_path_lookup_when_missing(tmp_path):
    assert _default_ltx_bin(tmp_path) == "ltx-2-mlx"
