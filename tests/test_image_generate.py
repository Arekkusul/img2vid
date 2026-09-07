from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from img2vid.image_generate import ImageGenerationError, _default_krea_bin, generate_image


@pytest.fixture
def snapshot_dir(tmp_path):
    d = tmp_path / "snapshot"
    d.mkdir()
    return d


@pytest.fixture
def edit_image(tmp_path):
    p = tmp_path / "source.png"
    p.write_bytes(b"fake-image-bytes")
    return p


def _mock_success(output_path: Path, stdout: str = "done"):
    def _run(argv, **kwargs):
        output_path.write_bytes(b"fake-png-bytes")
        return MagicMock(returncode=0, stdout=stdout, stderr="")
    return _run


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_rejects_empty_prompt_before_invoking_subprocess(snapshot_dir, tmp_path, prompt):
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError):
            generate_image(prompt, output_path=tmp_path / "out.png", snapshot=snapshot_dir)
    mock_run.assert_not_called()


def test_rejects_missing_snapshot_before_invoking_subprocess(tmp_path):
    missing = tmp_path / "nope"
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError, match="snapshot"):
            generate_image("a prompt", output_path=tmp_path / "out.png", snapshot=missing)
    mock_run.assert_not_called()


def test_rejects_missing_edit_source_before_invoking_subprocess(snapshot_dir, tmp_path):
    missing = tmp_path / "nope.png"
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        with pytest.raises(FileNotFoundError, match="Edit source"):
            generate_image(
                "a prompt", output_path=tmp_path / "out.png", snapshot=snapshot_dir, edit_image_path=missing
            )
    mock_run.assert_not_called()


def test_raises_image_generation_error_with_stderr_on_nonzero_exit(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="krea-gen: boom")
        with pytest.raises(ImageGenerationError, match="boom"):
            generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)


def test_raises_when_exit_zero_but_output_file_missing(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(ImageGenerationError, match="output file"):
            generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)


def test_raises_when_output_file_exists_but_empty(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"")
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        with pytest.raises(ImageGenerationError, match="empty"):
            generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)


def test_returns_output_path_on_success(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run", side_effect=_mock_success(output_path)):
        result = generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)
    assert result == output_path
    assert result.stat().st_size > 0


def test_seed_omitted_from_argv_when_none(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir, seed=None)
    argv = mock_run.call_args.args[0]
    assert "--seed" not in argv


def test_seed_included_when_set(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir, seed=42)
    argv = mock_run.call_args.args[0]
    assert "--seed" in argv
    assert "42" in argv


def test_edit_source_flags_omitted_in_t2i_mode(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)
    argv = mock_run.call_args.args[0]
    assert "--edit-source" not in argv
    assert "--lora" not in argv
    assert "--turbo-edit" not in argv


def test_edit_source_and_lora_included_in_edit_mode(snapshot_dir, edit_image, tmp_path):
    output_path = tmp_path / "out.png"
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"fake-lora")
    with patch("img2vid.image_generate.subprocess.run", side_effect=_mock_success(output_path)) as mock_run:
        generate_image(
            "an edit instruction",
            output_path=output_path,
            snapshot=snapshot_dir,
            edit_image_path=edit_image,
            lora_path=lora,
            turbo_edit=True,
        )
    argv = mock_run.call_args.args[0]
    assert "--edit-source" in argv
    assert str(edit_image) in argv
    assert "--lora" in argv
    assert str(lora) in argv
    assert "--turbo-edit" in argv


def test_timeout_raises_image_generation_error(snapshot_dir, tmp_path):
    import subprocess as sp

    output_path = tmp_path / "out.png"
    with patch(
        "img2vid.image_generate.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="krea-gen", timeout=5),
    ):
        with pytest.raises(ImageGenerationError, match="did not finish"):
            generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir, timeout=5)


def test_missing_krea_bin_gives_actionable_message(snapshot_dir, tmp_path):
    output_path = tmp_path / "out.png"
    with patch("img2vid.image_generate.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(ImageGenerationError, match="build_krea_gen.sh"):
            generate_image("a prompt", output_path=output_path, snapshot=snapshot_dir)


def test_default_output_path_derived_when_not_given(snapshot_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _run(argv, **kwargs):
        out_idx = argv.index("--output")
        Path(argv[out_idx + 1]).write_bytes(b"fake-png-bytes")
        return MagicMock(returncode=0, stdout="done", stderr="")

    with patch("img2vid.image_generate.subprocess.run", side_effect=_run):
        result = generate_image("a prompt", snapshot=snapshot_dir)
    assert result.exists()
    assert result.suffix == ".png"


def test_default_krea_bin_prefers_built_release_binary(tmp_path):
    repo_root = tmp_path / "repo"
    bin_dir = repo_root / "krea-gen" / "target" / "release"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "krea-gen"
    binary.write_text("")

    assert _default_krea_bin(repo_root) == str(binary)


def test_default_krea_bin_falls_back_to_path_lookup_when_not_built(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert _default_krea_bin(repo_root) == "krea-gen"


def test_rejects_turbo_edit_without_edit_source(snapshot_dir, tmp_path):
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError, match="edit_image_path"):
            generate_image(
                "a prompt", output_path=tmp_path / "out.png", snapshot=snapshot_dir, turbo_edit=True
            )
    mock_run.assert_not_called()


def test_rejects_lora_without_edit_source(snapshot_dir, tmp_path):
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"fake-lora")
    with patch("img2vid.image_generate.subprocess.run") as mock_run:
        with pytest.raises(ValueError, match="edit_image_path"):
            generate_image(
                "a prompt", output_path=tmp_path / "out.png", snapshot=snapshot_dir, lora_path=lora
            )
    mock_run.assert_not_called()
