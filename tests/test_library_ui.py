import os

from img2vid.library_ui import _list_videos


def test_list_videos_returns_empty_for_missing_dir(tmp_path):
    assert _list_videos(tmp_path / "nope") == []


def test_list_videos_returns_empty_for_dir_with_no_videos(tmp_path):
    assert _list_videos(tmp_path) == []


def test_list_videos_finds_mp4_files(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    assert _list_videos(tmp_path) == [str(video)]


def test_list_videos_ignores_non_video_files(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "c.txt").write_text("fake")
    assert _list_videos(tmp_path) == [str(video)]


def test_list_videos_sorted_newest_first(tmp_path):
    old = tmp_path / "old.mp4"
    new = tmp_path / "new.mp4"
    old.write_bytes(b"fake")
    new.write_bytes(b"fake")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert _list_videos(tmp_path) == [str(new), str(old)]


def test_list_videos_ignores_subdirectories(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    (tmp_path / "subdir.mp4").mkdir()
    assert _list_videos(tmp_path) == [str(video)]
