from unittest.mock import MagicMock, patch

from img2vid.studio_ui import build_demo


def test_build_demo_composes_video_and_image_tabs():
    fake_video_demo = MagicMock(name="video_demo")
    fake_image_demo = MagicMock(name="image_demo")
    with (
        patch("img2vid.studio_ui.build_video_demo", return_value=fake_video_demo) as mock_video,
        patch("img2vid.studio_ui.build_image_demo", return_value=fake_image_demo) as mock_image,
        patch("img2vid.studio_ui.gr.TabbedInterface") as mock_tabbed,
    ):
        build_demo()

    mock_video.assert_called_once()
    mock_image.assert_called_once()
    args, kwargs = mock_tabbed.call_args
    assert args[0] == [fake_video_demo, fake_image_demo]
    assert len(kwargs["tab_names"]) == 2


def test_build_demo_returns_real_tabbed_interface():
    import gradio as gr

    demo = build_demo()
    assert isinstance(demo, gr.Blocks)
