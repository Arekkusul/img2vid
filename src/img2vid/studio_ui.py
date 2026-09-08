import gradio as gr

from img2vid.image_ui import build_demo as build_image_demo
from img2vid.ui import build_demo as build_video_demo


def build_demo() -> gr.Blocks:
    """Combine the standalone video (LTX-2.5) and image (Krea 2) UIs into one tabbed app.

    Reuses each app's own `build_demo()` as-is (including its own `.queue()` call) rather
    than reimplementing their `_run`/layout logic -- `gr.TabbedInterface` is Gradio's own
    built-in primitive for exactly this "one app, multiple functions" composition.
    """
    return gr.TabbedInterface(
        [build_video_demo(), build_image_demo()],
        tab_names=["Video (LTX-2.5)", "Image (Krea 2)"],
        title="img2vid studio",
    )


def launch():
    build_demo().launch()


if __name__ == "__main__":
    launch()
