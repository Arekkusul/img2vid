from pathlib import Path

import gradio as gr

DEFAULT_OUTPUTS_DIR = Path("outputs")
VIDEO_EXTENSIONS = (".mp4",)


def _list_videos(outputs_dir: Path) -> list[str]:
    """Video files directly under `outputs_dir`, newest first.

    Empty list if the directory doesn't exist yet (nothing generated there).
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []
    videos = [p for p in outputs_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in videos]


def build_demo(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> gr.Blocks:
    """Read-only browser for previously generated videos in `outputs_dir`."""

    def _refresh():
        videos = _list_videos(outputs_dir)
        choice = videos[0] if videos else None
        return gr.update(choices=videos, value=choice), choice

    with gr.Blocks(title="Library") as demo:
        gr.Markdown(f"Videos generated into `{outputs_dir}/`. Click Refresh after generating a new one.")
        with gr.Row():
            dropdown = gr.Dropdown(label="Video", choices=_list_videos(outputs_dir), scale=4)
            refresh_btn = gr.Button("Refresh", scale=0)
        player = gr.Video(label="Preview")
        dropdown.change(fn=lambda path: path, inputs=dropdown, outputs=player)
        refresh_btn.click(fn=_refresh, outputs=[dropdown, player])
        demo.load(fn=_refresh, outputs=[dropdown, player])
    return demo


def launch():
    build_demo().launch()


if __name__ == "__main__":
    launch()
