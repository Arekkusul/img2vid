import gradio as gr

from img2vid.generate import GenerationError, generate_video


def _run(image, prompt, width, height, frames, steps, cfg_scale, frame_rate, dev, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    try:
        output_path = generate_video(
            prompt,
            image_path=image,
            width=int(width),
            height=int(height),
            frames=int(frames),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            frame_rate=float(frame_rate),
            dev=bool(dev),
            seed=None if seed is None else int(seed),
        )
    except GenerationError as exc:
        raise gr.Error(str(exc)) from exc

    return str(output_path)


def build_demo() -> gr.Interface:
    demo = gr.Interface(
        fn=_run,
        inputs=[
            gr.Image(type="filepath", label="Input image (optional -- omit for text-to-video)"),
            gr.Textbox(label="Prompt", placeholder="Describe the desired scene/motion"),
            gr.Slider(256, 1280, value=704, step=32, label="Width"),
            gr.Slider(256, 1280, value=480, step=32, label="Height"),
            gr.Slider(9, 121, value=97, step=8, label="Frames"),
            gr.Slider(4, 50, value=30, step=1, label="Steps (dev mode only)"),
            gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="CFG scale (dev mode only)"),
            gr.Slider(8, 30, value=24, step=1, label="Frame rate"),
            gr.Checkbox(
                label="Use dev model (slower CFG path; leave off for the fast distilled default)",
                value=False,
            ),
            gr.Number(label="Seed (optional)", precision=0),
        ],
        outputs=gr.Video(label="Generated video"),
        title="img2vid",
        description="Local text-to-video and image-to-video generation (LTX-2.5 via a from-scratch MLX conversion).",
    )
    demo.queue(max_size=1)
    return demo


def launch():
    build_demo().launch()


if __name__ == "__main__":
    launch()
