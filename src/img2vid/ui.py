import gradio as gr

from img2vid.generate import GenerationError, generate_video


def _run(image, prompt, width, height, frames, steps, seed):
    if not image:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    try:
        output_path = generate_video(
            image,
            prompt,
            width=int(width),
            height=int(height),
            frames=int(frames),
            steps=int(steps),
            seed=None if seed is None else int(seed),
        )
    except GenerationError as exc:
        raise gr.Error(str(exc)) from exc

    return str(output_path)


def build_demo() -> gr.Interface:
    demo = gr.Interface(
        fn=_run,
        inputs=[
            gr.Image(type="filepath", label="Input image"),
            gr.Textbox(label="Prompt", placeholder="Describe the motion/scene you want"),
            gr.Slider(256, 1280, value=832, step=16, label="Width"),
            gr.Slider(256, 1280, value=480, step=16, label="Height"),
            gr.Slider(9, 121, value=81, step=4, label="Frames"),
            gr.Slider(4, 50, value=25, step=1, label="Steps"),
            gr.Number(label="Seed (optional)", precision=0),
        ],
        outputs=gr.Video(label="Generated video"),
        title="img2vid",
        description="Local image-to-video generation (Wan2.2 TI2V-5B via mlx-gen).",
    )
    demo.queue(max_size=1)
    return demo


def launch():
    build_demo().launch()


if __name__ == "__main__":
    launch()
