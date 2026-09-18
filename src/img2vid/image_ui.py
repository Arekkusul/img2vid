import gradio as gr

from img2vid.image_generate import ImageGenerationError, generate_image


def _run(edit_source, prompt, width, height, steps, guidance, seed, negative_prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    try:
        output_path = generate_image(
            prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance=float(guidance),
            seed=None if seed is None else int(seed),
            negative_prompt=negative_prompt or None,
            edit_image_path=edit_source,
        )
    except ImageGenerationError as exc:
        raise gr.Error(str(exc)) from exc

    return str(output_path)


def build_demo() -> gr.Interface:
    demo = gr.Interface(
        fn=_run,
        inputs=[
            gr.Image(type="filepath", label="Reference image (optional — enables edit mode)"),
            gr.Textbox(label="Prompt", placeholder="What to generate, or how to edit the reference image"),
            gr.Slider(256, 2048, value=1024, step=16, label="Width"),
            gr.Slider(256, 2048, value=1024, step=16, label="Height"),
            gr.Slider(4, 60, value=52, step=1, label="Steps"),
            gr.Slider(0.0, 10.0, value=3.5, step=0.5, label="Guidance"),
            gr.Number(label="Seed (optional)", precision=0),
            gr.Textbox(label="Negative prompt (optional)"),
        ],
        outputs=gr.Image(label="Generated image"),
        title="img2vid — Krea 2 image generation",
        description="Local text-to-image generation via Krea 2. Upload a reference image to switch to edit mode.",
    )
    demo.queue(max_size=1)
    return demo


def launch():
    build_demo().launch()


if __name__ == "__main__":
    launch()
