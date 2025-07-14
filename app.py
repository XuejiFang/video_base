import gradio as gr
import torch
from opensora.pipelines import OpenSoraPipeline
from opensora.models.causalvideovae import CausalVAEModelWrapper
from diffusers.utils import export_to_video
import tempfile
import os

device = 'cuda:0'
dtype = torch.float16
FRAMES = 29
HEIGHT = 480
WIDTH = 640

# 初始化模型
vae = CausalVAEModelWrapper('models/vae').to(device)
pipeline = OpenSoraPipeline.from_pretrained('./models', vae=vae).to(device, dtype)

# 默认 negative prompt
DEFAULT_NEG_PROMPT = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"""

def generate_video(prompt, negative_prompt, seed, step, cfg):
    generator = torch.Generator('cuda').manual_seed(int(seed))
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=step,
        guidance_scale=cfg,
        generator=generator
    ).frames[0]

    tmp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    export_to_video(result, tmp_path, fps=8)
    return tmp_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Text-to-Video Diffusion Model (Updated on 2025-02-10)")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEG_PROMPT, lines=4)

    with gr.Row():
        seed = gr.Number(label="Seed", value=20250210, precision=0)
        step = gr.Slider(label="Steps", minimum=5, maximum=50, value=30, step=1)
        cfg = gr.Slider(label="CFG", minimum=1.1, maximum=15.0, value=5.5, step=0.1)

    with gr.Row():
        gr.Markdown(f"**Frames**: {FRAMES} &nbsp;&nbsp;&nbsp; **Height**: {HEIGHT} &nbsp;&nbsp;&nbsp; **Width**: {WIDTH}")

    with gr.Row():
        run_btn = gr.Button("Generate Video")
        video_output = gr.Video(label="Generated Video")

    run_btn.click(fn=generate_video, inputs=[prompt, negative_prompt, seed, step, cfg], outputs=video_output)

if __name__ == "__main__":
    demo.launch()
