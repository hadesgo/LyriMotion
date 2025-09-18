import torch, gc
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)


class ImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            "./checkpoints/waiNSFWIllustrious_v150.safetensors",
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,
        ).to(self.device)

    def generate(self, prompt: str, width: int, height: int, output_path: str):
        img = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=30,
        ).images[0]
        img.save(output_path)

    def __del__(self):
        """析构时释放显存"""
        try:
            del self.pipe
        except AttributeError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("🧹 ImageGenerator destroyed, GPU memory released.")
