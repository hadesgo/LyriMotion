import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_single_file(
        "./checkpoints/waiNSFWIllustrious_v150.safetensors",
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    img = pipe("cinematic portrait, ultra detailed", num_inference_steps=30).images[0]
    img.save("wai_local.png")


if __name__ == "__main__":
    main()
