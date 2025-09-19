import math
import torch
import gc

from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipeline,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
from nunchaku.utils import get_precision


class ImageGenerator:
    def __init__(self):
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        model_name = (
            f"svdq-{get_precision()}_r128-qwen-image-lightningv1.1-8steps.safetensors"
        )
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            f"./checkpoints/{model_name}"
        )
        self.pipe = QwenImagePipeline.from_pretrained(
            "./checkpoints/Qwen-Image",
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
            scheduler=scheduler,
        )
        # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
        self.transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=50)
        # increase num_blocks_on_gpu if you have more VRAM
        self.pipe._exclude_from_cpu_offload.append("transformer")
        self.pipe.enable_sequential_cpu_offload()

    def generate(
        self,
        output_path: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
    ):
        img = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=8,
            true_cfg_scale=1.0,
        ).images[0]
        img.save(output_path)

    def __del__(self):
        """析构时释放显存"""
        try:
            del self.transformer
            del self.pipe
        except AttributeError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("🧹 ImageGenerator destroyed, GPU memory released.")
