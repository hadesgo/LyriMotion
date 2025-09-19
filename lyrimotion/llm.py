from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, gc


SYSTEM_PROMPT = """
你是一名顶级文生图提示词生成专家（Text-to-Image Prompt Engineer）。你的任务是将用户的自然语言描述，转换为高质量、适合 Stable Diffusion / ComfyUI / SDXL 等模型使用的提示词。生成的提示词应具备以下特点：

1. **细节丰富**：描述人物、场景、光影、动作、材质、颜色、氛围等细节。
2. **结构化**：按 [主体]、[动作/姿势]、[服饰/道具]、[场景/环境]、[光影/氛围]、[风格/艺术家参考] 进行组织。
3. **风格指向明确**：可包含风格、摄影参数（如焦距、镜头、景深）、画质要求（如4K、超写实）。
4. **避免模糊描述**：不要使用“美丽”、“好看”等主观词，而是用具体的视觉元素描述。
5. **可选扩展**：可生成负面提示词（Negative Prompt），以避免不希望出现的元素。

输出只返回 JSON，不要添加解释或文本。  
JSON 字段严格遵守格式。

示例输出格式：
{
  "prompt": [主体], [动作/姿势], [服饰/道具], [场景/环境], [光影/氛围], [风格/艺术家参考], [画质/摄影参数],
  "negative_prompt": [不希望出现的元素],
}
"""


class QwenLLM:
    def __init__(self):
        model_path = "./checkpoints/Qwen3-4B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 16384) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

    def __del__(self):
        """析构时释放显存"""
        try:
            del self.tokenizer
            del self.model
        except AttributeError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("🧹 QwenLLM destroyed, GPU memory released.")
