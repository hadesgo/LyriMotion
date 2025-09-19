from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, gc


class QwenLLM:
    def __init__(self):
        model_path = "./checkpoints/Qwen3-4B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(
        self, system_prompt: str, prompt: str, max_new_tokens: int = 16384
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
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
