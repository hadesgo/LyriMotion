import json

from lyrimotion import QwenLLM, ImageGenerator


def main():
    llm = QwenLLM()
    text = llm.generate(
        "穿着黑丝的美女，站在酒吧门口前",
    )
    del llm
    json_text = json.loads(text)
    print("prompt:", json_text["prompt"])
    print("negative_prompt:", json_text["negative_prompt"])
    image_generator = ImageGenerator()
    image_generator.generate(
        prompt=json_text["prompt"],
        negative_prompt=json_text["negative_prompt"],
        width=1664,
        height=928,
        output_path="output.png",
    )


if __name__ == "__main__":
    main()
