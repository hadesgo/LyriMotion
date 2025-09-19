import json
import os

from .utils import read_lrc
from .llm import QwenLLM
from .image_generator import ImageGenerator
from .constant import ANALYSIS_LYRICS, TEXT_TO_IMAGE_PROMPT
from .data import Lyric, LyricPrompt


class LyriMotion:
    def __init__(
        self, lrc_path: str, music_path: str, style: str, out_dir: str
    ) -> None:
        self.lrcs = read_lrc(lrc_path)
        self.music_path = music_path
        self.style = style
        self.out_dir = out_dir
        pass

    def analysis_lyrics(self, llm: QwenLLM, lyrics: list[Lyric]) -> list[LyricPrompt]:
        content = llm.generate(
            ANALYSIS_LYRICS,
            str(lyrics),
        )
        content = content.replace("```json", "").replace("```", "")
        lyrics_dicts = json.loads(content)
        lyrics_objects = [LyricPrompt(**d) for d in lyrics_dicts]
        return lyrics_objects

    def generate_image_prompt(
        self, llm: QwenLLM, lyrics: list[LyricPrompt]
    ) -> list[dict[str, str]]:
        image_prompts = []
        for lyric in lyrics:
            content = llm.generate(
                TEXT_TO_IMAGE_PROMPT,
                lyric.prompt + self.style,
            )
            content = content.replace("```json", "").replace("```", "")
            image_prompt = json.loads(content)
            image_prompts.append(image_prompt)
        return image_prompts

    def generate_images(
        self, image_generator: ImageGenerator, image_prompts: list[dict[str, str]]
    ):
        for index, image_prompt in enumerate(image_prompts):
            image_generator.generate(
                prompt=image_prompt["prompt"],
                negative_prompt=image_prompt["negative_prompt"],
                width=1024,
                height=1024,
                output_path=os.path.join(self.out_dir, f"{index}.png"),
            )
        pass

    def run(self):
        llm = QwenLLM()
        lyric_prompt_list = self.analysis_lyrics(llm, self.lrcs)
        image_prompt_list = self.generate_image_prompt(llm, lyric_prompt_list)
        del llm
        image_generator = ImageGenerator()
        self.generate_images(image_generator, image_prompt_list)
        pass

    def generate(self):
        pass
