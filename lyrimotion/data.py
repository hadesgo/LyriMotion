from dataclasses import dataclass
from typing import Optional


@dataclass
class Lyric:
    time: str
    seconds: float
    lyric: str


@dataclass
class LyricPrompt:
    time: str
    lyric: str
    prompt: str
    camera: str
    seconds: Optional[float] = 0
