import re

from .data import Lyric


def read_lrc(file_path) -> list[Lyric]:
    """
    读取 lrc 文件，返回一个包含 (时间, 歌词) 的列表
    时间格式：mm:ss.xx
    """
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    lyrics = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                minutes, seconds, text = match.groups()
                total_seconds = int(minutes) * 60 + float(seconds)
                lyrics.append(
                    {
                        "time": f"{int(minutes):02}:{float(seconds):05.2f}",
                        "seconds": total_seconds,
                        "lyric": text.strip(),
                    }
                )
    return lyrics
