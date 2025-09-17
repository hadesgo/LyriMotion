# LyriMotion

**自动化生成歌词MV的Python库，结合音乐节拍、情感分析与文生图，实现歌词驱动的动态视频创作。**

---

## 功能特点

- 🎵 **音乐节拍分析**：自动检测歌曲节奏，为视频剪辑提供参考。  
- 😍 **情感驱动视觉**：根据歌曲情感和歌词生成视觉提示。  
- 🎨 **文生图支持**：可接入任意文生图模型生成高质量画面。  
- 🎬 **动态视频生成**：使用 MoviePy 自动拼接图像、添加转场和文字，生成完整MV。  
- 🔧 **高度可扩展**：模块化设计，可自由替换情感模型或图像生成接口。  

---

## 安装

```bash
git clone https://github.com/yourusername/LyriMotion.git
cd LyriMotion
pip install -r requirements.txt
```

---

## 快速使用

```python
from auto_lyric_mv import LyricMVGenerator

lyrics = [
    "清晨的阳光洒在窗前",
    "风轻轻吹过心田",
    "我们一起唱着歌"
]

generator = LyricMVGenerator(
    audio_path="my_song.mp3",
    lyrics=lyrics,
    img_api=None,  # 可替换为图生图接口
    img_key=None
)

generator.generate("my_song_mv.mp4")
```
运行后会生成对应的歌词MV文件 ```my_song_mv.mp4```，自动按照歌曲节拍和情感生成画面和转场。

---

## 贡献

欢迎提交 issue 或 pull request，一起完善 LyriMotion！

