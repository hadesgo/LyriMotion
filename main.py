from lyrimotion import LyriMotion


def main():
    lyri_motion = LyriMotion(
        "./temp/海鸥 - 逃跑计划.lrc",
        "./temp/海鸥 - 逃跑计划.mp3",
        "动漫",
        "./temp/output",
    )
    lyri_motion.run()


if __name__ == "__main__":
    main()
