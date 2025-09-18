from lyrimotion import ImageGenerator


def main():
    image_gen = ImageGenerator()
    image_gen.generate(
        "Landscape, beautiful sky, rainbow in the sky", 1920, 1080, "./local.png"
    )


if __name__ == "__main__":
    main()
