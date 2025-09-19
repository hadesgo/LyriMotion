from modelscope import snapshot_download, model_file_download
from nunchaku.utils import get_precision


def download_checkpoints():
    # 下载qwen llm 模型
    snapshot_download(
        "Qwen/Qwen3-4B-Instruct-2507",
        local_dir="./checkpoints/Qwen3-4B-Instruct",
    )
    # 下载qwen imgae 模型
    snapshot_download(
        "Qwen/Qwen-Image",
        local_dir="./checkpoints/Qwen-Image",
        ignore_patterns=["transformer/*.safetensors"],
    )

    # 下载qwen imgae 量化模型
    model_name = (
        f"svdq-{get_precision()}_r128-qwen-image-lightningv1.1-8steps.safetensors"
    )
    model_file_download(
        "nunchaku-tech/nunchaku-qwen-image",
        file_path=model_name,
        local_dir="./checkpoints",
    )


if __name__ == "__main__":
    download_checkpoints()
