from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ziheng1234/ImageCritic",
    filename="detail_encoder.safetensors",
    local_dir="models"     # 下载到本地 models/ 目录
)
hf_hub_download(
    repo_id="ziheng1234/ImageCritic",
    filename="lora.safetensors",
    local_dir="models"
)

