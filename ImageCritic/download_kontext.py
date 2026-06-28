from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Kontext-dev",
    repo_type="model",
    token="hf_xxx_your_token",   
    local_dir="kontext",
    resume_download=True,
)
print("successfully download",local_dir)
