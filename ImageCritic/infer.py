import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.lora_helper import set_single_lora, set_multi_lora
from safetensors.torch import load_file
from src.detail_encoder import DetailEncoder
from src.kontext_custom_pipeline import FluxKontextPipelineWithPhotoEncoderAddTokens
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

# hf_hub_download(
#     repo_id="ziheng1234/ImageCritic",
#     filename="detail_encoder.safetensors",
#     local_dir="models"     # 下载到本地 models/ 目录
# )
# hf_hub_download(
#     repo_id="ziheng1234/ImageCritic",
#     filename="lora.safetensors",
#     local_dir="models"
# )

# from huggingface_hub import snapshot_download

# repo_id = "ziheng1234/kontext"
# local_dir = "./kontext"
# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     repo_type="model",
#     resume_download=True,    
#     max_workers=8    
# )

def load_image_safely(image_path, size):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print("file error: "+image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))

base_path = "./models"


detail_encoder_path = f"{base_path}/detail_encoder.safetensors"
kontext_lora_path = f"{base_path}/lora.safetensors"

image_path_A = "./test_imgs/product_2.png"
image_path_B = "./test_imgs/generated_2.png"
product_tag = ""
if product_tag == "":
    product_tag = "product"

print("CUDA 可用：", torch.cuda.is_available())
print("当前设备：", torch.cuda.current_device())
print("设备名称：", torch.cuda.get_device_name(0))

device = "cuda:0"

pipeline = FluxKontextPipelineWithPhotoEncoderAddTokens.from_pretrained("./kontext", torch_dtype=torch.bfloat16)
pipeline.to(device)
# transformer = FluxTransformer2DModel.from_pretrained(
#     "./kontext", 
#     subfolder="transformer",
#     torch_dtype=torch.bfloat16, 
# )
# transformer.to(device)

state_dict = load_file(detail_encoder_path)
detail_encoder = DetailEncoder().to(dtype=pipeline.transformer.dtype, device=device)
detail_encoder.to(device)
with torch.no_grad():
    for name, param in detail_encoder.named_parameters():
        if name in state_dict:
            added = state_dict[name].to(param.device)  
            param.add_(added)

pipeline.detail_encoder = detail_encoder
set_single_lora(pipeline.transformer, kontext_lora_path, lora_weights=[1])

cond_A_image = load_image(image_path_A)
cond_B_image = load_image(image_path_B)

def pick_kontext_resolution(w: int, h: int) -> tuple[int, int]:
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568),(688, 1504),(720, 1456),(752, 1392),
        (800, 1328),(832, 1248),(880, 1184),(944, 1104),
        (1024, 1024),(1104, 944),(1184, 880),(1248, 832),
        (1328, 800),(1392, 752),(1456, 720),(1504, 688),(1568, 672),
    ]
    # 计算目标宽高比
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio)
    )

orig_w, orig_h = cond_B_image.size
target_w, target_h = pick_kontext_resolution(orig_w, orig_h)
width_for_model, height_for_model = target_w, target_h
cond_A_image = cond_A_image.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)
cond_B_image = cond_B_image.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)

size = cond_B_image.size
prompt = f"use the {product_tag} in IMG1 as a reference to refine, replace, enhance the {product_tag} in IMG2"
print("prompt1:", prompt)

# 使用 pipeline 生成图像
image = pipeline(
    image_A=cond_A_image,
    image_B=cond_B_image,
    prompt=prompt,
    height=size[1],
    width=size[0],
    guidance_scale=3.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

display_width = size[0] 
display_height = size[1]

image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
# 保存生成的图像
image.save(f"result.png")
concatenated_image = Image.new('RGB', (cond_A_image.width * 3, cond_A_image.height))
concatenated_image.paste(cond_A_image, (0, 0))  # 放置原始图像A
concatenated_image.paste(cond_B_image, (cond_A_image.width, 0))  # 放置原始图像B
concatenated_image.paste(image, (cond_A_image.width * 2, 0))  # 放置生成图像
concatenated_image.save(f"concatenated.png")

