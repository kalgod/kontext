import argparse
import os
import torch
import numpy as np
from PIL import Image
# from diffusers import (
#     AutoencoderKL,
#     FlowMatchEulerDiscreteScheduler,
#     FluxTransformer2DModel
# )
from src.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.lora_helper import set_single_lora, set_multi_lora
from safetensors.torch import load_file
from src.detail_encoder import DetailEncoder
from src.kontext_custom_pipeline import FluxKontextPipelineWithPhotoEncoderAddTokens
from src.regione_adapter import enable_regione, RegionEArgs
from src.teacache_adapter import enable_teacache, TeaCacheArgs, teacache_summary
from src.sage_adapter import enable_sageattn, SageArgs
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
        print("file error: " + image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))


def pick_kontext_resolution(w: int, h: int) -> tuple[int, int]:
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568), (688, 1504), (720, 1456), (752, 1392),
        (800, 1328), (832, 1248), (880, 1184), (944, 1104),
        (1024, 1024), (1104, 944), (1184, 880), (1248, 832),
        (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672),
    ]
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio),
    )


def compute_psnr(a_path: str, b_path: str) -> float:
    a = np.array(Image.open(a_path).convert("RGB"), dtype=np.float64)
    b_img = Image.open(b_path).convert("RGB")
    if b_img.size != (a.shape[1], a.shape[0]):
        b_img = b_img.resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS)
    b = np.array(b_img, dtype=np.float64)
    mse = ((a - b) ** 2).mean()
    if mse <= 1e-10:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_regione", action="store_true",
                        help="Enable RegionE sparse-Q acceleration on the middle steps")
    parser.add_argument("--use_teacache", action="store_true",
                        help="Enable TeaCache (timestep-level skip with residual reuse). "
                             "Composes with --use_regione: in that case TeaCache only "
                             "fires during RegionE FULL steps that are not warmup-1 / refresh / post.")
    parser.add_argument("--rel_l1_thresh", type=float, default=0.4,
                        help="TeaCache rel_l1_thresh. 0.25=1.5x, 0.4=1.8x, 0.6=2.0x, 0.8=2.25x.")
    parser.add_argument("--use_sageattn", action="store_true",
                        help="Replace SDPA with SageAttention (INT8 quantized) in LoRA processors.")
    parser.add_argument("--sage_scope", choices=["full", "all"], default="all",
                        help="full: only LoRA processors (full/refresh/post/warmup steps). "
                             "all : LoRA processors + RegionE sparse path. "
                             "Only relevant when --use_sageattn is set.")
    parser.add_argument("--sage_kernel",
                        choices=["auto", "qk_int8_pv_fp16", "qk_int8_pv_fp8", "qk_int8_pv_fp16_triton"],
                        default="auto",
                        help="Which sageattn kernel variant to call.")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--warmup_step", type=int, default=6)
    parser.add_argument("--post_step", type=int, default=2)
    parser.add_argument("--refresh_step", type=str, default="16")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--erosion_dilation", action="store_true", default=True)
    parser.add_argument("--no_erosion_dilation", dest="erosion_dilation", action="store_false")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_path_A", type=str, default="./test_imgs/product_2.png")
    parser.add_argument("--image_path_B", type=str, default="./test_imgs/generated_2.png")
    parser.add_argument("--product_tag", type=str, default="")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename. Defaults to result.png (baseline) or result_regione.png (RegionE).")
    parser.add_argument("--baseline_for_psnr", type=str, default="result.png",
                        help="When --use_regione, compute PSNR against this baseline image.")
    parser.add_argument("--debug", action="store_true", help="Print per-step RegionE state for debugging.")
    parser.add_argument("--fixed_noise", type=str, default="./fixed_noise_step0000.pt",
                        help="Path to a fixed initial noise tensor (unpacked, VAE space shape [B,C,Hl,Wl]). "
                             "If the file exists it is loaded and replaces the random init; "
                             "set to '' to disable and use the random generator instead.")
    args = parser.parse_args()

    base_path = "./models"
    detail_encoder_path = f"{base_path}/detail_encoder.safetensors"
    kontext_lora_path = f"{base_path}/lora.safetensors"

    image_path_A = args.image_path_A
    image_path_B = args.image_path_B
    product_tag = args.product_tag if args.product_tag else "product"

    print("CUDA 可用：", torch.cuda.is_available())
    print("当前设备：", torch.cuda.current_device())
    print("设备名称：", torch.cuda.get_device_name(0))
    device = "cuda:0"

    transformer = FluxTransformer2DModel.from_pretrained(
        "./kontext",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipeline = FluxKontextPipelineWithPhotoEncoderAddTokens.from_pretrained(
        "./kontext",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to(device)

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

    # SageAttention is a class-level patch on the LoRA processor classes,
    # so install it right after LoRA processors are registered.  Order vs
    # TeaCache / RegionE is flexible because sage's patch surface (LoRA
    # processor.__call__) is independent of theirs (transformer.forward
    # and Attention.processor).
    if args.use_sageattn:
        enable_sageattn(
            pipeline,
            SageArgs(
                enable=True,
                scope=args.sage_scope,
                kernel=args.sage_kernel,
                debug=args.debug,
            ),
        )

    # IMPORTANT: TeaCache patches transformer.forward.  RegionE wraps whatever
    # forward is currently bound on the transformer at enable time.  So the
    # order MUST be: lora -> teacache -> regione.  Reversing this would
    # bypass TeaCache entirely.
    if args.use_teacache:
        enable_teacache(
            pipeline,
            TeaCacheArgs(
                enable=True,
                num_inference_steps=args.num_inference_steps,
                rel_l1_thresh=args.rel_l1_thresh,
                debug=args.debug,
            ),
        )

    if args.use_regione:
        enable_regione(
            pipeline,
            RegionEArgs(
                enable=True,
                num_inference_steps=args.num_inference_steps,
                warmup_step=args.warmup_step,
                post_step=args.post_step,
                refresh_step=args.refresh_step,
                threshold=args.threshold,
                erosion_dilation=args.erosion_dilation,
                debug=args.debug,
            ),
        )
        print(f"[RegionE] enabled — warmup={args.warmup_step}, post={args.post_step}, "
              f"refresh={args.refresh_step}, threshold={args.threshold}")

    cond_A_image = load_image(image_path_A)
    cond_B_image = load_image(image_path_B)

    orig_w, orig_h = cond_B_image.size
    target_w, target_h = pick_kontext_resolution(orig_w, orig_h)
    width_for_model, height_for_model = target_w, target_h
    cond_A_image = cond_A_image.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)
    cond_B_image = cond_B_image.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)

    size = cond_B_image.size
    prompt = f"use the {product_tag} in IMG1 as a reference to refine, replace, enhance the {product_tag} in IMG2"
    print("prompt1:", prompt)

    # ----------------------------------------------------------------------
    # Optional: load a fixed initial noise saved with shape
    #   [batch, vae_latent_channels, Hl, Wl]    (VAE space, NOT packed)
    # The pipeline's prepare_latents() will skip its random init when
    # `latents=` is provided, but it ALSO skips pack_latents in that branch
    # (see kontext_custom_pipeline.py:1660), so we have to pack it ourselves.
    # ----------------------------------------------------------------------
    init_latents = None
    if args.fixed_noise and os.path.isfile(args.fixed_noise):
        fixed = torch.load(args.fixed_noise, weights_only=True)
        print(f"[fixed_noise] loaded {args.fixed_noise} shape={tuple(fixed.shape)} "
              f"mean={fixed.float().mean().item():.6f} std={fixed.float().std().item():.6f}")
        # Pack [B, C, Hl, Wl] -> [B, Hl/2 * Wl/2, C*4] using the same routine
        # the pipeline uses internally for its own random latents.
        nc = pipeline.transformer.config.in_channels // 4
        vae_sf = pipeline.vae_scale_factor
        # height/width (image space) -> latent space (VAE 8x), rounded for the 2x2 patching
        lh = 2 * (size[1] // (vae_sf * 2))
        lw = 2 * (size[0] // (vae_sf * 2))
        assert fixed.shape[-2:] == (lh, lw), (
            f"fixed noise shape {tuple(fixed.shape)} does not match expected "
            f"latent (Hl,Wl)=({lh},{lw}). Regenerate the .pt at the current resolution."
        )
        fixed = fixed.to(device=device, dtype=pipeline.transformer.dtype)
        init_latents = pipeline._pack_latents(fixed, fixed.shape[0], nc, lh, lw)
        print(f"[fixed_noise] packed -> {tuple(init_latents.shape)}; "
              f"random `--seed {args.seed}` is ignored")
    elif args.fixed_noise:
        print(f"[fixed_noise] {args.fixed_noise} not found, falling back to random seed={args.seed}")

    # ---- profile pipeline call (CUDA-event based, async-safe) ----
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()

    image = pipeline(
        image_A=cond_A_image,
        image_B=cond_B_image,
        prompt=prompt,
        height=size[1],
        width=size[0],
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        latents=init_latents,
    ).images[0]

    ev_end.record()
    torch.cuda.synchronize()
    pipeline_ms = ev_start.elapsed_time(ev_end)
    # Build a tag describing the active acceleration combo
    tag_parts = []
    if args.use_regione:
        tag_parts.append("RegionE")
    if args.use_teacache:
        tag_parts.append(f"TeaCache(t={args.rel_l1_thresh})")
    if args.use_sageattn:
        tag_parts.append(f"Sage({args.sage_scope})")
    tag = "+".join(tag_parts) if tag_parts else "baseline"
    print(f"[time] pipeline ({tag}) = {pipeline_ms / 1000:.3f} s "
          f"({pipeline_ms / args.num_inference_steps:.1f} ms/step over {args.num_inference_steps} steps)")
    if args.use_teacache:
        print(teacache_summary(pipeline))

    display_width = size[0]
    display_height = size[1]
    image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)

    # ----------------------------------------------------------------------
    # Save under ./outputs/ with a descriptive filename:
    #   baseline:         outputs/baseline_<gpu>_<sec>s.png
    #   with accel:       outputs/<combo>_<gpu>_<sec>s_psnr<XX.XX>dB.png
    # The baseline run also drops a small meta file so future runs can find
    # it for PSNR / speedup comparison without relying on path conventions.
    # ----------------------------------------------------------------------
    OUTPUT_DIR = "./outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sanitize GPU name into a short slug ("h200", "a100", "l40", ...).
    gpu_full = torch.cuda.get_device_name(0)
    gpu_slug = gpu_full.replace("NVIDIA", "").replace(" ", "").lower()
    for tok in ("h200", "h100", "a100", "a800", "l40", "l4", "v100", "rtx4090", "rtx3090"):
        if tok in gpu_slug:
            gpu_slug = tok
            break

    # Combo slug (used in filename instead of the pretty parens-y tag)
    combo_parts = []
    if args.use_regione:
        combo_parts.append("regione")
    if args.use_teacache:
        combo_parts.append(f"teacache{args.rel_l1_thresh}")
    if args.use_sageattn:
        combo_parts.append(f"sage-{args.sage_scope}")
    combo = "+".join(combo_parts) if combo_parts else "baseline"

    # ----------------------------------------------------------------------
    # Core args that always change the output
    #   steps<N>     = --num_inference_steps
    #   gs<G>        = --guidance_scale
    #   <noise>      = "noise<basename>" if fixed_noise loaded, else "seed<N>"
    # ----------------------------------------------------------------------
    core_parts = [
        f"steps{args.num_inference_steps}",
        f"gs{args.guidance_scale}",
    ]
    if init_latents is not None:
        # use the .pt file basename so different fixed-noise files are distinguishable
        noise_tag = os.path.splitext(os.path.basename(args.fixed_noise))[0]
        # strip non-alnum to keep filename safe
        noise_tag = "".join(ch for ch in noise_tag if ch.isalnum() or ch in "-.")
        core_parts.append(f"noise{noise_tag}")
    else:
        core_parts.append(f"seed{args.seed}")

    # ----------------------------------------------------------------------
    # Per-accelerator args that affect the result (only included when the
    # corresponding accelerator is enabled)
    # ----------------------------------------------------------------------
    accel_parts = []
    if args.use_regione:
        # warmup, post, refresh, threshold, erosion_dilation
        ed_flag = 1 if args.erosion_dilation else 0
        # refresh_step may be "16" or "11,16,21" — make it filename-safe
        refresh_safe = args.refresh_step.replace(",", "-")
        accel_parts += [
            f"w{args.warmup_step}",
            f"p{args.post_step}",
            f"r{refresh_safe}",
            f"th{args.threshold}",
            f"ed{ed_flag}",
        ]
    # TeaCache: rel_l1_thresh already encoded in combo, nothing else affects result
    if args.use_sageattn:
        # sage_kernel: different kernel can give slightly different numerical result
        accel_parts.append(f"k{args.sage_kernel}")

    sec_str = f"{pipeline_ms / 1000:.2f}s"

    # Compose the base filename:
    #   <combo>__<core_args>__<accel_args>__<gpu>_<sec>[__psnrXXdB].png
    # core_args: always present (steps / gs / seed-or-noise)
    # accel_args: only when an accelerator is on
    core_str = "_".join(core_parts)
    accel_str = ("__" + "_".join(accel_parts)) if accel_parts else ""

    # Baseline meta file: records the canonical baseline png path + ms.
    BASELINE_META = os.path.join(OUTPUT_DIR, ".baseline_meta.txt")

    if combo == "baseline":
        # Baseline filename: no PSNR (it IS the reference)
        out_basename = f"{combo}__{core_str}{accel_str}__{gpu_slug}_{sec_str}.png"
        out_name = os.path.join(OUTPUT_DIR, out_basename)
        image.save(out_name)
        # Record this file as the baseline for future PSNR / speedup runs.
        with open(BASELINE_META, "w") as f:
            f.write(f"{out_name}\n{pipeline_ms:.3f}\n")
        psnr_dB = None
    else:
        # Acceleration runs: compute PSNR vs baseline image (if available)
        baseline_img_path = None
        baseline_ms = None
        # Prefer the meta written by the baseline run; otherwise fall back
        # to args.baseline_for_psnr if it exists.
        if os.path.isfile(BASELINE_META):
            with open(BASELINE_META) as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) >= 1 and os.path.isfile(lines[0]):
                baseline_img_path = lines[0]
            if len(lines) >= 2:
                try:
                    baseline_ms = float(lines[1])
                except ValueError:
                    pass
        if baseline_img_path is None and os.path.isfile(args.baseline_for_psnr):
            baseline_img_path = args.baseline_for_psnr

        psnr_dB = None
        if baseline_img_path is not None:
            # Save to a temp PIL buffer first to compute PSNR vs the saved
            # baseline png.  We compute PSNR on the in-memory PIL image to
            # avoid a write-then-read roundtrip.
            import numpy as _np
            ref = _np.array(Image.open(baseline_img_path).convert("RGB"), dtype=_np.float64)
            cur = _np.array(image.convert("RGB"), dtype=_np.float64)
            if ref.shape != cur.shape:
                cur_pil = image.convert("RGB").resize(
                    (ref.shape[1], ref.shape[0]), Image.Resampling.LANCZOS
                )
                cur = _np.array(cur_pil, dtype=_np.float64)
            mse = ((ref - cur) ** 2).mean()
            psnr_dB = float("inf") if mse <= 1e-10 else float(
                20.0 * _np.log10(255.0) - 10.0 * _np.log10(mse)
            )

        psnr_str = "psnrINF" if psnr_dB == float("inf") else (
            f"psnr{psnr_dB:.2f}dB" if psnr_dB is not None else "psnrNA"
        )
        out_basename = f"{combo}__{core_str}{accel_str}__{gpu_slug}_{sec_str}__{psnr_str}.png"
        out_name = os.path.join(OUTPUT_DIR, out_basename)
        image.save(out_name)

    # Concatenated triptych alongside, same naming convention with prefix
    concatenated_image = Image.new('RGB', (cond_A_image.width * 3, cond_A_image.height))
    concatenated_image.paste(cond_A_image, (0, 0))
    concatenated_image.paste(cond_B_image, (cond_A_image.width, 0))
    concatenated_image.paste(image, (cond_A_image.width * 2, 0))
    concat_name = os.path.join(OUTPUT_DIR, f"concat_{out_basename}")
    concatenated_image.save(concat_name)
    print(f"[saved] {out_name}, {concat_name}")

    # Speedup vs baseline (only when current run is an accel run AND a
    # baseline meta record exists)
    if combo != "baseline":
        if os.path.isfile(BASELINE_META):
            with open(BASELINE_META) as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) >= 2:
                try:
                    bl_ms = float(lines[1])
                    print(f"[time] baseline = {bl_ms / 1000:.3f} s, "
                          f"{tag} = {pipeline_ms / 1000:.3f} s, "
                          f"speedup = {bl_ms / pipeline_ms:.2f}x "
                          f"(saved {(bl_ms - pipeline_ms) / 1000:.3f} s)")
                except ValueError:
                    pass
        else:
            print("[time] no baseline meta found; "
                  "run `python infer.py` (no flags) once to record baseline.")

        if psnr_dB is not None:
            print(f"[PSNR] vs baseline = {psnr_dB:.4f} dB")
        else:
            print("[PSNR] baseline image not found; cannot compute PSNR.")


if __name__ == "__main__":
    main()
