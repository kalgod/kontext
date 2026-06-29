# ImageCritic 推理流程 + RegionE 加速移植分析

> 假设默认 1024×1024 输入（实际 `pick_kontext_resolution` 会落到 Kontext 推荐分辨率，下文以 H=W=1024、`vae_scale_factor=8`、`patch_size=2`、`max_sequence_length=512`、`num_inference_steps=28` 为参数估算 shape）。

---

## 一、ImageCritic（`infer.py`）执行链路

### 1. 入口（`infer.py`）
1. 读两张图：`image_path_A`（参考图 product）、`image_path_B`（生成图 product）。
2. 构造 prompt：
   `"use the {tag} in IMG1 as a reference to refine, replace, enhance the {tag} in IMG2"`。
3. `FluxKontextPipelineWithPhotoEncoderAddTokens.from_pretrained("./kontext", torch_dtype=torch.bfloat16)` 加载完整 Flux-Kontext pipeline（`vae` / `text_encoder`(CLIP) / `text_encoder_2`(T5) / `transformer`(`FluxTransformer2DModel`) / `scheduler`(`FlowMatchEulerDiscreteScheduler`)）。
4. 加载 `detail_encoder.safetensors` → 注入到 `DetailEncoder`（CLIP-ViT-L/14 + 2 个 visual_projection + FuseModule MLP），把它挂到 `pipeline.detail_encoder`。
5. `set_single_lora(pipeline.transformer, lora.safetensors, lora_weights=[1])`：
   - 19 个 `transformer_blocks` 用 `MultiDoubleStreamBlockLoraProcessor` 替换默认 `FluxAttnProcessor2_0`，给 `to_q/to_k/to_v/proj` 各加一组 LoRA（`down 3072→r`、`up r→3072`，bf16）。
   - 38 个 `single_transformer_blocks` 用 `MultiSingleStreamBlockLoraProcessor`，对 q,k,v 加 LoRA（无 proj）。
6. `pick_kontext_resolution(orig_w, orig_h)` → 17 档 Kontext 推荐分辨率里选最接近的 (target_w, target_h)，把两张 condition resize 成同尺寸。
7. 调 `pipeline(image_A, image_B, prompt, height, width, guidance_scale=3.5, generator=…)`。

### 2. `FluxKontextPipelineWithPhotoEncoderAddTokens.__call__`

按代码分阶段并标注每一步的张量形状（以 1024×1024 为例）。

| 步骤 | 关键操作 | shape 变化 |
|---|---|---|
| 1. resize / multiple_of 对齐 | `width = width // 16 * 16`，`height = height // 16 * 16` | int |
| 2. `check_inputs` | 校验 prompt/embed 是否冲突等 | — |
| 3. 加 trigger token | `tokenizer_2.add_tokens(['IMG1','IMG2'], special_tokens=True)` 取得对应 `image_A_token_id / image_B_token_id` | — |
| 4. `encode_prompt` | `_get_clip_prompt_embeds`：CLIP pooled → `pooled_prompt_embeds` `[1, 768]`<br>`_get_t5_prompt_embeds`：T5 → `prompt_embeds` `[1, 512, 4096]`，并返回 `text_input_ids` `[1, 512]`<br>`text_ids` `[512, 3]` 全 0 | `[1,768]`、`[1,512,4096]`、`[512,3]`、`[1,512]` |
| 5. preprocess image | `image_processor.resize/preprocess` 把 A、B 各自变成 `[1, 3, H, W]` bf16 | `[1,3,1024,1024]`×2 |
| 6. **detail_encoder 注入** | 在 `text_input_ids` 中找 IMG1、IMG2 位置 → 生成 `class_tokens_A_mask / B_mask` `[1,512] bool`<br>`A_pixel_values, B_pixel_values` 各 `[1,1,3,1024,1024]`<br>`detail_encoder(A_pixel, prompt_embeds, A_mask)` 内部：reshape→双线性 resize 到 `[1,3,224,224]` → CLIP-ViT 取 pooled `[1,1024]` → `visual_projection`→`[1,768]`、`visual_projection_2`→`[1,1280]` → cat 成 `[1,1280+768=2048]` → 在 prompt_embeds 上 IMG1 位置以 FuseModule(MLP) 替换 → 输出 `[1,512,4096]`；同理 B。 | `prompt_embeds: [1,512,4096]` |
| 7. `prepare_latents` | 1) 把 A、B 各自 VAE encode（`AutoencoderKL` 8× 压缩）：每张得 `[1,16,128,128]`<br>2) 两张 image latents 各做 `_pack_latents`（2×2 patch）：`[1,16,128,128] → [1, 64×64=4096, 16×4=64]`<br>3) `image_latents = cat(A, B, dim=1)` → `[1, 8192, 64]`<br>4) `image_ids_A/B` 各 `[4096,3]`，`A[...,0]=1`、`B[...,0]=2`，cat → `[8192, 3]`<br>5) 噪声 latent `[1,16,128,128] → pack → [1, 4096, 64]`<br>6) `latent_ids` 噪声 `[4096, 3]` | `latents: [1,4096,64]`<br>`image_latents: [1,8192,64]`<br>`latent_ids: [12288,3]`（cat 后） |
| 8. timesteps / scheduler | `sigmas = linspace(1, 1/28, 28)`，`calculate_shift(image_seq_len=4096)` 得 `mu`，`scheduler.set_timesteps(28, sigmas, mu)` → `timesteps [28]` | — |
| 9. guidance | 因为 transformer 配置 `guidance_embeds=True`，`guidance = full(3.5, [1])` 扩到 `[batch]` | `[1]` |
| 10. **去噪循环（28 次）** | 见下表 | — |
| 11. unpack & VAE decode | `_unpack_latents([1,4096,64]) → [1,16,128,128]` → `vae.decode` → `[1,3,1024,1024]` → PIL | — |

### 3. 单步去噪 (`for i, t in enumerate(timesteps)`，i = 0..27)

```
latent_model_input = cat([latents, image_latents], dim=1)  # [1, 4096+8192=12288, 64]
timestep = t.expand([1])                                    # [1]

noise_pred = self.transformer(
    hidden_states          = latent_model_input,            # [1,12288,64]
    encoder_hidden_states  = prompt_embeds,                 # [1, 512, 4096]
    pooled_projections     = pooled_prompt_embeds,          # [1, 768]
    timestep, guidance, txt_ids[512,3], img_ids[12288,3]
)[0]
noise_pred = noise_pred[:, : latents.size(1)]               # 截前 4096，丢弃 condition
latents = scheduler.step(noise_pred, t, latents)            # [1,4096,64]
```

### 4. `FluxTransformer2DModel.forward` 内部（`src/transformer_flux.py`）

| 子步 | 作用 | shape |
|---|---|---|
| `x_embedder(hidden_states)` | Linear 64→3072 | `[1, 12288, 3072]` |
| `time_text_embed(timestep, guidance, pooled_proj)` | 得到 `temb` | `[1, 3072]` |
| `context_embedder(prompt_embeds)` | Linear 4096→3072 | `[1, 512, 3072]` |
| `pos_embed(cat(txt_ids, img_ids))` | RoPE：3 个轴 (16,56,56) → 复数 cos/sin | `(cos[12800,128], sin[12800,128])` |
| **19 × `FluxTransformerBlock` (double)** | text + image 联合 attention（self-attn 形式，QKV 各分 text/image 两路再 cat）；带 LoRA q/k/v/proj（`MultiDoubleStreamBlockLoraProcessor`） | seq_len=512+12288=12800，`q,k,v: [1, 24, 12800, 128]` |
| `cat([encoder_hidden_states, hidden_states], dim=1)` | 准备进 single block | `[1, 12800, 3072]` |
| **38 × `FluxSingleTransformerBlock` (single)** | text+image 同一序列，proj_mlp 把 hidden 投到 12288 与 attn output cat 后再投回 3072；带 LoRA q/k/v（`MultiSingleStreamBlockLoraProcessor`） | `[1, 12800, 3072]` |
| 切回 `hidden_states[:, 512:, :]` | 丢 text | `[1, 12288, 3072]` |
| `norm_out`+`proj_out` | 3072→64（patch_size²·out_channels） | `[1, 12288, 64]` |

外层再切 `[:, :4096, :]` 当作 noise_pred。

### 5. 开销重点（瓶颈拆解）

按 **每一步推理** 的 wall-clock 排序（1024² 配置）：

1. **Self-attention（19 double + 38 single = 57 层 × 28 step ≈ 1596 次）**
   - **double block**：`q,k,v shape [1, 24, 12800, 128]`，`Q·Kᵀ` 是 `O(B·H·N²·d)` ≈ `1·24·12800²·128 ≈ 5×10¹¹` FMA / 层。这是绝对的开销大头。
   - **single block** 同尺度（38 层），合计 attention FLOPs 约是 double 的 2 倍。
2. **proj_mlp / proj_out / FFN**（FeedForward in double, single 的 proj_mlp 12288 hidden）：每层每 token 约 `3072·12288·2≈7.5×10⁷` FLOPs，乘 token 数和层数同样不小。
3. **LoRA 三件套（q/k/v + proj）×（19+38）层 × 28 step**：每层每 token 多 `2·rank·dim` FMA（rank 通常 16~64）。属次级瓶颈。
4. **VAE 编码 / 解码**：encode 两张 condition + 最后 decode 一次。512×512→128×128 的下采样卷积，单次成本大约在 2~3 GFlops，整体相对小。
5. **DetailEncoder（CLIP-ViT-L/14 224×224）**：只跑两次（A、B 各一次），是常数项。
6. **T5-XXL / CLIP text encode**：只跑一次，约 ~10⁹–10¹⁰ FLOPs，相对总体可忽略。

> **关键瓶颈**：拼接后序列长度 `N = 512 (text) + 4096 (noise) + 8192 (cond_A+cond_B) = 12800`，attention 是 N² 复杂度，因此 condition 部分（A+B 共 8192 个 token）几乎吃掉了 2/3 长度，attention 计算量被 condition 撑大了 6.25×（相比单图 N=4608 来对比）。**这就是 RegionE 想加速的点**：很多 step 里 condition 和 1-mask 区域的 hidden state 没有变化，但仍被反复 attend。

---

## 二、RegionE（`src/FluxKontext/`）的加速思路

### 1. 阶段划分（28 步默认配置）

`script/FluxKontext.sh`：`warmup_step=6, post_step=2, refresh_step="16", threshold=0.93, cache_threshold=0.01, erosion_dilation=True`。

| step (0-index) | 类型 | 行为 |
|---|---|---|
| 0,1,2,3,4,5 | **warmup**（前 6 步） | 跑全长 transformer（latent + image_latents 全 cat），稳定噪声分布 |
| 5 (== `warmup_step-1`) | **mask 划分点** | 用当前 step 估计的 `x0 = sample + dt_final · model_output` 与 `image_latents`（这里就是 condition_B）做 cosine 相似度 → 选 `edited_ids`（mask 部分）和 `unedited_ids`（1-mask 部分）；同时把这一步的 K/V 写进 `k_cache/v_cache`（后续 mask 外的 token 复用） |
| 6,7,…,15 | **region-aware** | 只跑 mask 部分；K/V cache 复用，1-mask 与 condition **不重算** |
| 16 (== `refresh_step`) | **refresh** | 重新跑全长，刷新 `k_cache/v_cache` 和 `unedited_latent` |
| 17,…,25 | **region-aware** | 同 6–15 |
| 26,27 (倒数 `post_step=2` 步) | **post / smooth** | 重新跑全长，让 mask 边界平滑 |

> 因此**用户问的 "step 7–27 没算 1-mask 和 condition"**，更精确的说法是：
> - 只在 step 6..15 与 17..25 这些 region-aware step 里，1-mask 和 condition 部分**确实不再进 transformer 重新计算**；
> - step 16 (refresh) 和 step 26,27 (post) 仍然把 `latents + image_latents` 全长 cat 进去重新算。
> - step 0..5 (warmup) 也是全长。

### 2. region-aware step 里发生了什么

#### (a) Pipeline 层 (`RegionEFluxKontextPipeline.__call__`)

```python
if not should_cache:
    latent_model_input = latents          # 注意：默认就是 mask 部分 latents（已被上一次 MANAGER.step 用 ids_gather 选出）
    if (current_step <= warmup-1) or (current_step > infer-post-1) or (current_step == prev_refresh):
        latent_model_input = cat([latents, image_latents], dim=1)   # 只在全长 step 才 cat condition
```

- 在 region-aware step（6..15、17..25），`latent_model_input` 只是 **edited 部分的 latent**——长度从 4096 减到约 `K = mask 区域 token 数`（假设 K≈2000，仅举例），condition 不再加进去。
- `MANAGER.step()` 在 `warmup_step` 那一步执行：`unedited_latent = ids_gather(latent, unedited_ids)` 把 1-mask 区域的 latent 缓存到 `MANAGER.unedited_latent`；`latent = ids_gather(latent, edited_ids)`、`latent_ids = ids_gather(latent_ids, edited_ids)`，序列长度从 12288 缩到 K。

#### (b) Transformer forward (`RegionEFluxTransformer2DModelforward`)

- `MANAGER.image_rotary_emb = pos_embed(cat(txt_ids, MANAGER.latent_ids))` 这是**全长的 RoPE**（text + edited + unedited + condition），attention 时给 K 用。
- `image_rotary_emb = pos_embed(cat(txt_ids, img_ids))` 这是当前 forward 的 RoPE：text + 仅 edited（因为 img_ids 已被 gather）。给 Q 用。

#### (c) Attention processor (`RegoionEFluxAttnProcessor2_0.__call__`)

这是核心：

```python
query = attn.to_q(hidden_states)        # 只对 edited 部分(+text in single block) 做 Q 投影 —— 短序列

if region_aware step:
    if single:
        selection = cat([0..txt_length], edited_ids + txt_length)
    else:
        selection = edited_ids

    _partially_linear(hidden_states, attn.to_k.weight, attn.to_k.bias, selection, k_cache)
    _partially_linear(hidden_states, attn.to_v.weight, attn.to_v.bias, selection, v_cache)

    key   = self.k_cache    # 全长 K：[1, N_full, C_out]
    value = self.v_cache    # 全长 V

# RoPE 用两套
query = apply_rotary_emb(query, image_rotary_emb)              # 短 RoPE（text + edited）
key   = apply_rotary_emb(key,   MANAGER.image_rotary_emb)      # 全长 RoPE
hidden_states = flash_attn_func(q, k, v)                       # Q[K] × K[N_full] × V[N_full]
```

- `_partially_linear` 是一个 **triton kernel**，传入完整 weight + bias、`selection` 索引（要写入哪些 token 的位置）、以及目标 `k_cache / v_cache`。kernel 只对 selection 指定的位置重新做线性投影并原地写入 cache；其余 (1-mask 与 condition) 位置保留**上一次 refresh / warmup 末尾算好的旧 K、V**。
  - **double block** 的 Attn 输入 `hidden_states` 不含 text（text 走 `add_q/k/v_proj`），所以 selection 直接用 `edited_ids`（cache 里的 condition 段就是 image_latents 部分，对应索引 4096..12287，本来在 warmup-1 时已写入，region-aware 步**保持不动**）。
  - **single block** 的 Attn 输入 `hidden_states = cat([encoder_hidden_states, hidden_states_image])`，所以 selection = `[0..512] ∪ (edited_ids + 512)`，即 text 段每步都更新（因为 text 可能仍有变化），加上 edited 段。

- Q 长度只有 `len(text) + |edited|`（double block 里 Q 不含 text，只含 edited）；K/V 长度仍是全长 `N_full = 512 + 4096 + 8192 = 12800`。
- 所以 attention 复杂度从 **`N_full²`** 降到 **`Q_len × N_full`**：
  - 设 mask 比例 ρ = |edited| / 4096（典型局部编辑 ρ ≈ 0.2~0.4），则 Q_len ≈ 512 + ρ·4096，K_len = 12800。
  - 节省比例 ≈ `1 - Q_len/N_full`，往往能到 60–80% 的 attention FLOPs 砍掉，且 to_k / to_v 投影也只跑 selection 那部分。

#### (d) Scheduler 同步推进 (`RegionEFlowMatchEulerDiscreteScheduler.step`)

为了让 1-mask 和 condition 的 hidden state 在跳过的 step 里仍然"在数值上"自洽，这里用**长 dt 直接跳到下一个 refresh 节点**：

- 在 `current_step == warmup_step - 1` (例：step 5)：
  - `dt`：当前 sigma 到 next sigma，给 edited 用（正常一小步）。
  - `dt_direct = sigma[refresh_step] - sigma[current]`：给 unedited（1-mask）用，**一步把它从 step 5 直接推进到 step 16 (refresh) 的 sigma 值**。
  - 在 `prev_sample` 中按 `edited_ids / unedited_ids` 分别 scatter 回去。
- 在 `current_step == prev_refresh_step` (例：step 16)：再把 unedited 部分用 `dt_direct = sigma[next_refresh] - sigma[refresh]` 推进到下一个 refresh。
- 在 `current_step == infer_step - post_step` (例：step 26)：把 `unedited_latent`（缓存中的 1-mask 部分）`scatter` 回到全长 latent，恢复完整序列继续跑 post 步。

### 3. 直接回答你的两个问题

> **Q1：在 step 7–27 期间，1-mask 部分的 hidden state 怎么算？**
- **不再算**。这些 step 的 transformer 输入只剩 edited（mask）+ text，1-mask 区域的 hidden state 不进 transformer。
- 在 attention 内部，1-mask 对应的 token 仍然出现在 K/V 里——但 K、V 是 **`k_cache / v_cache`**，是 step 5 (`warmup_step-1`) 那一次全长 forward 时算好留下来的（每个 attention 层各一份），region-aware step 不去刷新它们。
- 在隐空间侧，1-mask 部分的 latent 也不会跟着 mask 一起被 `scheduler.step` 一小步一小步更新，而是被 `MANAGER.unedited_latent` 单独缓存，并且**在 step 5 一次性用 `dt_direct` 推进到下一个 refresh 节点的 sigma**（step 16 或 step 26）；refresh 时再 scatter 回完整 latent，重新跑一次全长 forward 来刷新 cache 和 unedited_latent。

> **Q2：condition 部分的 hidden state 怎么算？**
- region-aware step 里，`latent_model_input` 不再 cat `image_latents`，所以 condition 也不进 transformer。
- 但 attention 的 K/V 中 condition 段（索引 `[len(text)+len(noise) .. len(text)+len(noise)+len(cond)]`，即 4608..12800）始终保留**最近一次全长 step（warmup-1 或 refresh）**算出来的 K、V。`_partially_linear` 的 `selection` **永远不包含 condition 索引**，所以从来不去写 cache 里 condition 对应的位置——它们一路被复用，直到下一个 refresh / post step 触发全长重算。
- 由于 condition latent 来自固定输入图（不会被 scheduler 推进），而 attention 的 K/V 又是某一时刻 timestep 嵌入下的产物，所以"条件不变"是合理近似；但 timestep 越远，cache 越偏离真值——这就是为什么要 refresh_step 和 cache_threshold 自适应控制：`gamma * (1 + (t - t_prev)/1000)` 累乘成 `accumulate`，`error = 1 - accumulate` 超过 `cache_threshold` 就强制走全长 forward。

---

## 三、移植到 ImageCritic 的对照清单（最小修改集）

| 改动点 | ImageCritic 对应位置 | 移植要点 |
|---|---|---|
| 1) `MANAGER` 持有所有 region-aware 状态 | 在 `kontext_custom_pipeline.py` 顶层引入 `Manager` | `condition_latent` 要存 cat 后的 `image_latents`（A+B 合并）；长度 `condition_length = 8192`（双 condition 与 RegionE 单 condition 不同，**`token_selector` 也要改成对 noise vs A 或 noise vs B 的相似度——双 condition 这里需要显式选哪一张作为参考**，建议跟 IMG2 (B) 比，因为 prompt 的语义就是"refine IMG2"） |
| 2) Pipeline 改造 | `FluxKontextPipelineWithPhotoEncoderAddTokens.__call__` 的去噪 for 循环 | 复刻 RegionE 的"按阶段决定 latent_model_input 是否 cat condition"和"MANAGER.step + scheduler 改 dt"逻辑；保留 detail_encoder 注入流程 |
| 3) Transformer forward | `src/transformer_flux.py` `FluxTransformer2DModel.forward` 中的 `pos_embed` | 加一行 `MANAGER.image_rotary_emb = self.pos_embed(cat(txt_ids, MANAGER.latent_ids))`；其余结构（含 cond_hidden_states 这条支路）需要决定**cond 路是否仍被启用**——RegionE 是直接关掉条件分支的，ImageCritic 现在 cond_hidden_states 会被 `x_embedder(cond_hidden_states)`（注意当 latent_model_input 不再 cat condition 时，这里要避开 None） |
| 4) Attention processor | `src/layers.py` `MultiDoubleStreamBlockLoraProcessor` / `MultiSingleStreamBlockLoraProcessor` | 在 `to_q/to_k/to_v` 处插入和 `RegoionEFluxAttnProcessor2_0` 相同的三段式分支（warmup → cache→ region-aware partially_linear）；需要把 LoRA 同样按 `selection` 只投影 selection 子集（关键：LoRA 的 down/up 要对 selected 部分 hidden_states 重新算并加到对应位置的 cache 上，否则 cache 会丢 LoRA 增量）。这是和 RegionE 不同的最棘手点——RegionE 不带 LoRA，ImageCritic 带 LoRA |
| 5) Scheduler | 替换为 `RegionEFlowMatchEulerDiscreteScheduler` | 直接照搬 |
| 6) RoPE 长度对齐 | LoRA processor 内部 | Q 用短 RoPE，K 用长 RoPE（即 `MANAGER.image_rotary_emb`），同时 text 段也要对齐 |
| 7) Triton kernel 复用 | `fused_kernels.py` | 直接复用 `_partially_linear`；注意 LoRA 分支需要再写一个 partially-LoRA kernel，或者用 `selection` 切片在 Python 端跑（小 rank 时损失可控） |

### 双 condition 的特殊取舍
ImageCritic 的 `image_latents = cat(image_latents_A, image_latents_B, dim=1)` 长度是 RegionE 单 condition 的 2×，是 attention 开销主要来源。token_selector 用谁做 reference：
- 直接用 image_B（生成图）做相似度：和 RegionE 一致，识别"已经基本对"的区域不再 attend。
- 用 image_A（参考图 product）：意味着取出与参考一致的区域作为 unedited，几乎所有 product 像素都会被划进 unedited，效果上接近"只在背景区域重算"，与本身任务方向相反，不可取。
- 推荐：以 B 为基准。

---

## 四、TL;DR

- ImageCritic 单步推理瓶颈在 **57 层 attention**，序列长度 N=512+4096+8192=12800（双 condition 是主因）。
- RegionE 通过 (i) 仅在 warmup（前 6 步）、refresh_step（第 16 步）、post（最后 2 步）跑全长 forward 刷新 K/V cache 与 unedited latent；(ii) 中间 step 只算 mask 部分的 Q（K/V 用 cache，其中 mask 位置由 `_partially_linear` 增量更新，1-mask 与 condition 永远复用旧值）；(iii) 用长 dt 把 unedited latent 一次性推进到下一个 refresh 节点；从而让 attention 复杂度从 `N²` 降到 `Q_len × N`，同时省掉 1-mask 与 condition 的 to_k/to_v/MLP/LoRA 投影。
- 用户原问"step 7–27 没算 1-mask 与 condition"实质上是**6–15 与 17–25 这两个 region-aware 区段**才如此；step 16（refresh）和 step 26,27（post）仍然全长重算。
- 在那些跳过的 step 里：**1-mask 部分的 hidden state 不再算**，其 latent 由 scheduler 用 `dt_direct` 一次性推进到下一刷新点；**condition 的 hidden state 也不再算**，attention 中沿用上一次 refresh 时的 K/V cache。

---

## 五、与 TeaCache / SageAttention 的兼容性分析

> 联网核对的论文（proxychains 拉取的 arXiv 摘要）：
> - **TeaCache**：[arXiv:2411.19108 — *Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model* (Liu et al., 2024)](https://arxiv.org/abs/2411.19108)。核心：用 timestep embedding 调制后的 noisy input 之间的 L1 差作为"输出差异"代理，累积差超阈值才算一步，否则**复用上一拍的 transformer 残差**作为本拍输出。是**沿时间轴跳整步**的 caching。
> - **SageAttention** / **SageAttention2**：[arXiv:2410.02367](https://arxiv.org/abs/2410.02367)、[arXiv:2411.10958](https://arxiv.org/abs/2411.10958)。核心：把 attention 内部 `Q,K → INT8/INT4`、`P,V → FP8` 量化，配合 outlier smoothing 与 thread-level 量化粒度，做到 plug-and-play 替换 `F.scaled_dot_product_attention`，吞吐 2.1×–3× 且无明显精度损失。是**纵向把每次 attention 计算变快**。
> - **RegionE**：[arXiv:2510.25590 — *RegionE: Adaptive Region-Aware Generation for Efficient Image Editing* (Chen et al., 2025)](https://arxiv.org/abs/2510.25590)。核心三件套：Adaptive Region Partition、Region-Instruction KV Cache、Adaptive Velocity Decay Cache；这正是源码里 `should_cache / accumulate / cache_threshold` 的对应。
> - **ImageCritic**：[arXiv:2511.20614 — *The Consistency Critic: Correcting Inconsistencies in Generated Images via Reference-Guided Attentive Alignment* (Ouyang et al., 2025)](https://arxiv.org/abs/2511.20614)。在已生成图上做 reference-guided post-editing，所以 prompt 模板是 IMG1（参考图）+ IMG2（待修图）。

下面分别分析这三种加速能否堆叠。

### 1. RegionE × TeaCache —— **会重叠，不要叠满，但能"互补地"配**

#### 重叠点（必须处理）
- RegionE 已经主动在 region-aware step（默认 6–15、17–25）里用 `should_cache` 走一条"复用上一步 noise_pred 并按 `gamma·(1+(t-t_prev)/1000)` 做线性衰减"的快路径——这就是论文里的 **Adaptive Velocity Decay Cache**，本质是和 TeaCache **同维度的"沿时间轴跳整步"** 缓存。
- TeaCache 跳步靠的是 *timestep-embed-modulated input 的 L1 累积差 ≥ rel_l1_thresh* 触发，跳的时候直接把上一拍的 `residual = output - input` 加到当前 input 上。也就是说：
  - 如果 RegionE 在 step k 已经决定 `should_cache=True`、根本没跑 transformer，那 TeaCache 在那一步也无 input 可比、无残差可加——**两者抢同一个动作**。
  - 如果 RegionE 决定 `should_cache=False` 走 transformer，TeaCache 仍然可以基于 input 差异决定再跳一次，相当于**两层独立的跳步阈值**叠加，跳得过狠会破坏 RegionE 的 cache_threshold 误差预算。
- 还有一个"重复浪费"：RegionE 的 warmup（前 6 步）是绝对要算的，因为它要拿全长 noise_pred 来稳定噪声分布并准备 mask 划分；如果 TeaCache 在 warmup 里跳了 step 4 或 5，会让 step 5 (`warmup_step-1`) 里用来做 mask 划分的 `onestep_estimated_latent = sample + dt_final * model_output` 用到一个被复用的旧 model_output，**mask 划分的精度会显著下降**。同理 refresh_step (step 16) 和 post（step 26,27）也是 RegionE 用来刷新 KV cache 与边界的关键节点，绝不能被 TeaCache 跳掉。

#### 共存的安全策略（推荐）
- **TeaCache 仅在 RegionE 的"全长 forward" 区段（warmup, refresh, post 之外的 region-aware step）禁用**——而那恰好就是 RegionE 已经在做"velocity decay cache"的位置，所以 TeaCache 的额外贡献几乎为零。
- 反过来：**TeaCache 只在 warmup 末尾不能动、refresh 不能动、post 不能动以外的全长 step**生效。但 28 步设置下，全长 step ≈ 6 + 1 + 2 = 9，区间内已经被 RegionE 的 cache 覆盖，TeaCache 实质能介入的 step 极少。
- 因此结论是：**理论上能"非重复地"共存，但实际增益基本被 RegionE 的 Velocity Decay Cache 吃掉**；强行同时打开两个跳步逻辑容易把 cache_threshold 与 rel_l1_thresh 的误差预算累加，画质明显劣化。
- 如果想取 TeaCache 的好处，更划算的做法是：**用 TeaCache 替换掉 RegionE 的 Velocity Decay Cache（保留 Region Partition + KV Cache，把 should_cache 决策切换成 TeaCache 的 L1 阈值）**，因为 TeaCache 是沿 timestep-embedding 维度更精细的差异预测（而 RegionE 的 velocity decay 是单一 gamma 表）。

### 2. RegionE × SageAttention —— **完全正交，强烈推荐叠**

- SageAttention 改的是 `F.scaled_dot_product_attention` 内部，把 `softmax(QKᵀ/√d)V` 换成量化矩阵乘。RegionE 改的是 **谁进 Q、谁进 K/V**（mask 控制 Q 长度、cache 控制 K/V 长度），两者**操作维度正交**：RegionE 砍 token 数、SageAttention 砍单次 attention 的 ms。
- 实际叠加方式：把 `inplace.py` 里 `RegoionEFluxAttnProcessor2_0` 用到的 `F.scaled_dot_product_attention(query, key, value, ...)` 与 `flash_attn_func(...)` 两条分支统一替换为 `sageattn(...)`（thu-ml/SageAttention 仓库提供的 drop-in API）。注意：
  - SageAttention 对 head_dim 有要求（典型 64/128 都支持），Flux 的 `head_dim=128`，OK。
  - SageAttention 量化对 RoPE 后的 Q、K 一样适用，因为 SageAttn 内部对 Q、K 做 channel/thread-level smoothing；RegionE 把 Q 用短 RoPE、K 用长 RoPE 这个事实**对 Sage 透明**，无需额外修改。
  - `_partially_linear`（写 K/V cache）发生在 attention 之前，SageAttention 不参与；它只在最后那一次 SDPA 上接管。
- 因此 **RegionE + SageAttention 是无副作用叠加**，是两条不同维度的优化（行/列 vs. bit-width）。

### 3. TeaCache × SageAttention —— **正交可叠**
两者一个在 timestep 轴上跳整步、一个在 attention bit-width 上压缩单步开销，互不感知。需要保证 SageAttention 做的"低精度 attention"输出在 TeaCache 的 L1 差预测里仍然单调，但实测 SageAttention2 的精度损失基本可忽略，对 TeaCache 阈值估计影响很小。

### 4. 三者共存 —— **2+1 或 1+1+轻量混用**
- 推荐组合：**RegionE（含其 Velocity Decay Cache）+ SageAttention2** —— 即一个跳 token、一个压 attention，**TeaCache 关闭**或仅作为消融对照。
- 想吃满 caching 收益又不想跑 RegionE 的 Velocity Decay：**RegionE Region Partition + Region-Instruction KV Cache + TeaCache 跳步 + SageAttention2** —— 即 TeaCache 接手 should_cache 的角色，并和 SageAttention 正交叠加；但务必把 TeaCache 的"禁跳 step 列表"设为 `{0..warmup_step-1, refresh_step, infer-post..infer-1}`。

---

## 六、把 TeaCache + SageAttention 加进 ImageCritic 的代码改动指南

> 以下假设你已经按第三章"移植到 ImageCritic 的最小修改集"把 RegionE 移植完毕（即新建了 `MANAGER`、改了 pipeline、attention processor、scheduler 等）。下面只描述**额外**要加的部分。

### 6.1 SageAttention 接入（最低风险，建议先做）

**安装**：
```bash
proxychains4 pip install sageattention   # 或者从 https://github.com/thu-ml/SageAttention 源码安装
```

**改动点 1 — `src/layers.py` 中所有 `F.scaled_dot_product_attention` 调用**：
RegionE 移植版本里 attention 在 `MultiDoubleStreamBlockLoraProcessor` / `MultiSingleStreamBlockLoraProcessor` 的 `__call__` 末尾。把：

```python
hidden_states = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
)
```

改成（双后端兜底）：

```python
try:
    from sageattention import sageattn
    _HAS_SAGE = True
except ImportError:
    _HAS_SAGE = False

if _HAS_SAGE and attention_mask is None and not self.training:
    # sageattn 期待 (B, H, N, D) 布局，这里 query/key/value 已经是这个布局
    hidden_states = sageattn(query, key, value, is_causal=False)
else:
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
```

> 注意：
> - SageAttention 不支持 `attn_mask`（不是 None 时退回 SDPA），ImageCritic 这条路径里 mask=None，可以直接走 sage。
> - dtype 必须是 fp16 或 bf16。pipeline 默认 bf16，匹配。
> - **必须放在 `apply_rotary_emb` 之后**，因为 sage 要量化的是 RoPE 后的 Q,K（保持和 SDPA 一致）。

**改动点 2 — DetailEncoder 的 CLIP-ViT 内部不要替换**。CLIP-ViT 是一次性前向、量级很小，且 transformers 库内部 attn 不容易 monkey-patch；动它收益小风险高。

**改动点 3 — 当前推理脚本头部加全局开关**：
```python
import os
os.environ.setdefault("SAGEATTN_BACKEND", "auto")  # 或 "triton" / "cuda"
```

### 6.2 TeaCache 接入（仅在选择"用 TeaCache 替换 RegionE Velocity Decay Cache"时）

**核心数据结构（pipeline 类内 self 字段）**：

```python
# 放在 RegionEFluxKontextPipeline.__init__ 之后或 __call__ 开头初始化
self.tea_enable = True
self.tea_rel_l1_thresh = 0.4   # Flux 上经验阈值
self.tea_accumulated_rel_l1 = 0.0
self.tea_prev_input = None       # 上一拍 modulated input
self.tea_prev_residual = None    # 上一拍 (output - input)
```

**改动点 1 — pipeline 去噪循环里替换 should_cache 判定**（替换 `inplace.py`/移植版 pipeline 中那段 if 判定）：

```python
# 强制不允许跳的 step（保护 RegionE 的关键节点）
hard_step = (
    MANAGER.current_step <= MANAGER.warmup_step
    or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1
    or MANAGER.current_step == MANAGER.prev_refresh_step
)

if hard_step:
    should_cache = False
    self.tea_accumulated_rel_l1 = 0.0
else:
    # 1) 取 timestep-embedded modulated input —— TeaCache 论文用的是
    #    transformer 第一层 norm1 之前的 modulation 输出 (1+scale)*x + shift
    #    在 Flux 上简化为：用 (timestep_embedding, hidden_states) 的乘积
    with torch.no_grad():
        # 临时跑 time_text_embed + 第一层 norm1 的 chunk，得到 modulated input
        temb = self.transformer.time_text_embed(
            (timestep / 1000) * 1000, guidance, pooled_prompt_embeds
        )
        # 取第一层 transformer_blocks[0].norm1 的 (scale, shift)
        norm1 = self.transformer.transformer_blocks[0].norm1
        # AdaLayerNormZero: emb -> 6 * dim, 切第 1、2 段是 shift_msa, scale_msa
        emb_out = norm1.linear(norm1.silu(temb))
        shift_msa, scale_msa = emb_out.chunk(6, dim=1)[1:3]
        modulated_input = (1 + scale_msa[:, None]) * latents + shift_msa[:, None]
        # 仅取 noise 段（4096 或 mask 缩短后的长度）
        # 形状 [1, K, C] —— K 已经被 RegionE gather 过

    if self.tea_prev_input is None:
        rel_l1 = float("inf")
    else:
        rel_l1 = ((modulated_input - self.tea_prev_input).abs().mean()
                  / self.tea_prev_input.abs().mean()).item()
    self.tea_accumulated_rel_l1 += rel_l1
    self.tea_prev_input = modulated_input

    if self.tea_accumulated_rel_l1 < self.tea_rel_l1_thresh and self.tea_prev_residual is not None:
        should_cache = True
    else:
        should_cache = False
        self.tea_accumulated_rel_l1 = 0.0
```

**改动点 2 — caching 数学**（pipeline 循环里）：

```python
if should_cache:
    # TeaCache: 上一拍的残差直接加到本拍的 latents 上
    noise_pred = latents + self.tea_prev_residual    # 仍然是 [1, K, C]
    # 但 RegionE 后续 scheduler.step 期望 noise_pred 形状 == latents.shape，OK
else:
    # 跑 transformer …
    noise_pred = self.transformer(...)[0][:, :latents.size(1)]
    self.tea_prev_residual = noise_pred - latents     # 注意 latents 是 transformer 入口 latents（pre x_embedder 那个）
```

> 这里有一个**已知冲突点**：RegionE 在 region-aware step 里 `latents` 已经是 mask 后的子序列（长度 K），而 refresh/post 步又恢复到全长（4096 或 12288）。**TeaCache 的 prev_residual 必须随 RegionE 阶段切换而 reset**，否则 mask 切到全长那一步会因长度不匹配崩。建议在 `MANAGER.step` 切换时（warmup→region、region→refresh、refresh→region、region→post 共 4 个边界）一并清空：

```python
# 在 MANAGER.step 末尾或每次切换 latent_ids 时
self.tea_prev_input = None
self.tea_prev_residual = None
self.tea_accumulated_rel_l1 = 0.0
```

**改动点 3 — 关闭 RegionE 的 velocity decay cache**（即不要把 `should_cache` 同时由 TeaCache 和 RegionE 的 gamma 公式赋值）。把 `inplace.py` 那段 `gamma[i-1]` 累乘的 if/else 删掉即可。

### 6.3 注意事项与陷阱

1. **TeaCache 的 modulated input 取自第一层 norm1**，论文里是把整张图的 input 做 timestep-embedding 调制后求 L1 差。RegionE 截短后，调制只对 mask 段做，**rel_l1 阈值需重新校准**（默认 0.4 是全长场景的经验，截短到约 30% 长度时阈值应相应放大，否则跳得过激）。
2. **SageAttention 只对 SDPA 生效**，对 RegionE 的 `_partially_linear`（写 K/V cache 的 Triton kernel）不构成增益也不构成冲突。`_partially_linear` 是 fp16 写入，SageAttention 量化的是后面 SDPA 阶段的 Q/K/V，类型一致，无需额外 cast。
3. **LoRA 与 SageAttention**：ImageCritic 的 LoRA 在 Q/K/V 投影上叠加（在 SDPA 之前），完全在 SageAttention 之外，不互相干扰。
4. **数值稳定性顺序建议**：先单独打 RegionE → 验收 PSNR/CLIP；再叠 SageAttention → 重新验收；最后再决定是否加 TeaCache（带阈值扫）。
5. **profiler 实测**（建议）：用 `torch.profiler` 在三种组合上跑同一张图（RegionE 单开 / RegionE+Sage / RegionE+Sage+Tea），对比每段的 wall-clock。RegionE 论文给出 Flux Kontext 上 2.41× 加速；叠 SageAttention2 通常再得到 1.5–2× attention 加速（端到端 1.2–1.5×）；TeaCache 在 RegionE 已经 cache 的前提下增益小（≤1.1×）且有画质风险。

### 6.4 一图概括三者的"操作维度"

```
                    时间轴 (timestep)
                    ───────────►
                    step 0  …  27
       ┌─────────────────────────────────┐
token  │  ████████████████████████████   │  ← RegionE：把图片 token 分成 edited/unedited
轴     │  ▓▓ █████ ▓▓ █████ ▓▓           │      只对 ▓ (mask) 跑全部 step
 │     │  ▓▓ █████ ▓▓ █████ ▓▓           │      ▓ 段 K/V 仍然全长 attend (cache)
 ▼     └─────────────────────────────────┘
                    │
                    │ ←── TeaCache：判断这一整步是否要算，否则复用上一拍残差
                    │
                    ▼
            ┌───────────────────┐
            │ Q · Kᵀ · V (SDPA) │  ←── SageAttention：把这一格 attention 量化成 INT8/INT4 + FP8
            └───────────────────┘
```

三者命中的是三条不同坐标轴：**RegionE = token 轴**，**TeaCache = timestep 轴**，**SageAttention = bit-width 轴**。RegionE 与 TeaCache 在 timestep 轴上有重叠（RegionE 自带 velocity decay），所以二者要么互斥要么替换；RegionE × SageAttention 与 TeaCache × SageAttention 都是干净正交。
