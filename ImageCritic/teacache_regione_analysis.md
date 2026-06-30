# TeaCache + RegionE 融合到 ImageCritic：可行性分析与实现方案

本文档分析在 ImageCritic 的自定义 Flux Kontext pipeline 上同时启用 **TeaCache** 与
**RegionE** 两种加速的可行性，并给出非侵入式实现方案。

---

## 1. 三种方案的本质对比

| | RegionE | TeaCache | 两者粒度 |
|---|---|---|---|
| 加速对象 | **token 维度** —— 中间步只对 edited tokens 算 Q（K/V cache） | **timestep 维度** —— 整个 transformer 跳过若干步，复用 residual | 正交 |
| 触发条件 | warmup 后按 cosine 选 mask（target vs cond_B） | 每步算 timestep embedding 的 L1 距离阈值 | — |
| 跳过粒度 | 每步内部 sequence 维度做稀疏（attn 的 K/V 全长，Q 缩到 K） | 整步跳过 transformer，写 `latent += previous_residual` | — |
| 复用对象 | per-attn-layer 的 K/V cache + per-step LoRA proj | 一个 `(hidden_states - input)` residual 张量 | — |
| 数学等价性 | 假设 unedited token 的 noise_pred 在窗口内恒定 | 假设 transformer 整体增量在窗口内恒定 | — |
| 单独加速比 | 实测 ~2.1× | 论文 ~2.0× (rel_l1_thresh=0.6) | — |

**结论：粒度正交，可以叠加。**

- TeaCache 决定"这一步要不要跑 transformer"
- 如果跑 → RegionE 决定"sequence 哪部分参与 Q/K/V"

---

## 2. 关键挑战：两个状态机如何共存

两个加速都是有限状态机，按 step 推进。需要梳理冲突：

### 2.1 RegionE 的状态机

```
step 0 .. warmup-2  : full forward
step warmup-1       : full + 写 K/V cache + token_selector(选 edited)
step warmup .. T-post-1 (中间)
    sparse step     : Q 只对 edited 算，K/V 用 cache
    refresh step    : full + 重写 cache (中间穿插)
step T-post .. T    : full forward
```

scheduler.step：在 `warmup-1` 和 `refresh` 边界对 unedited 行用 dt_direct 一次性 Euler。

### 2.2 TeaCache 的状态机

```
每步：
  modulated_inp = norm1(hidden_states, temb)   # 廉价
  if cnt == 0 or cnt == T-1:
      should_calc = True
  else:
      delta = ||mod_inp - prev_mod_inp|| / ||prev_mod_inp||
      accumulated += rescale_func(delta)
      should_calc = (accumulated >= threshold)
  
  if should_calc:
      跑完整 transformer; previous_residual = out - in
  else:
      out = in + previous_residual    # 跳过整步 transformer
```

### 2.3 冲突点 & 处理

| 冲突点 | 描述 | 处理 |
|---|---|---|
| **C1：sparse 步不能被 TeaCache 跳过** | sparse 步 hidden_states 是 shrunk 形状 [B,K,C]，previous_residual 是上一次写入时的形状（可能是 full 长度）。直接 `+=` 会形状不匹配 | sparse 步**强制 should_calc=True** |
| **C2：cache-write 步绝不能跳** | 跳了 K/V cache 就更新不了，后续 sparse 全废 | warmup-1 / refresh 步**强制 should_calc=True** |
| **C3：post-step 阶段** | post 用 full latent + 修复细节，跳过会损失精度 | post 阶段**强制 should_calc=True**（保险） |
| **C4：previous_residual 的形状** | TeaCache 的 residual 来自 transformer 输入/输出，其 sequence 长度 = target+cond（warmup/post）或 target（sparse）。如果跨形状会 shape mismatch | residual 只在 **same-shape** 步保留有效；跨边界（warmup→sparse 或 sparse→refresh）必须 calc | 
| **C5：modulated_inp 的形状变化** | TeaCache 比较的 `modulated_inp` 在 warmup（full L+2L）和 sparse（K）形状不同 | 形状变了直接 `should_calc=True`，不与历史比较 |
| **C6：accumulated_rel_l1_distance 的语义** | 跨形状变化后这个累加量已经不可比 | 形状变了清零 |
| **C7：cnt 与 T 的对齐** | TeaCache 用 cnt% num_steps 触发首末，RegionE 也用 current_step | 共享 MANAGER.current_step（**已有**），TeaCache 只读不写 |

### 2.4 安全规则汇总

允许 TeaCache 跳过的步 = **当且仅当**：
1. `MANAGER.mode == "full"` 且非 `current_step == 0` 且非最后一步
2. 不在 warmup-1 / refresh 边界
3. 不在 post 阶段
4. 形状与 previous_residual 一致

换句话说，**TeaCache 只能在 RegionE 的 full 模式步里运作**——但实际上 RegionE 只在 warmup 前 6 步、refresh 1 步、post 2 步是 full 模式，**其中能真正跳的只有 warmup 中的步 0..warmup-2 和 post 中的非首末步**。

---

## 3. 收益评估（修正版）

设 28 步、warmup=6、post=2、refresh=[16]、edited≈11%。

**关键洞察**：sparse 步是 TeaCache 的最佳应用场景，不是禁区。理由：

- sparse 步 K/V cache 完全冻结（在 boundary 写一次后不变）
- sparse 步的 transformer 退化成 **f(latent_edited)** 的纯函数
- TeaCache 假设 `transformer(x_t) - x_t ≈ const`，纯函数比 full 模式更稳定
- sparse 步之间形状 `[B, K=463, C]` 完全一致，previous_residual 可以直接复用

**真正不能跳的步只有**：
- warmup-1（要写 K/V cache + 给 token_selector 提供 noise_pred）
- refresh 边界（要重写 K/V cache）
- TeaCache 自身首末步约束（cnt==0 / cnt==T-1）
- 边界后**第一个**形状变化的步（自动通过 `prev_residual.shape != hs.shape` 兜底）

| 阶段 | 步数 | 模式 | 能否被 TeaCache 跳 |
|---|---|---|---|
| step 0     | 1 | full       | 否（cnt==0） |
| step 1..4  | 4 | full       | **是（4 步可跳）** |
| step 5     | 1 | warmup-1   | 否（cache write） |
| step 6     | 1 | sparse 首  | 否（形状从 3L→K，自动兜底） |
| step 7..14 | 8 | sparse     | **是（8 步可跳）** |
| step 15    | 1 | refresh    | 否（cache rewrite） |
| step 16    | 1 | sparse 首  | 否（形状从 3L→K，自动兜底） |
| step 17..25| 9 | sparse     | **是（9 步可跳）** |
| step 26    | 1 | post 首    | 否（形状从 K→3L，自动兜底） |
| step 27    | 1 | post 末    | 否（cnt==T-1） |

**理论可跳：21 步 / 28 步**。rel_l1_thresh=0.4 实测可能跳 8-15 步。

### 收益估算

- baseline 纯 RegionE：23s → 11s（已实测，2.1×）
- 中间 sparse 步本身已经很便宜（~250ms vs full 步 ~800ms），跳一个 sparse 仅省 ~250ms
- 跳一个 full 步省 ~800ms（warmup 区段）
- 估计 RegionE+TeaCache 联动：~6-8s（**3.0×~3.8× 综合加速**）
- 单独 TeaCache：~10s（约 2× 加速，与论文一致）

**风险**：sparse 步累计 rel_l1_distance 用的是 vanilla FLUX 拟合的多项式系数；ImageCritic 加了 LoRA 后 modulated_inp 分布会偏移，threshold 0.4 可能偏激进。


---

## 4. 替代方案：纯 TeaCache（不开 RegionE）

如果用户不要 RegionE：直接对 ImageCritic 的自定义 transformer.forward 应用 TeaCache。

**问题**：ImageCritic 自定义 transformer 的 forward **多了 cond 分支**：

```python
hidden_states = self.x_embedder(hidden_states)
if use_condition:
    cond_hidden_states = self.x_embedder(cond_hidden_states)
...
# 每个 block 同时处理 hidden_states 和 cond_hidden_states
```

但实际 pipeline 调用时**永远不传 cond_hidden_states**（cond 被 cat 进 hidden_states 的序列维），所以 `use_condition=False`，cond 分支全部走空 if。**等于退化成普通 Flux transformer**。

→ TeaCache 原生算法可以无缝套用，因为：
1. `hidden_states.shape[1] = L + 2L = 3L`（target + cond_A + cond_B）始终一致
2. `modulated_inp` 通过第一个 block 的 norm1 计算，公式与原版相同
3. 整个 transformer 跳过 = 直接复用 `previous_residual`，形状匹配

**纯 TeaCache 是完全可行的，且改动最小。**

---

## 5. 设计：以"纯 TeaCache"为主路径，以"RegionE+TeaCache"为可选叠加

### 5.1 入口：`enable_teacache(pipeline, args)`

```python
@dataclass
class TeaCacheArgs:
    enable: bool = False
    rel_l1_thresh: float = 0.6   # 0.25/0.4/0.6/0.8 = 1.5x/1.8x/2.0x/2.25x
    num_inference_steps: int = 28
```

`enable_teacache` 执行：
1. 在 `pipeline.transformer` 上挂载状态字段（cnt, num_steps, threshold, accumulated, previous_modulated_input, previous_residual, enable_teacache）
2. 用 `types.MethodType` 替换 `pipeline.transformer.forward` 为 `teacache_forward_imagecritic`（**不动 src 文件**）

`teacache_forward_imagecritic`：
- **完全不调用** ImageCritic 的 cond 分支（`use_condition=False`）
- 用 `transformer_blocks[0].norm1` 计算 `modulated_inp` —— 这是 ImageCritic 自定义的 `AdaLayerNormZero`，签名兼容
- 算 should_calc，跳过则 `hidden_states += previous_residual`，否则跑全量 blocks 并记 residual
- shape mismatch 自动 force calc

### 5.2 与 RegionE 的兼容

如果同时 `--use_regione --use_teacache`：

**触发顺序**（每步 transformer 入口）：
1. RegionE 的 `transformer.forward` 已经被 `_patch_transformer` 包了一层，先发布 `image_rotary_emb_full`
2. TeaCache 包装 ImageCritic 原版的 forward —— 这是**最里面那一层**

为了不冲突：在 `teacache_forward_imagecritic` 里**先检测 RegionE 状态**：
```python
if MANAGER.enable and MANAGER.mode != "full":
    # sparse / refresh 步：禁用 TeaCache 旁路，走完整 forward
    should_calc = True
elif current_step == warmup_step - 1 or current_step == prev_refresh_step:
    # cache-write 边界：禁用旁路
    should_calc = True
elif current_step > inference_step - post_step - 1:
    # post 阶段：禁用旁路
    should_calc = True
else:
    # 普通 full step：走 TeaCache 决策
    ...
```

shape 检查（C4）也保留：

```python
if previous_residual is not None and previous_residual.shape != hidden_states.shape:
    should_calc = True
    accumulated_rel_l1_distance = 0
```

### 5.3 patch 链路

```
不开任何加速:
  pipeline.transformer.forward = ImageCritic 原版

只开 TeaCache:
  pipeline.transformer.forward = teacache_forward_imagecritic（含 ImageCritic blocks 的循环）

只开 RegionE:
  pipeline.transformer.forward = RegionE wrapper(ImageCritic 原版)

两者都开:
  pipeline.transformer.forward = RegionE wrapper(teacache_forward_imagecritic)
  
  RegionE wrapper:
    1. 发布 image_rotary_emb_full 到 MANAGER
    2. 调用底层 forward = teacache_forward_imagecritic
    3. teacache 内部检查 MANAGER 状态 → 只在 full 模式且非边界时考虑跳过
```

为此需要：让 RegionE 的 `_patch_transformer` 不直接绑死 `orig_forward`，而是在每次调用时拿当前 `transformer.forward`。已经满足这条——它 `orig_forward = transformer.forward` 是绑定时刻的引用，所以**必须先开 TeaCache 再开 RegionE**。

### 5.4 调用顺序

```python
set_single_lora(pipeline.transformer, ...)

# 顺序：先 TeaCache，再 RegionE
if args.use_teacache:
    enable_teacache(pipeline, TeaCacheArgs(...))   # 替换 forward

if args.use_regione:
    enable_regione(pipeline, RegionEArgs(...))     # 包装 forward + 替换 attn / scheduler
```

---

## 6. 风险 & 限制

1. **首次运行 cnt 状态污染**：TeaCache 的 cnt 是类属性（teacache_flux.py:321 用 `__class__.cnt = 0`）。这会污染所有该类的实例。我**不用类属性**，挂在 instance 上。
2. **修改噪声路径不变**：`--fixed_noise` 加载的初始 latent 与 RegionE/TeaCache 完全无关，仍然有效。
3. **预测系数**：TeaCache 用了一个针对 FLUX 的 5 阶多项式 `[4.98e+02, -2.83e+02, ...]`。这是**对原版 FLUX 模型拟合的**，ImageCritic 加了 LoRA + DetailEncoder 后 modulated_inp 的分布会有偏差，accumulated_rel_l1_distance 的物理意义会变。**实际表现可能比原版 FLUX 差**——需要实测验证。
4. **TeaCache 与 cond 序列**：ImageCritic 的 hidden_states 是 [target, cond_A, cond_B] 共 3L 长度，TeaCache 比较的是整个 3L 的 `modulated_inp[0]`。cond 部分理论上每步几乎不变（cond_A/B 是固定 VAE latent），所以 modulated_inp 的变化主要来自 target，公式仍然有效。
5. **num_inference_steps 必须正确传入**：TeaCache 用首末步 forced，错了会全程不跳或全程跳。
6. **PSNR 评估**：rel_l1_thresh=0.6 是论文推荐的 2x 配置，**单独跑应该 PSNR 25-30 dB 之间，叠加 RegionE 后会更低**。如果质量不可接受，改 0.25/0.4。

---

## 7. 验证方案

```bash
# baseline
python infer.py
# 纯 TeaCache
python infer.py --use_teacache --rel_l1_thresh 0.4
# 纯 RegionE
python infer.py --use_regione
# 双开
python infer.py --use_teacache --rel_l1_thresh 0.4 --use_regione
```

每次打印：
- `[time] pipeline (...) = X s`
- `[teacache] skipped K/N steps`（新增统计）
- `[PSNR] vs baseline`
- 加速比

---

## 8. 实现决策

✅ **可行，实现纯 TeaCache + 与 RegionE 叠加**。

新增文件：
- `src/teacache_adapter.py`：包含 `enable_teacache(pipeline, args)`、`TeaCacheArgs`、`teacache_forward_imagecritic`

修改文件：
- `infer.py`：加 `--use_teacache --rel_l1_thresh` 参数；调用顺序 lora → teacache → regione
- **不修改** `src/transformer_flux.py`、`src/kontext_custom_pipeline.py`、`src/regione_adapter.py`、`src/layers.py`
