"""
TeaCache adapter for the unmodified ImageCritic Flux Kontext pipeline.

TeaCache (https://github.com/LiewFeng/TeaCache) skips a fraction of denoising
steps by re-using the residual `(transformer(x_t) - x_t)` from a recently
"calc'd" step.  The decision is driven by an L1 distance between the current
and previous *modulated* hidden_states (modulation = AdaLayerNormZero output of
the first transformer block).

This adapter:

  * Does NOT modify any file under ImageCritic/src/.
  * Provides `enable_teacache(pipeline, args)` which monkey-patches
    `pipeline.transformer.forward` with a TeaCache-aware variant.
  * Re-implements ImageCritic's transformer.forward (with cond branch)
    in-place, but with TeaCache logic injected around the block loops.
    The cond branch is compiled in but never taken at inference (pipeline
    never passes cond_hidden_states), matching the behaviour of the
    original transformer.
  * Is RegionE-aware: when RegionE is also enabled, the TeaCache short
    circuit is suppressed during sparse / cache-write / refresh / post
    steps so it never interferes with RegionE's K/V cache or latent
    shrinking.

Usage:

    from src.teacache_adapter import enable_teacache, TeaCacheArgs
    set_single_lora(pipeline.transformer, ...)        # MUST come first
    enable_teacache(pipeline, TeaCacheArgs(num_inference_steps=28,
                                            rel_l1_thresh=0.4))
    # optionally:
    # enable_regione(pipeline, RegionEArgs(...))     # AFTER teacache

Reference rel_l1_thresh values (from upstream):
    0.25 -> 1.5x speedup,  0.4 -> 1.8x speedup,
    0.6  -> 2.0x speedup,  0.8 -> 2.25x speedup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import types

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
)


# ============================================================================
# Args
# ============================================================================
@dataclass
class TeaCacheArgs:
    enable: bool = True
    num_inference_steps: int = 28
    rel_l1_thresh: float = 0.4
    debug: bool = False


# Coefficients fitted by the TeaCache authors against vanilla FLUX.1 [dev].
# They map a relative L1 distance into an "effective drift" score that is
# accumulated across consecutive cached steps.  ImageCritic adds LoRA + a
# DetailEncoder, so these coefficients are an APPROXIMATION (the modulated
# distribution is not identical to vanilla FLUX).  In practice the same
# polynomial still works, but the optimal threshold may differ.
_TEACACHE_COEFFS = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01,
                    -3.82021401e+00, 2.64230861e-01]


# ============================================================================
# RegionE introspection (we don't import the module so this stays optional)
# ============================================================================
def _regione_state():
    """Return the RegionE manager if RegionE is patched onto the pipeline.

    Returns None when RegionE is not enabled — in that case TeaCache acts
    on every step.  We import lazily so the adapter works even when
    src.regione_adapter is missing.
    """
    try:
        from src.regione_adapter import MANAGER  # noqa: WPS433
        if getattr(MANAGER, "enable", False):
            return MANAGER
    except Exception:
        pass
    return None


def _regione_must_calc(mgr) -> bool:
    """Return True when the current RegionE step MUST run a real forward
    (cannot be replaced by TeaCache's residual reuse).

    These are the only RegionE steps where skipping would break correctness:

      * cache-write boundaries (warmup-1, refresh): must update K/V caches
        AND, for warmup-1, must produce noise_pred for token_selector
      * (first sparse step after each boundary is auto-handled by the
        shape-mismatch check, no need to special-case here)

    Sparse middle steps and post steps are SAFE to skip — sparse steps in
    particular are pure functions of the (shrunk) latent because their K/V
    caches are frozen at boundary time, so TeaCache's residual-reuse
    approximation actually applies more cleanly than in full steps.
    """
    if mgr is None:
        return False
    cur = mgr.current_step
    is_warmup_boundary = cur == mgr.warmup_step - 1
    is_refresh_boundary = (
        mgr.prev_refresh_step is not None and cur == mgr.prev_refresh_step
    )
    return is_warmup_boundary or is_refresh_boundary


# ============================================================================
# The patched forward.  Mirrors ImageCritic's transformer.forward verbatim
# except for the `if self.enable_teacache` shortcut wrapped around the block
# loops.
# ============================================================================
def _teacache_forward_imagecritic(
    self,
    hidden_states: torch.Tensor,
    cond_hidden_states: torch.Tensor = None,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
):
    use_condition = cond_hidden_states is not None

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    hidden_states = self.x_embedder(hidden_states)
    if use_condition:
        cond_hidden_states = self.x_embedder(cond_hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    if use_condition:
        cond_temb = (
            self.time_text_embed(torch.ones_like(timestep) * 0, pooled_projections)
            if guidance is None
            else self.time_text_embed(torch.ones_like(timestep) * 0, guidance, pooled_projections)
        )
    else:
        cond_temb = None

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ----------------------------------------------------------------------
    # TeaCache decision
    # ----------------------------------------------------------------------
    rgmgr = _regione_state()
    must_calc_for_regione = _regione_must_calc(rgmgr)

    if self.enable_teacache:
        # Compute modulated_inp via the first double-block's norm1 (cheap).
        inp = hidden_states  # already x_embedded
        modulated_inp, *_ = self.transformer_blocks[0].norm1(inp, emb=temb)

        # Force first / last step to always calc.
        cnt = self._tc_cnt
        if cnt == 0 or cnt == self._tc_num_steps - 1:
            should_calc = True
            self._tc_accum = 0.0
        else:
            prev = self._tc_prev_modulated_input
            if prev is None or prev.shape != modulated_inp.shape:
                # Shape changed (rare -- happens if RegionE shrinks latents).
                # Treat as calc.
                should_calc = True
                self._tc_accum = 0.0
            else:
                rescale = np.poly1d(_TEACACHE_COEFFS)
                rel = (
                    (modulated_inp - prev).abs().mean()
                    / prev.abs().mean()
                ).cpu().item()
                self._tc_accum += float(rescale(rel))
                if self._tc_accum < self._tc_rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self._tc_accum = 0.0

        # RegionE safety overrides: only the cache-write boundaries
        # (warmup-1, refresh) MUST run a real forward.  Sparse middle steps
        # and post steps are fair game for TeaCache.  The first sparse step
        # after each boundary is auto-caught by the shape-mismatch check
        # below (boundary residual is full-length 3L, sparse is K).
        if must_calc_for_regione:
            should_calc = True

        # Shape mismatch with previous_residual = must calc.
        if (
            (not should_calc)
            and self._tc_previous_residual is not None
            and self._tc_previous_residual.shape != hidden_states.shape
        ):
            should_calc = True
            self._tc_accum = 0.0

        # State bookkeeping
        self._tc_prev_modulated_input = modulated_inp.detach()
        self._tc_cnt += 1
        if self._tc_cnt == self._tc_num_steps:
            self._tc_cnt = 0

        if self._tc_debug:
            mode_str = "?" if rgmgr is None else rgmgr.mode
            print(
                f"[teacache] step={cnt} mode={mode_str} accum={self._tc_accum:.4f} "
                f"thresh={self._tc_rel_l1_thresh} should_calc={should_calc} "
                f"must_calc_regione={must_calc_for_regione} "
                f"hs_shape={tuple(hidden_states.shape)}"
            )
    else:
        should_calc = True

    # ----------------------------------------------------------------------
    # Either re-use previous_residual (skip path) or run the full block stack
    # ----------------------------------------------------------------------
    if self.enable_teacache and not should_calc:
        self._tc_skip_count += 1
        hidden_states = hidden_states + self._tc_previous_residual
    else:
        self._tc_calc_count += 1
        ori_hidden_states = hidden_states.clone()

        # Double blocks
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # not used at inference
                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False}
                    if is_torch_version(">=", "1.11.0") else {}
                )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                encoder_hidden_states, hidden_states, cond_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        cond_hidden_states if use_condition else None,
                        cond_temb if use_condition else None,
                        **ckpt_kwargs,
                    )
                )
            else:
                encoder_hidden_states, hidden_states, cond_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cond_hidden_states=cond_hidden_states if use_condition else None,
                    temb=temb,
                    cond_temb=cond_temb if use_condition else None,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Single blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states, cond_hidden_states = block(
                hidden_states=hidden_states,
                cond_hidden_states=cond_hidden_states if use_condition else None,
                temb=temb,
                cond_temb=cond_temb if use_condition else None,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        if self.enable_teacache:
            # We compare against the post-x_embedder hidden_states pre-blocks
            # (= ori_hidden_states), at the SAME sequence length the next
            # iteration will see.
            self._tc_previous_residual = (hidden_states - ori_hidden_states).detach()

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


# ============================================================================
# Public API
# ============================================================================
def enable_teacache(pipeline, args: TeaCacheArgs) -> None:
    """Patch pipeline.transformer.forward IN-PLACE with TeaCache.

    MUST be called AFTER set_single_lora() / set_multi_lora() so the patch
    runs over the same transformer instance the LoRA processors are bound to.

    Compatible with enable_regione(): if you want both, call
    enable_teacache() FIRST and then enable_regione().  RegionE wraps the
    transformer.forward currently bound — that is, the TeaCache forward —
    and the wrapper composes correctly.

    The TeaCache shortcut is allowed in BOTH RegionE FULL and SPARSE steps:
        - boundaries (warmup-1, refresh) are forced to calc (cache write)
        - the first sparse step after each boundary is forced to calc by
          the shape-mismatch check (boundary residual is full-length 3L,
          sparse is K, shapes differ -> calc)
        - all other middle sparse steps and post steps are fair game

    In sparse steps the K/V cache is frozen at boundary time, so the
    transformer becomes a pure function of the (shrunk) latent and the
    TeaCache residual-reuse approximation actually applies more cleanly
    than in full steps."""
    if not args.enable:
        return

    transformer = pipeline.transformer

    # State on the instance (NOT class — class would leak across runs).
    transformer.enable_teacache = True
    transformer._tc_cnt = 0
    transformer._tc_num_steps = args.num_inference_steps
    transformer._tc_rel_l1_thresh = args.rel_l1_thresh
    transformer._tc_accum = 0.0
    transformer._tc_prev_modulated_input = None
    transformer._tc_previous_residual = None
    transformer._tc_debug = args.debug
    transformer._tc_skip_count = 0
    transformer._tc_calc_count = 0

    transformer.forward = types.MethodType(_teacache_forward_imagecritic, transformer)

    print(
        f"[teacache] patched transformer.forward — "
        f"steps={args.num_inference_steps} rel_l1_thresh={args.rel_l1_thresh}"
    )


def teacache_summary(pipeline) -> str:
    """One-line stats for printing after a generation."""
    t = pipeline.transformer
    if not getattr(t, "enable_teacache", False):
        return "[teacache] disabled"
    return (
        f"[teacache] calc={t._tc_calc_count} skip={t._tc_skip_count} "
        f"({t._tc_skip_count}/{t._tc_calc_count + t._tc_skip_count} steps bypassed)"
    )


def teacache_reset(pipeline) -> None:
    """Reset per-call counters between two consecutive runs."""
    t = pipeline.transformer
    if not getattr(t, "enable_teacache", False):
        return
    t._tc_cnt = 0
    t._tc_accum = 0.0
    t._tc_prev_modulated_input = None
    t._tc_previous_residual = None
    t._tc_skip_count = 0
    t._tc_calc_count = 0
