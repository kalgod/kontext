"""
RegionE acceleration adapter for ImageCritic — original-spec port.

Faithful port of RegionE/src/FluxKontext/inplace.py to the ImageCritic
pipeline.  Does NOT modify any file under ImageCritic/src/.  Instead it
monkey-patches:

  * pipeline.__class__.__call__   (denoising loop, MANAGER.step every step)
  * pipeline.transformer.forward  (publishes long-rope to MANAGER for
                                   sparse-step attn)
  * pipeline.scheduler.step       (Euler with dt_direct on unedited rows
                                   at warmup-1 / refresh boundary)
  * each Attention.processor      (cache-write at boundary, sparse Q/K/V
                                   with cache reuse in middle steps)

Token layout in ImageCritic:
    target_latents  -> first L rows        (the noisy latent we're denoising)
    cond_A latents  -> next L rows         (image_A reference)
    cond_B latents  -> last L rows         (image_B, the target reference)
Token selector compares 1-step-estimated target vs cond_B.

Usage in driver:
    from src.regione_adapter import enable_regione, RegionEArgs
    pipeline = ...
    set_single_lora(pipeline.transformer, ...)
    enable_regione(pipeline, RegionEArgs(...))
    image = pipeline(image_A=..., image_B=..., prompt=..., ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import inspect
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# ============================================================================
# Args
# ============================================================================
@dataclass
class RegionEArgs:
    enable: bool = True
    num_inference_steps: int = 28
    warmup_step: int = 6
    post_step: int = 2
    refresh_step: str = "16"
    threshold: float = 0.93
    erosion_dilation: bool = True
    patch_size: int = 2
    vae_scale_factor: int = 8
    debug: bool = False


# ============================================================================
# id_gather / id_scatter / token_selector  (semantics identical to RegionE)
# ============================================================================
def _ids_gather(latent: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    B, K = ids.shape
    bidx = torch.arange(B, device=latent.device).unsqueeze(1).expand(-1, K)
    return latent[bidx, ids, :]


def _ids_scatter(gathered: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B = gathered.shape[0]
    dst[torch.arange(B, device=dst.device).unsqueeze(1), ids] = gathered
    return dst


def _make_kernel(size: int, kind: str) -> torch.Tensor:
    if kind == "square":
        return torch.ones(1, 1, size, size)
    k = torch.zeros(1, 1, size, size)
    mid = size // 2
    k[0, 0, mid, :] = 1
    k[0, 0, :, mid] = 1
    return k


def _erode(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    pad = kernel.shape[-1] // 2
    kernel = kernel.float().to(image)
    out = F.conv2d(image.float(), kernel, padding=pad)
    return (out == kernel.sum()).float().squeeze()


def _dilate(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    pad = kernel.shape[-1] // 2
    kernel = kernel.float().to(image)
    out = F.conv2d(image.float(), kernel, padding=pad)
    return (out > 0).float().squeeze()


def _remove_scattered(mask: torch.Tensor) -> torch.Tensor:
    erk = _make_kernel(3, "cross").to(mask)
    dik = _make_kernel(5, "square").to(mask)
    return _dilate(_erode(mask, erk), dik)


def _token_selector(
    target_latent: torch.Tensor,
    cond_latent: torch.Tensor,
    threshold: float,
    height: int,
    width: int,
    erosion_dilation: bool,
    patch_size: int,
    vae_scale_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = F.normalize(target_latent.float(), dim=-1)
    b = F.normalize(cond_latent.float(), dim=-1)
    sim = (a * b).sum(dim=-1)
    selected_mask = sim <= threshold
    if erosion_dilation:
        h = height // (patch_size * vae_scale_factor)
        w = width // (patch_size * vae_scale_factor)
        m2d = selected_mask.float().squeeze().reshape(h, w)
        m2d = _remove_scattered(m2d)
        selected_mask = m2d.bool().flatten().unsqueeze(0)

    B, L = sim.shape
    arange = torch.arange(L, device=target_latent.device).unsqueeze(0).expand(B, -1)
    edited = arange[selected_mask].unsqueeze(0)
    n_sel = edited.shape[1]
    unedited = arange[~selected_mask].view(B, L - n_sel)
    return edited, unedited


# ============================================================================
# Manager — owns regime state across one pipeline call
# ============================================================================
class _RegionEManager:
    def __init__(self) -> None:
        self.enable = False
        self.inference_step = 28
        self.warmup_step = 6
        self.post_step = 2
        self.threshold = 0.93
        self.erosion_dilation = True
        self.patch_size = 2
        self.vae_scale_factor = 8
        self.refresh_step: List[int] = []
        # per-call mutables
        self.refresh_step_real_time: List[int] = []
        self.prev_refresh_step: Optional[int] = None
        self.next_refresh_step: Optional[int] = None
        self.current_step = 0
        # geometry
        self.txt_length = 0
        self.target_length = 0
        self.cond_length = 0           # 2L
        self.height = 0
        self.width = 0
        # latent ids of full target [L, 3] (without cond ids)
        self.full_target_ids: Optional[torch.Tensor] = None
        # selection
        self.edited_ids: Optional[torch.Tensor] = None
        self.unedited_ids: Optional[torch.Tensor] = None
        self.unedited_latent: Optional[torch.Tensor] = None
        # cond reference for token_selector
        self.cond_B_latent: Optional[torch.Tensor] = None
        # full image_rotary_emb (covers [text(512), full target(L), cond(2L)])
        # used by attn processors during sparse steps to apply rope on the
        # cached full-length keys.
        self.image_rotary_emb_full: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def configure(self, args: RegionEArgs) -> None:
        self.enable = args.enable
        self.inference_step = args.num_inference_steps
        self.warmup_step = args.warmup_step
        self.post_step = args.post_step
        self.threshold = args.threshold
        self.erosion_dilation = args.erosion_dilation
        self.patch_size = args.patch_size
        self.vae_scale_factor = args.vae_scale_factor
        self.debug = getattr(args, "debug", False)
        self.refresh_step = sorted(int(x) for x in str(args.refresh_step).split(",") if x.strip())
        if self.refresh_step:
            assert min(self.refresh_step) > self.warmup_step + 1, "refresh_step must be > warmup_step + 1"
            assert max(self.refresh_step) <= self.inference_step - self.post_step - 1
            for i in range(len(self.refresh_step) - 1):
                assert abs(self.refresh_step[i] - self.refresh_step[i + 1]) > 1, "Refresh steps must not be adjacent."
        self._refresh_extended = list(self.refresh_step) + [self.inference_step - self.post_step + 1]

    def begin_call(
        self,
        target_length: int,
        cond_length: int,
        txt_length: int,
        full_target_ids: torch.Tensor,
        cond_B_latent: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        self.target_length = target_length
        self.cond_length = cond_length
        self.txt_length = txt_length
        self.full_target_ids = full_target_ids
        self.cond_B_latent = cond_B_latent
        self.height = height
        self.width = width
        self.current_step = 0
        self.prev_refresh_step = None
        self.next_refresh_step = None
        self.refresh_step_real_time = list(self._refresh_extended)
        self.edited_ids = None
        self.unedited_ids = None
        self.unedited_latent = None
        self.image_rotary_emb_full = None

    # ---- RegionE.MANAGER.step (latent shrinking / restoring) ----
    def step(self, latent: torch.Tensor, latent_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called after scheduler.step every iteration.  Returns (latent, latent_ids)
        possibly resized.  Tracks current_step.
        """
        self.current_step += 1

        if self.current_step == self.warmup_step:
            # boundary just done: shrink for the upcoming sparse steps
            self.unedited_latent = _ids_gather(latent, self.unedited_ids)
            latent = _ids_gather(latent, self.edited_ids)
            latent_ids = _ids_gather(
                latent_ids.unsqueeze(0), self.edited_ids
            ).squeeze(0)

        elif self.current_step == self.inference_step - self.post_step:
            # entering post zone: restore full latent
            full = torch.zeros(
                latent.shape[0], self.target_length, latent.shape[-1],
                dtype=latent.dtype, device=latent.device,
            )
            full = _ids_scatter(latent, self.edited_ids, full)
            full = _ids_scatter(self.unedited_latent, self.unedited_ids, full)
            latent = full
            latent_ids = self.full_target_ids
            self.prev_refresh_step = None

        elif self.prev_refresh_step is not None and self.current_step == self.prev_refresh_step:
            # restore for the upcoming refresh full-step
            full = torch.zeros(
                latent.shape[0], self.target_length, latent.shape[-1],
                dtype=latent.dtype, device=latent.device,
            )
            full = _ids_scatter(latent, self.edited_ids, full)
            full = _ids_scatter(self.unedited_latent, self.unedited_ids, full)
            latent = full
            latent_ids = self.full_target_ids

        elif self.prev_refresh_step is not None and self.current_step == self.prev_refresh_step + 1:
            # shrink again after the refresh full-step
            self.unedited_latent = _ids_gather(latent, self.unedited_ids)
            latent = _ids_gather(latent, self.edited_ids)
            latent_ids = _ids_gather(
                latent_ids.unsqueeze(0), self.edited_ids
            ).squeeze(0)
            self.prev_refresh_step = self.next_refresh_step

        return latent, latent_ids


MANAGER = _RegionEManager()


# ============================================================================
# Attention processor wrapper
# ============================================================================
class RegionEAttnWrapper(nn.Module):
    """Wrap inner processor (LoRA / vanilla).

    MUST be an nn.Module because ImageCritic's LoRA processors ARE nn.Modules
    (they own LoRA parameters), and torch.nn.Module.__setattr__ refuses to
    swap an nn.Module child for a non-Module object.

    Three regimes (gated by MANAGER.current_step + manager refresh state):

      * pre-warmup-1 / post zone : delegate to inner (full forward)
      * cache-write boundary (warmup-1, refresh): full forward + capture
        K, V (pre-norm pre-RoPE) for later cache reuse
      * sparse middle steps : compute Q on current (shrunk) hidden_states
                              with full LoRA delta; for K/V take the cached
                              full-length tensor and overwrite only the rows
                              corresponding to the current shrunk hidden
                              (text rows + edited target rows)
    """

    def __init__(self, inner, single: bool):
        super().__init__()
        # Register inner as a submodule so its parameters stay accessible.
        # If inner is not an nn.Module (e.g. plain FluxAttnProcessor2_0),
        # store as a regular attribute via __dict__ to bypass nn.Module's
        # type check.
        if isinstance(inner, nn.Module):
            self.inner = inner
        else:
            object.__setattr__(self, "inner", inner)
        self.single = single
        # Caches store pre-norm pre-RoPE K/V at FULL length:
        #   single block:  [B, 512 + L + 2L, dim_inner]  (text + target + cond)
        #   double block:  [B,        L + 2L, dim_inner] (image side only)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        # Debug counters
        self._dbg_full = 0
        self._dbg_capture = 0
        self._dbg_sparse = 0
        self._dbg_sparse_fallback = 0

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    # ------------------------------------------------------------------
    # NOTE: we override __call__ (not forward) so that
    # `inspect.signature(processor.__call__)` exposes our named kwargs
    # (image_rotary_emb, use_cond) to diffusers' Attention.forward, which
    # filters cross_attention_kwargs based on the processor's __call__
    # signature.  The default nn.Module __call__ is `(*args, **kwargs)` and
    # that filter would silently drop every named kwarg — including
    # image_rotary_emb, which would corrupt every attention output.
    #
    # ImageCritic's LoRA processors do exactly the same.
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond: bool = False,
    ):
        m = MANAGER
        if not m.enable:
            return self._delegate(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, image_rotary_emb, use_cond)

        cur = m.current_step
        is_pre = cur < m.warmup_step - 1
        is_cache_write = (cur == m.warmup_step - 1) or (
            m.prev_refresh_step is not None and cur == m.prev_refresh_step
        )
        is_post = cur > m.inference_step - m.post_step - 1

        if is_pre or is_post:
            self._dbg_full += 1
            return self._delegate(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, image_rotary_emb, use_cond)
        if is_cache_write:
            self._dbg_capture += 1
            return self._forward_capture(attn, hidden_states, encoder_hidden_states,
                                         attention_mask, image_rotary_emb, use_cond)
        # SPARSE
        if self.k_cache is None or self.v_cache is None:
            self._dbg_sparse_fallback += 1
        else:
            self._dbg_sparse += 1
        return self._forward_sparse(attn, hidden_states, encoder_hidden_states, image_rotary_emb)

    # ------------------------------------------------------------------
    def _delegate(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, use_cond=False):
        # Inner ImageCritic processors accept (attn, hidden_states,
        # encoder_hidden_states, attention_mask, image_rotary_emb, use_cond);
        # vanilla diffusers FluxAttnProcessor2_0 doesn't accept use_cond.
        try:
            return self.inner(attn, hidden_states=hidden_states,
                              encoder_hidden_states=encoder_hidden_states,
                              attention_mask=attention_mask,
                              image_rotary_emb=image_rotary_emb,
                              use_cond=use_cond)
        except TypeError:
            return self.inner(attn, hidden_states=hidden_states,
                              encoder_hidden_states=encoder_hidden_states,
                              attention_mask=attention_mask,
                              image_rotary_emb=image_rotary_emb)

    # ------------------------------------------------------------------
    def _forward_capture(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, use_cond=False):
        """Full forward; tap K/V (pre-norm pre-RoPE) via forward hooks on
        attn.to_k / attn.to_v.  These hooks fire BEFORE LoRA delta is added
        by the inner processor; we re-add the LoRA delta ourselves so the
        cached tensors match the inner processor's exact (k_base + LoRA*x)
        composition."""
        captured: Dict[str, torch.Tensor] = {}

        def hk_k(_module, inputs, output):
            captured["k_base"] = output.detach()
            captured["k_in"] = inputs[0].detach()

        def hk_v(_module, inputs, output):
            captured["v_base"] = output.detach()

        h_k = attn.to_k.register_forward_hook(hk_k)
        h_v = attn.to_v.register_forward_hook(hk_v)
        try:
            out = self._delegate(attn, hidden_states, encoder_hidden_states,
                                 attention_mask, image_rotary_emb, use_cond)
        finally:
            h_k.remove()
            h_v.remove()

        # Compose K, V with LoRA delta (matches inner LoRA processor exactly).
        # NOTE: the inner processor calls to_k/to_v exactly once on
        # `hidden_states`, so captured['k_in'] equals hidden_states.
        k = captured["k_base"]
        v = captured["v_base"]
        n_loras = getattr(self.inner, "n_loras", 0)
        if n_loras:
            x_in = captured["k_in"]
            for i in range(n_loras):
                k = k + self.inner.lora_weights[i] * self.inner.k_loras[i](x_in)
                v = v + self.inner.lora_weights[i] * self.inner.v_loras[i](x_in)
        self.k_cache = k
        self.v_cache = v
        return out

    # ------------------------------------------------------------------
    # SPARSE FORWARD
    #
    # single block:
    #   hidden_states = [B, 512+K, dim]   (text already concat'd)
    #   image_rotary_emb (current) covers [512+K]
    #   K-cache full layout  : [B, 512+L+2L, dim_inner]
    #   selection (rows to overwrite in cache) = [0..512) ∪ [512+edited_ids]
    #
    # double block:
    #   hidden_states         = [B, K, dim]
    #   encoder_hidden_states = [B, 512, dim]
    #   image_rotary_emb (current) covers [512+K]
    #   K-cache full layout   : [B, L+2L, dim_inner]    (image side)
    #   selection             = edited_ids                 (image-local)
    # ------------------------------------------------------------------
    def _forward_sparse(self, attn, hidden_states, encoder_hidden_states, image_rotary_emb):
        m = MANAGER
        if self.k_cache is None or self.v_cache is None:
            # No cache yet (first sparse step right after warmup): fall back.
            return self._delegate(attn, hidden_states, encoder_hidden_states, None, image_rotary_emb)

        B = hidden_states.shape[0]
        H = attn.heads
        edited = m.edited_ids.to(hidden_states.device).squeeze(0)  # [K]

        # ---- Q on current hidden_states (with LoRA) ----
        n_loras = getattr(self.inner, "n_loras", 0)
        q = attn.to_q(hidden_states)
        if n_loras:
            for i in range(n_loras):
                q = q + self.inner.lora_weights[i] * self.inner.q_loras[i](hidden_states)

        # ---- K/V: write current rows into cache, then use full cache ----
        # current hidden_states only contributes to a SUBSET of cache rows.
        # for single block: the rows are [0..512) (text) ∪ [512+edited] (target)
        # for double block: the rows are [edited]          (image side cache)
        if self.single:
            # cache rows to overwrite
            text_rows = torch.arange(m.txt_length, device=hidden_states.device)
            tgt_rows = m.txt_length + edited
            full_rows = torch.cat([text_rows, tgt_rows])    # [512+K]
            # current hidden_states is exactly [text(512), target_edited(K)]
            cur_h = hidden_states
        else:
            # double block: hidden_states is image side only (= edited target rows)
            full_rows = edited                               # [K]
            cur_h = hidden_states

        cur_k = attn.to_k(cur_h)
        cur_v = attn.to_v(cur_h)
        if n_loras:
            for i in range(n_loras):
                cur_k = cur_k + self.inner.lora_weights[i] * self.inner.k_loras[i](cur_h)
                cur_v = cur_v + self.inner.lora_weights[i] * self.inner.v_loras[i](cur_h)
        # write to cache (in-place is fine, same shape)
        K_full = self.k_cache
        V_full = self.v_cache
        K_full[:, full_rows, :] = cur_k
        V_full[:, full_rows, :] = cur_v
        # cache stays same object across steps (no clone): RegionE-style.

        # ---- reshape & norm ----
        D = q.shape[-1] // H
        q = q.view(B, -1, H, D).transpose(1, 2)             # [B, H, 512+K|K, D]
        kk = K_full.view(B, K_full.shape[1], H, D).transpose(1, 2)   # [B, H, N_full, D]
        vv = V_full.view(B, V_full.shape[1], H, D).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            kk = attn.norm_k(kk)

        # ---- (double block) prepend encoder Q/K/V ----
        if not self.single:
            ehs = encoder_hidden_states
            eq = attn.add_q_proj(ehs)
            ek = attn.add_k_proj(ehs)
            ev = attn.add_v_proj(ehs)
            eq = eq.view(B, -1, H, D).transpose(1, 2)
            ek = ek.view(B, -1, H, D).transpose(1, 2)
            ev = ev.view(B, -1, H, D).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            q = torch.cat([eq, q], dim=2)        # [B, H, 512+K, D]
            kk = torch.cat([ek, kk], dim=2)      # [B, H, 512+L+2L, D]
            vv = torch.cat([ev, vv], dim=2)

        # ---- RoPE: q with current (short) rope, k with full (long) rope ----
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
        if m.image_rotary_emb_full is not None:
            kk = apply_rotary_emb(kk, m.image_rotary_emb_full)

        # ---- SDPA ----
        out = F.scaled_dot_product_attention(q, kk, vv, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, -1, H * D)
        out = out.to(q.dtype)

        # ---- output split (double) or return raw (single) ----
        if not self.single:
            encoder_attn_out = out[:, : m.txt_length, :]
            image_attn_out = out[:, m.txt_length :, :]
            hidden_out = attn.to_out[0](image_attn_out)
            if n_loras:
                for i in range(n_loras):
                    hidden_out = hidden_out + self.inner.lora_weights[i] * self.inner.proj_loras[i](image_attn_out)
            hidden_out = attn.to_out[1](hidden_out)
            encoder_out = attn.to_add_out(encoder_attn_out)
            return hidden_out, encoder_out

        return out


# ============================================================================
# Patched scheduler.step
# ============================================================================
def _make_patched_scheduler_step(scheduler):
    base_step = type(scheduler).step

    def step(self, model_output, timestep, sample, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
             s_noise=1.0, generator=None, per_token_timesteps=None, return_dict=True):
        m = MANAGER
        if not m.enable:
            return base_step(self, model_output, timestep, sample, s_churn, s_tmin, s_tmax,
                             s_noise, generator, per_token_timesteps, return_dict)

        if self.step_index is None:
            self._init_step_index(timestep)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma
        sample32 = sample.to(torch.float32)
        mo32 = model_output.to(torch.float32)

        cur = m.current_step    # NOT incremented yet (MANAGER.step bumps after)

        dt_direct = None
        if cur == m.warmup_step - 1 and len(m.refresh_step_real_time) > 0:
            m.prev_refresh_step = m.refresh_step_real_time.pop(0) - 1
            sigma_refresh = self.sigmas[m.prev_refresh_step]
            dt_direct = sigma_refresh - sigma
        elif m.prev_refresh_step is not None and cur == m.prev_refresh_step and len(m.refresh_step_real_time) > 0:
            m.next_refresh_step = m.refresh_step_real_time.pop(0) - 1
            sigma_refresh = self.sigmas[m.next_refresh_step]
            dt_direct = sigma_refresh - sigma

        if dt_direct is not None and m.edited_ids is not None and m.unedited_ids is not None:
            edited = m.edited_ids.to(sample.device)
            unedited = m.unedited_ids.to(sample.device)
            sel_s = _ids_gather(sample32, edited)
            sel_o = _ids_gather(mo32, edited)
            sel_n = sel_s + dt * sel_o
            uns_s = _ids_gather(sample32, unedited)
            uns_o = _ids_gather(mo32, unedited)
            uns_n = uns_s + dt_direct * uns_o
            prev = torch.zeros_like(sample32)
            prev = _ids_scatter(sel_n, edited, prev)
            prev = _ids_scatter(uns_n, unedited, prev)
        else:
            prev = sample32 + dt * mo32

        prev = prev.to(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev,)
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev)

    return types.MethodType(step, scheduler)


# ============================================================================
# Patched transformer.forward (publishes long-rope to MANAGER for sparse attn)
# ============================================================================
def _patch_transformer(transformer):
    orig_forward = transformer.forward

    @torch.no_grad()
    def new_forward(*args, **kwargs):
        m = MANAGER
        if not m.enable:
            return orig_forward(*args, **kwargs)

        # Compute & cache the FULL long-rope so the sparse attn processors
        # can apply it on the cached full-length keys.  RegionE does this
        # in its own custom transformer forward; here we just patch.
        txt_ids = kwargs.get("txt_ids")
        if txt_ids is None and len(args) >= 6:
            # signature varies; index 5 corresponds to txt_ids in custom src
            # but we conservatively read by name only and skip otherwise.
            pass
        if txt_ids is not None and m.full_target_ids is not None:
            # 2L cond ids = the second half of (full_target_ids... + cond_ids).
            # We don't have cond ids here; but the FULL rope used during
            # warmup is exactly self.pos_embed(cat(txt_ids, full_img_ids))
            # where full_img_ids = cat(target_ids[L,3], cond_ids[2L,3]).
            # We saved cond_ids when patching prepare_latents.
            full_img_ids = m._full_img_ids
            if full_img_ids is not None:
                if txt_ids.ndim == 3:
                    txt_ids = txt_ids[0]
                full_ids = torch.cat([txt_ids.to(full_img_ids.device), full_img_ids], dim=0)
                m.image_rotary_emb_full = transformer.pos_embed(full_ids)

        return orig_forward(*args, **kwargs)

    transformer.forward = new_forward


# ============================================================================
# Patched pipeline.__call__ — RegionE-style denoising loop
# ============================================================================
def _patch_pipeline_call(pipeline_cls):
    if getattr(pipeline_cls, "_regione_patched", False):
        return
    orig_call = pipeline_cls.__call__

    @torch.no_grad()
    def new_call(self, *cargs, **ckwargs):
        m = MANAGER
        if not m.enable:
            return orig_call(self, *cargs, **ckwargs)

        # Reset attn caches at the start of every generation
        for _, mod in self.transformer.named_modules():
            if isinstance(mod, Attention) and isinstance(mod.processor, RegionEAttnWrapper):
                mod.processor.reset_cache()

        return _regione_pipeline_call(self, orig_call, *cargs, **ckwargs)

    pipeline_cls.__call__ = new_call
    pipeline_cls._regione_patched = True


# ----------------------------------------------------------------------------
# Reimplemented pipeline call mirroring ImageCritic's
# FluxKontextPipelineWithPhotoEncoderAddTokens.__call__ structure but with
# RegionE control flow inserted (cond drop in sparse, MANAGER.step every step).
# ----------------------------------------------------------------------------
def _regione_pipeline_call(self, orig_call, *cargs, **ckwargs):
    """Call self with RegionE injected — we don't reimplement the entire
    pipeline; we let orig_call build everything up to the denoising loop and
    then steer the loop by patching `self.scheduler.step` and inserting a
    callback that runs MANAGER.step + cond drop logic.

    Strategy: bind a callback_on_step_end into the call that:
      - drives MANAGER.step
      - reshapes latents/latent_ids appropriately
      - signals next-step cond concat decision via a module-level flag
        consumed by a wrapped transformer call.
    """
    # We can't easily intercept the cond-concat decision without rewriting
    # the loop — so we DO rewrite the loop here, but only the loop, by
    # inlining a copy of the relevant slice from
    # FluxKontextPipelineWithPhotoEncoderAddTokens.__call__.  This is the
    # narrowest intrusion that achieves a faithful RegionE port.
    #
    # The arguments accepted are identical to the original call.
    return _regione_inline_call(self, *cargs, **ckwargs)


def _regione_inline_call(
    self,
    image_A=None,
    image_B=None,
    prompt=None,
    prompt_2=None,
    negative_prompt=None,
    negative_prompt_2=None,
    true_cfg_scale: float = 1.0,
    height=None,
    width=None,
    num_inference_steps: int = 28,
    sigmas=None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: int = 1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    ip_adapter_image=None,
    ip_adapter_image_embeds=None,
    negative_ip_adapter_image=None,
    negative_ip_adapter_image_embeds=None,
    negative_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    output_type: str = "pil",
    return_dict: bool = True,
    joint_attention_kwargs=None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=("latents",),
    max_sequence_length: int = 512,
    max_area: int = 1024 ** 2,
    _auto_resize: bool = True,
    trigger_word=None,
):
    """Faithful copy of FluxKontextPipelineWithPhotoEncoderAddTokens.__call__
    with RegionE control flow inserted.  Only the parts that diverge from the
    original are commented; everything else is verbatim."""
    from PIL import Image
    from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput

    # ---- inherit defaults from the original method's signature ----
    sig_defaults = inspect.signature(type(self).__mro__[1].__call__) if False else None

    # Preserve original behaviour if user passed fewer args than expected
    if trigger_word is None:
        # The custom pipeline expects trigger_word - default to ('<image_A>', '<image_B>')
        # but we won't actually use it directly since encode_prompt handles it.
        trigger_word = ("<image_A>", "<image_B>")

    multiple_of = self.vae_scale_factor * 2
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
        (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
        (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
        (1504, 688), (1568, 672),
    ]

    # 1. Preprocess image_A / image_B  (verbatim from original)
    if image_A is not None and image_B is not None and not (isinstance(image_A, torch.Tensor) and image_A.size(1) == self.latent_channels) and not (isinstance(image_B, torch.Tensor) and image_B.size(1) == self.latent_channels):
        img = image_B[0] if isinstance(image_B, list) else image_B
        image_height, image_width = self.image_processor.get_default_height_width(img)
        aspect_ratio = image_width / image_height
        if _auto_resize:
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        image_A_p = self.image_processor.resize(image_A, image_height, image_width)
        image_A_p = self.image_processor.preprocess(image_A_p, image_height, image_width)
        image_B_p = self.image_processor.resize(image_B, image_height, image_width)
        image_B_p = self.image_processor.preprocess(image_B_p, image_height, image_width)
        image_A, image_B = image_A_p, image_B_p
    else:
        raise ValueError("Image input is not supported for custom kontext pipeline.")

    # 2. Check inputs
    self.check_inputs(
        prompt, prompt_2, height, width,
        negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=list(callback_on_step_end_tensor_inputs),
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (self.joint_attention_kwargs.get("scale", None)
                  if self.joint_attention_kwargs is not None else None)
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    image_A_token_id = self.tokenizer_2.convert_tokens_to_ids(trigger_word[0])
    image_B_token_id = self.tokenizer_2.convert_tokens_to_ids(trigger_word[1])
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    (
        prompt_embeds, pooled_prompt_embeds, text_ids, text_input_ids,
    ) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        device=device, num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length, lora_scale=lora_scale,
    )

    # detail_encoder injection  (verbatim)
    if self.detail_encoder is not None:
        input_ids = text_input_ids[0]
        model_dtype = next(self.detail_encoder.parameters()).dtype
        class_token_A_index = -1
        class_token_B_index = -1
        for index, token_id in enumerate(input_ids.tolist()):
            if token_id == image_A_token_id:
                class_token_A_index = index
                break
        for index, token_id in enumerate(input_ids.tolist()):
            if token_id == image_B_token_id:
                class_token_B_index = index
                break
        cA_mask = [True if class_token_A_index <= i < class_token_A_index + 1 else False for i in range(prompt_embeds.shape[1])]
        cB_mask = [True if class_token_B_index <= i < class_token_B_index + 1 else False for i in range(prompt_embeds.shape[1])]
        cA_mask = torch.tensor(cA_mask, dtype=torch.bool).unsqueeze(0)
        cB_mask = torch.tensor(cB_mask, dtype=torch.bool).unsqueeze(0)
        A_pix = image_A.to(device=device, dtype=model_dtype).unsqueeze(0)
        B_pix = image_B.to(device=device, dtype=model_dtype).unsqueeze(0)
        prompt_embeds = self.detail_encoder(A_pix, prompt_embeds, cA_mask)
        prompt_embeds = self.detail_encoder(B_pix, prompt_embeds, cB_mask)

    # 4. Prepare latents
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = self.prepare_latents(
        image_A, image_B, batch_size * num_images_per_prompt, num_channels_latents,
        height, width, prompt_embeds.dtype, device, generator, latents,
    )
    full_img_ids = torch.cat([latent_ids, image_ids], dim=0) if image_ids is not None else latent_ids
    if image_ids is not None:
        latent_ids_full_for_loop = torch.cat([latent_ids, image_ids], dim=0)
    else:
        latent_ids_full_for_loop = latent_ids
    full_target_ids = latent_ids.clone()    # [L, 3]   (no cond)

    # ---- RegionE setup ----
    cond_B_latent = image_latents[:, image_latents.shape[1] // 2 :, :].clone() if image_latents is not None else None
    MANAGER.begin_call(
        target_length=latents.size(1),
        cond_length=image_latents.size(1) if image_latents is not None else 0,
        txt_length=text_ids.size(0),
        full_target_ids=full_target_ids,
        cond_B_latent=cond_B_latent,
        height=height, width=width,
    )
    MANAGER._full_img_ids = full_img_ids       # exposed to transformer patch

    # 5. Timesteps
    sigmas_arr = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]

    def _calc_shift(seq, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        return seq * m + b

    mu = _calc_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    from src.kontext_custom_pipeline import retrieve_timesteps  # reuse helper
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, sigmas=sigmas_arr, mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if self.joint_attention_kwargs is None:
        self._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None

    # current latent_ids tracks the (possibly shrunk) target ids; on full
    # steps it gets cat'd with cond ids before being passed to the transformer.
    latent_ids_target = latent_ids   # [L, 3] initially

    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            assert i == MANAGER.current_step

            # Decide whether cond is concat'd this step (RegionE rule)
            in_warmup = MANAGER.current_step <= MANAGER.warmup_step - 1
            in_post = MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1
            is_refresh = (MANAGER.prev_refresh_step is not None
                          and MANAGER.current_step == MANAGER.prev_refresh_step)
            cond_concat = (image_latents is not None) and (in_warmup or in_post or is_refresh)

            if MANAGER.debug:
                # Print regime + first-attn counters once per step
                first_attn = next(
                    (mod.processor for _, mod in self.transformer.named_modules()
                     if isinstance(mod, Attention) and isinstance(mod.processor, RegionEAttnWrapper)),
                    None,
                )
                if first_attn is not None:
                    print(f"[regione][step={i:02d}] cond_concat={cond_concat} "
                          f"latents={tuple(latents.shape)} edited={None if MANAGER.edited_ids is None else MANAGER.edited_ids.shape[1]} "
                          f"unedited={None if MANAGER.unedited_ids is None else MANAGER.unedited_ids.shape[1]} "
                          f"prev_refresh={MANAGER.prev_refresh_step} "
                          f"first_attn(full,cap,sparse,fb)=({first_attn._dbg_full},{first_attn._dbg_capture},{first_attn._dbg_sparse},{first_attn._dbg_sparse_fallback})")

            self._current_timestep = t
            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

            if cond_concat:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                this_latent_ids = torch.cat([latent_ids_target, image_ids], dim=0) if image_ids is not None else latent_ids_target
            else:
                latent_model_input = latents
                this_latent_ids = latent_ids_target

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=this_latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            # token_selector at warmup boundary, before scheduler.step
            if MANAGER.current_step == MANAGER.warmup_step - 1 and MANAGER.edited_ids is None:
                sigma = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_T = self.scheduler.sigmas[-1]
                dt_final = sigma_T - sigma
                onestep = latents + dt_final.to(latents.dtype) * noise_pred
                edited, unedited = _token_selector(
                    onestep, MANAGER.cond_B_latent.to(onestep.device), MANAGER.threshold,
                    MANAGER.height, MANAGER.width, MANAGER.erosion_dilation,
                    MANAGER.patch_size, MANAGER.vae_scale_factor,
                )
                MANAGER.edited_ids = edited
                MANAGER.unedited_ids = unedited

            # scheduler.step (patched: handles dt_direct on boundaries)
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            # callback
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # MANAGER.step does the latent shrink/restore + bumps current_step
            latents, latent_ids_target = MANAGER.step(latents, latent_ids_target)

    self._current_timestep = None

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return FluxPipelineOutput(images=image)


# ============================================================================
# Public API
# ============================================================================
def enable_regione(pipeline, args: RegionEArgs) -> None:
    """Patch pipeline IN-PLACE. Call AFTER set_single_lora()."""
    if not args.enable:
        return
    MANAGER.configure(args)

    # Wrap each Attention.processor.
    # ImageCritic's LoRA processors are nn.Module instances; our wrapper is
    # nn.Module too, so a Module->Module assignment via set_processor (which
    # boils down to nn.Module.__setattr__) is allowed.
    transformer = pipeline.transformer
    for name, mod in transformer.named_modules():
        if isinstance(mod, Attention):
            inner = mod.processor
            single = "single_transformer_blocks" in name
            wrapper = RegionEAttnWrapper(inner, single=single)
            mod.set_processor(wrapper)

    # Patch scheduler.step
    pipeline.scheduler.step = _make_patched_scheduler_step(pipeline.scheduler)

    # Patch transformer.forward (publishes long-rope to MANAGER)
    _patch_transformer(transformer)

    # Patch pipeline.__call__ (RegionE-style loop)
    _patch_pipeline_call(type(pipeline))
