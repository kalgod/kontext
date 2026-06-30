"""
SageAttention adapter for the unmodified ImageCritic Flux Kontext pipeline.

Replaces F.scaled_dot_product_attention with sageattention.sageattn (INT8
quantized attention) in:

  * MultiSingleStreamBlockLoraProcessor.__call__   (38 single blocks)
  * MultiDoubleStreamBlockLoraProcessor.__call__   (19 double blocks)
  * Optionally RegionEAttnWrapper._sparse_attention_*  (sparse middle steps)

Strategy: monkey-patch the LoRA processor classes' __call__ methods
(class-level patch).  We do NOT modify src/layers.py.  The new __call__
is byte-identical to the original except for the attention call site.

Usage:

    from src.sage_adapter import enable_sageattn, SageArgs
    set_single_lora(pipeline.transformer, ...)
    enable_sageattn(pipeline, SageArgs(scope="full"))
    # then optionally:
    # enable_teacache(pipeline, ...)
    # enable_regione(pipeline, ...)

Scopes:
  * "full"  : only LoRA processors (== full / refresh / post / warmup steps).
              When RegionE is also enabled, sparse middle steps stay on SDPA.
              Recommended baseline.
  * "all"   : LoRA processors + RegionE sparse path.  Faster but sparse Q is
              short (K~975 rows), sageattn quantization overhead may erode
              the gain.

Hardware:
  * H200 / H100 / H800: native INT8 Tensor Core, expected 1.6-2.0x vs SDPA.
  * A100 / A800       : works but smaller gain (~1.3x).
  * V100 / older      : not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb


# ============================================================================
# Args
# ============================================================================
@dataclass
class SageArgs:
    enable: bool = True
    scope: str = "full"   # "full" | "all"
    kernel: str = "auto"  # "auto" | "qk_int8_pv_fp16" | "qk_int8_pv_fp8" | "qk_int8_pv_fp16_triton"
    debug: bool = False


# ============================================================================
# Resolve the actual sageattn callable based on user choice
# ============================================================================
_SAGE_FN = None
_SAGE_NAME = "uninit"


def _resolve_sage_fn(kernel: str):
    """Return (callable, name) for the requested sageattn variant."""
    import sageattention
    if kernel == "auto":
        return sageattention.sageattn, "sageattn"
    if kernel == "qk_int8_pv_fp16":
        return sageattention.sageattn_qk_int8_pv_fp16_cuda, "sageattn_qk_int8_pv_fp16_cuda"
    if kernel == "qk_int8_pv_fp8":
        # Only available on SM89+ (Ada Lovelace) / SM90 (Hopper) with sageattn 2.x
        return sageattention.sageattn_qk_int8_pv_fp8_cuda, "sageattn_qk_int8_pv_fp8_cuda"
    if kernel == "qk_int8_pv_fp16_triton":
        return sageattention.sageattn_qk_int8_pv_fp16_triton, "sageattn_qk_int8_pv_fp16_triton"
    raise ValueError(f"Unknown sage kernel '{kernel}'")


def _sage_attn(query, key, value, *, fallback_msg_state):
    """Call sageattn with an SDPA fallback on error.  Returns same shape /
    dtype as F.scaled_dot_product_attention.

    sageattn signature: sageattn(q, k, v, is_causal=False, sm_scale=None)
    Same [B, H, N, D] tensor layout as SDPA.
    """
    try:
        return _SAGE_FN(query, key, value, is_causal=False)
    except Exception as e:
        if not fallback_msg_state.get("warned"):
            print(f"[sageattn] fallback to SDPA at first failure: {type(e).__name__}: {e}")
            fallback_msg_state["warned"] = True
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )


# Single shared per-class state so we only print the first warning per process
_FALLBACK_STATE_SINGLE = {}
_FALLBACK_STATE_DOUBLE = {}
_FALLBACK_STATE_SPARSE = {}


# ============================================================================
# Replacement __call__ for MultiSingleStreamBlockLoraProcessor
#   Identical to layers.py:93-134 except the F.scaled_dot_product_attention
#   call site (line 129) is replaced with sageattn.
# ============================================================================
def _single_call_sage(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    use_cond: bool = False,
) -> torch.FloatTensor:
    batch_size, seq_len, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    for i in range(self.n_loras):
        query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
        key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
        value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    # ---- the only diff vs original: sageattn instead of SDPA ----
    # Note: original passes attention_mask but pipeline never sets it,
    # and sageattn doesn't accept arbitrary masks. If a non-None mask
    # ever shows up we fall back to SDPA.
    if attention_mask is not None:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    else:
        hidden_states = _sage_attn(
            query, key, value, fallback_msg_state=_FALLBACK_STATE_SINGLE
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states


# ============================================================================
# Replacement __call__ for MultiDoubleStreamBlockLoraProcessor
#   Identical to layers.py:162-242 except the F.scaled_dot_product_attention
#   call site (line 224) is replaced with sageattn.
# ============================================================================
def _double_call_sage(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    use_cond: bool = False,
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    inner_dim = 3072
    head_dim = inner_dim // attn.heads
    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)

    if attn.norm_added_q is not None:
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
    if attn.norm_added_k is not None:
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    for i in range(self.n_loras):
        query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
        key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
        value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    # ---- the only diff vs original ----
    if attention_mask is not None:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    else:
        hidden_states = _sage_attn(
            query, key, value, fallback_msg_state=_FALLBACK_STATE_DOUBLE
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, : encoder_hidden_states.shape[1]],
        hidden_states[:, encoder_hidden_states.shape[1] :],
    )

    hidden_states = attn.to_out[0](hidden_states)
    for i in range(self.n_loras):
        hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
    return (hidden_states, encoder_hidden_states)


# ============================================================================
# Optional: replace RegionE sparse attention SDPA with sageattn
# ============================================================================
def _patch_regione_sparse():
    """Wrap the SDPA call inside RegionEAttnWrapper._sparse_attention_*
    methods.  Returns True if RegionE is present and was patched."""
    try:
        from src.regione_adapter import RegionEAttnWrapper
    except Exception:
        return False

    # We can't easily edit a method body in place, so we replace the whole
    # method.  But the sparse methods are 80+ lines -- too risky to copy.
    #
    # Trick: monkey-patch F.scaled_dot_product_attention WITHIN the
    # regione_adapter module's namespace only.  layers.py & elsewhere keep
    # the original SDPA.
    import src.regione_adapter as rga
    orig_sdpa = rga.F.scaled_dot_product_attention

    def routed_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if attn_mask is not None or dropout_p > 0.0:
            return orig_sdpa(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kw
            )
        try:
            return _SAGE_FN(q, k, v, is_causal=is_causal, sm_scale=scale)
        except Exception as e:
            if not _FALLBACK_STATE_SPARSE.get("warned"):
                print(f"[sageattn] sparse-path fallback: {type(e).__name__}: {e}")
                _FALLBACK_STATE_SPARSE["warned"] = True
            return orig_sdpa(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kw
            )

    # We need a wrapper namespace — replace `F` on the regione_adapter
    # module with a shim that exposes scaled_dot_product_attention as our
    # routed version.  Other F.* attributes still resolve through the real
    # torch.nn.functional via __getattr__ delegation.
    class _ShimF:
        scaled_dot_product_attention = staticmethod(routed_sdpa)
        def __getattr__(self, name):
            return getattr(orig_sdpa.__module__, name)
    # Simpler: just swap the attribute on the real F object visible to rga.
    # But that would leak to layers.py too.  Instead replace `rga.F` with
    # a tiny proxy module:
    import types as _types
    proxy = _types.ModuleType("rga_F_proxy")
    # Copy ALL attrs from torch.nn.functional, override SDPA
    real_F = rga.F
    for attr in dir(real_F):
        if not attr.startswith("_"):
            try:
                setattr(proxy, attr, getattr(real_F, attr))
            except Exception:
                pass
    proxy.scaled_dot_product_attention = routed_sdpa
    rga.F = proxy
    return True


# ============================================================================
# Public API
# ============================================================================
def enable_sageattn(pipeline, args: SageArgs) -> None:
    """Patch ImageCritic LoRA processors (and optionally RegionE sparse)
    to use SageAttention INT8 instead of SDPA.

    Must be called AFTER set_single_lora() (so the LoRA processors have
    been installed onto the transformer's attn modules — though the patch
    is class-level, it would still apply, but waiting for set_single_lora
    to finish keeps the order deterministic).

    Order vs other adapters: position is flexible because the patch is
    class-level on the LoRA processor.  Recommended order:
        set_single_lora -> enable_sageattn -> enable_teacache -> enable_regione
    """
    if not args.enable:
        return

    global _SAGE_FN, _SAGE_NAME
    _SAGE_FN, _SAGE_NAME = _resolve_sage_fn(args.kernel)

    from src.layers import (
        MultiSingleStreamBlockLoraProcessor,
        MultiDoubleStreamBlockLoraProcessor,
    )
    MultiSingleStreamBlockLoraProcessor.__call__ = _single_call_sage
    MultiDoubleStreamBlockLoraProcessor.__call__ = _double_call_sage

    msg = f"[sageattn] enabled — kernel={_SAGE_NAME}, scope={args.scope}"

    if args.scope == "all":
        ok = _patch_regione_sparse()
        if ok:
            msg += " (sparse path: sageattn)"
        else:
            msg += " (sparse path: SDPA — RegionE not loaded)"
    else:
        msg += " (sparse path: SDPA)"
    print(msg)
