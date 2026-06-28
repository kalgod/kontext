import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention

# Global variables for attention visualization
step = 0
global_timestep = 0
global_timestep2 = 0

def scaled_dot_product_average_attention_map(query, key, attn_mask=None, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = attn_weight.mean(dim=(1, 2))
    return attn_weight

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    
    
class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond = False,
    ) -> torch.FloatTensor:
                
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
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
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights


    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `context` projections.
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
        
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
             hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        return (hidden_states, encoder_hidden_states)
    
    
class MultiSingleStreamBlockLoraProcessorWithLoss(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond = False,
    ) -> torch.FloatTensor:
                
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        query = attn.to_q(hidden_states) 
        key = attn.to_k(hidden_states) 
        value = attn.to_v(hidden_states) 
        encoder_hidden_length = 512
        
        length = (hidden_states.shape[-2] - encoder_hidden_length) // 3
        
        
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
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # query_cond_a = query[:, :, encoder_hidden_length+length : encoder_hidden_length+2*length, :]
        # query_cond_b = query[:, :, encoder_hidden_length+2*length : encoder_hidden_length+3*length, :]
        
        # key_noise = key[:, :, encoder_hidden_length:encoder_hidden_length+length, :]
        
        
        # attention_probs_query_a_key_noise = scaled_dot_product_average_attention_map(query_cond_a, key_noise, attn_mask=attention_mask, is_causal=False)
        # attention_probs_query_b_key_noise = scaled_dot_product_average_attention_map(query_cond_b, key_noise, attn_mask=attention_mask, is_causal=False)
           
        # attn.attention_probs_query_a_key_noise = attention_probs_query_a_key_noise
        # attn.attention_probs_query_b_key_noise = attention_probs_query_b_key_noise
        

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states


class MultiDoubleStreamBlockLoraProcessorWithLoss(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights


    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `context` projections.
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
        length = hidden_states.shape[-2] // 3
       
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
        
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        encoder_hidden_length = 512
        
        query_cond_a = query[:, :, encoder_hidden_length+length : encoder_hidden_length+2*length, :]
        query_cond_b = query[:, :, encoder_hidden_length+2*length : encoder_hidden_length+3*length, :]
        
        key_noise = key[:, :, encoder_hidden_length:encoder_hidden_length+length, :]
            
        attention_probs_query_a_key_noise = scaled_dot_product_average_attention_map(query_cond_a, key_noise, attn_mask=attention_mask, is_causal=False)
        attention_probs_query_b_key_noise = scaled_dot_product_average_attention_map(query_cond_b, key_noise, attn_mask=attention_mask, is_causal=False)
        
        attn.attention_probs_query_a_key_noise = attention_probs_query_a_key_noise
        attn.attention_probs_query_b_key_noise = attention_probs_query_b_key_noise
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
             hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        return (hidden_states, encoder_hidden_states)
    


class MultiDoubleStreamBlockLoraProcessor_visual(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i],network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights


    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> torch.FloatTensor:
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `context` projections.
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
        length = hidden_states.shape[-2] // 3
       
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
        
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        encoder_hidden_length = 512
        
        query_cond_a = query[:, :, encoder_hidden_length+length : encoder_hidden_length+2*length, :]
        query_cond_b = query[:, :, encoder_hidden_length+2*length : encoder_hidden_length+3*length, :]
        
        key_noise = key[:, :, encoder_hidden_length:encoder_hidden_length+length, :]
            
        attention_probs_query_a_key_noise = scaled_dot_product_average_attention_map(query_cond_a, key_noise, attn_mask=attention_mask, is_causal=False)
        attention_probs_query_b_key_noise = scaled_dot_product_average_attention_map(query_cond_b, key_noise, attn_mask=attention_mask, is_causal=False)
        
        if not hasattr(attn, 'attention_probs_query_a_key_noise'):
            attn.attention_probs_query_a_key_noise = []
        if not hasattr(attn, 'attention_probs_query_b_key_noise'):
            attn.attention_probs_query_b_key_noise = []
        
        global global_timestep

        attn.attention_probs_query_a_key_noise.append((global_timestep//19, attention_probs_query_a_key_noise))
        attn.attention_probs_query_b_key_noise.append((global_timestep//19, attention_probs_query_b_key_noise))
        
        print(f"Global Timestep: {global_timestep//19}")

        global_timestep += 1

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # Linear projection (with LoRA weight applied to each proj layer)
        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
             hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        return (hidden_states, encoder_hidden_states)
    
    
    
class MultiSingleStreamBlockLoraProcessor_visual(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, n_loras=1):
        super().__init__()
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(in_features, out_features, ranks[i], network_alphas[i], device=device, dtype=dtype)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        

    def __call__(self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond = False,
    ) -> torch.FloatTensor:
                
        batch_size, seq_len, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        query = attn.to_q(hidden_states) 
        key = attn.to_k(hidden_states) 
        value = attn.to_v(hidden_states) 
        encoder_hidden_length = 512
        
        length = (hidden_states.shape[-2] - encoder_hidden_length) // 3
        
        
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
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            
        if not hasattr(attn, 'attention_probs_query_a_key_noise2'):
            attn.attention_probs_query_a_key_noise2 = []
        if not hasattr(attn, 'attention_probs_query_b_key_noise2'):
            attn.attention_probs_query_b_key_noise2 = []
        
        query_cond_a = query[:, :, encoder_hidden_length+length : encoder_hidden_length+2*length, :]
        query_cond_b = query[:, :, encoder_hidden_length+2*length : encoder_hidden_length+3*length, :]
        
        key_noise = key[:, :, encoder_hidden_length:encoder_hidden_length+length, :]
                
        attention_probs_query_a_key_noise2 = scaled_dot_product_average_attention_map(query_cond_a, key_noise, attn_mask=attention_mask, is_causal=False)
        attention_probs_query_b_key_noise2 = scaled_dot_product_average_attention_map(query_cond_b, key_noise, attn_mask=attention_mask, is_causal=False)
        
        
        global global_timestep2

        attn.attention_probs_query_a_key_noise2.append((global_timestep//38, attention_probs_query_a_key_noise2))
        attn.attention_probs_query_b_key_noise2.append((global_timestep//38, attention_probs_query_b_key_noise2))
        
        print(f"Global Timestep2: {global_timestep2//38}")

        global_timestep2 += 1

        

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states
