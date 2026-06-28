import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers.models.attention_processor import FluxAttnProcessor2_0

class VisualFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    """
    自定义的Flux注意力处理器，用于保存注意力图进行可视化
    """
    
    def __init__(self, save_attention=True, save_dir="attention_maps"):
        super().__init__()
        self.save_attention = save_attention
        self.save_dir = save_dir
        self.step_counter = 0
        
        # 创建保存目录
        if self.save_attention:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def save_attention_map(self, attn_weights, layer_name="", step=None):
        """保存注意力图"""
        if not self.save_attention:
            return
            
        if step is None:
            step = self.step_counter
            
        # 取第一个batch和第一个head的注意力权重
        attn_map = attn_weights[0, 0].detach().cpu().numpy()  # [seq_len, seq_len]
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        plt.imshow(attn_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Attention Map - {layer_name} - Step {step}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f"attention_{layer_name}_step_{step}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Attention map saved to: {save_path}")
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cond: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 应用旋转位置编码
        if image_rotary_emb is not None:
            query = attn.rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = attn.rotary_emb(key, image_rotary_emb)

        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 保存注意力图
        if self.save_attention and self.step_counter % 10 == 0:  # 每10步保存一次
            layer_name = f"layer_{self.step_counter // 10}"
            self.save_attention_map(attention_probs, layer_name, self.step_counter)
        
        # 应用dropout
        attention_probs = F.dropout(attention_probs, p=attn.dropout, training=attn.training)

        # 计算输出
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if use_cond:
            # 处理条件分支的情况
            seq_len = hidden_states.shape[1]
            if seq_len % 2 == 0:
                # 假设前半部分是原始hidden_states，后半部分是条件hidden_states
                mid_point = seq_len // 2
                original_hidden_states = hidden_states[:, :mid_point, :]
                cond_hidden_states = hidden_states[:, mid_point:, :]
                
                # 分别处理
                original_output = attn.to_out[0](original_hidden_states)
                cond_output = attn.to_out[0](cond_hidden_states)
                
                if len(attn.to_out) > 1:
                    original_output = attn.to_out[1](original_output)
                    cond_output = attn.to_out[1](cond_output)
                
                self.step_counter += 1
                return original_output, cond_output
        
        # 标准输出处理
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)

        self.step_counter += 1
        return hidden_states
