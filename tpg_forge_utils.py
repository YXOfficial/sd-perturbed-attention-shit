
from contextlib import suppress
from typing import Optional, Tuple, List

import torch

BACKEND = None

try:
    from ldm_patched.ldm.modules.attention import optimized_attention
    from ldm_patched.modules.model_patcher import ModelPatcher
    BACKEND = "reForge"
except ImportError:
    from backend.attention import attention_function as optimized_attention
    from backend.patcher.base import ModelPatcher
    BACKEND = "Forge"

# minimal helpers reused from existing utils
def parse_unet_blocks(model, block_list: str, target: Optional[str]):
    # parse like "input 0-2,middle,output 0-1, d2.2-9" style; fallback: all blocks
    from .guidance_utils import parse_unet_blocks as _parse
    return _parse(model, block_list, target)

def within_sigma_range(sigma, s_start: float, s_end: float) -> bool:
    if s_start == float('inf') and s_end < 0:
        return True
    if s_end < 0:
        return sigma >= s_start
    return (sigma >= s_start) and (sigma <= s_end)

def _shuffle_tokens_like_kv(k, v):
    # k, v: (B, H, T, D)
    B,H,T,D = k.shape
    idx = torch.randperm(T, device=k.device)
    return k[:,:,idx,:], v[:,:,idx,:]

def tpg_attn2_replace_wrapper(scale: float, sigma_start: float, sigma_end: float):
    # returns a function(attn2, q, k, v, extra_options) -> z (like optimized_attention replacement)
    def replace_fn(q, k, v, extra_options):
        # extra_options provides "cond_or_uncond", "sigma", etc in Forge; in reForge similar
        sigma = float(extra_options.get("sigma", 0.0))
        if not within_sigma_range(sigma, sigma_start, sigma_end):
            return optimized_attention(q, k, v, extra_options)

        # normal attention
        z = optimized_attention(q, k, v, extra_options)

        # perturbed attention: shuffle K/V tokens (keep Q fixed)
        k_p, v_p = _shuffle_tokens_like_kv(k, v)
        z_p = optimized_attention(q, k_p, v_p, extra_options)

        # TPG combine
        out = z + scale * (z - z_p)

        return out
    return replace_fn

class TokenPerturbationGuidance:
    def __init__(self):
        pass

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        unet_block_list: str = "",
    ):
        m = model.clone()
        inner = m.model

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        # Install replace on attn2 across selected blocks
        if BACKEND == "reForge":
            from ldm_patched.ldm.modules.attention import BasicTransformerBlock, CrossAttention
        else:
            from backend.nn.unet import BasicTransformer

        for name, module in inner.diffusion_model.named_modules():
            parts = name.split('.')
            block_name = parts[0]
            if block_names and block_name not in block_names:
                continue

            if BACKEND == "reForge":
                is_attn2 = hasattr(module, "attn2") and isinstance(getattr(module, "attn2"), CrossAttention)
                if isinstance(module, BasicTransformerBlock) and is_attn2:
                    block_id = int(parts[1]) if parts[1].isdigit() else 0
                    t_idx = None
                    if "transformer_blocks" in parts:
                        t_pos = parts.index("transformer_blocks") + 1
                        with suppress(Exception):
                            t_idx = int(parts[t_pos])

                    if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                        prev = None
                        with suppress(KeyError):
                            prev = m.patches.get(("attn2_replace", (block_name, block_id, t_idx)))
                        m.set_model_attn2_replace(tpg_attn2_replace_wrapper(scale, sigma_start, sigma_end), block_name, block_id, t_idx)
            else:
                # Forge path (similar logic, left minimal)
                if isinstance(module, BasicTransformer) and getattr(module, "attn2", None) is not None:
                    block_id = int(parts[1]) if parts[1].isdigit() else 0
                    m.set_model_attn2_replace(tpg_attn2_replace_wrapper(scale, sigma_start, sigma_end), block_name, block_id, None)

        return m
