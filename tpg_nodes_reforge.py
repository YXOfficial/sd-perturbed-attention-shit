
from functools import partial

BACKEND = None

# Try Comfy first (for compatibility), then reForge, then Forge
try:
    from comfy.ldm.modules.attention import optimized_attention
    from comfy.model_patcher import ModelPatcher

    from .guidance_utils import (
        parse_unet_blocks,
        rescale_guidance,
    )

    BACKEND = "ComfyUI"
except Exception:
    try:
        from ldm_patched.ldm.modules.attention import optimized_attention
        from ldm_patched.modules.model_patcher import ModelPatcher

        from .guidance_utils import (
            parse_unet_blocks,
            rescale_guidance,
        )

        BACKEND = "reForge"
    except Exception:
        from backend.attention import attention_function as optimized_attention
        from backend.patcher.base import ModelPatcher

        from .guidance_utils import (
            parse_unet_blocks,
            rescale_guidance,
        )

        BACKEND = "Forge"


def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)


def _shuffle_kv(k, v):
    # k,v: (B,H,T,D)
    import torch
    B,H,T,D = k.shape
    perm = torch.randperm(T, device=k.device)
    return k[:, :, perm, :], v[:, :, perm, :]


def tpg_replace(scale: float, s0: float, s1: float, rescale: float, mode: str):
    # expected signature: fn(q,k,v,extra_options)->z
    def _fn(q, k, v, extra_options):
        z = optimized_attention(q, k, v, extra_options)
        sigma = float(extra_options.get("sigma", 0.0))
        if scale == 0.0 or not _within_sigma(sigma, s0, s1):
            return z

        k_p, v_p = _shuffle_kv(k, v)
        z_p = optimized_attention(q, k_p, v_p, extra_options)

        out = z + scale * (z - z_p)
        if rescale > 0.0:
            out = rescale_guidance(out, z, rescale, mode)
        return out
    return _fn


class TokenPerturbationGuidance:
    # Minimal WebUI operator like PAG/SEG with .patch returning (ModelPatcher,)
    def __init__(self):
        pass

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()
        inner_model = m.model

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        if BACKEND == "reForge":
            from ldm_patched.ldm.modules.attention import BasicTransformerBlock, CrossAttention
        elif BACKEND == "Forge":
            from backend.nn.unet import BasicTransformer
        else:
            from comfy.ldm.modules.attention import BasicTransformerBlock as BasicTransformer

        for name, module in inner_model.diffusion_model.named_modules():
            parts = name.split(".")
            block_name = parts[0]
            if block_names and block_name not in block_names:
                continue

            if BACKEND == "reForge":
                if isinstance(module, BasicTransformerBlock) and hasattr(module, "attn2"):
                    try:
                        block_id = int(parts[1])
                    except Exception:
                        block_id = 0
                    t_idx = None
                    if "transformer_blocks" in parts:
                        pos = parts.index("transformer_blocks") + 1
                        try:
                            t_idx = int(parts[pos])
                        except Exception:
                            t_idx = None
                    # filter by parsed blocks
                    if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                        patch = tpg_replace(scale, sigma_start, sigma_end, rescale, rescale_mode)
                        m.set_model_attn2_replace(patch, block_name, block_id, t_idx)
            elif BACKEND == "Forge":
                if isinstance(module, BasicTransformer) and getattr(module, "attn2", None) is not None:
                    try:
                        block_id = int(parts[1])
                    except Exception:
                        block_id = 0
                    patch = tpg_replace(scale, sigma_start, sigma_end, rescale, rescale_mode)
                    m.set_model_attn2_replace(patch, block_name, block_id, None)
            else:
                # Comfy path
                if isinstance(module, BasicTransformer) and hasattr(module, "attn2"):
                    try:
                        block_id = int(parts[1])
                    except Exception:
                        block_id = 0
                    patch = tpg_replace(scale, sigma_start, sigma_end, rescale, rescale_mode)
                    m.set_model_attn2_replace(patch, block_name, block_id, None)

        return (m,)
