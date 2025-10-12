
# tpg_nodes.py — reForge-only (Test-ReForge) backend to avoid falling back to 'backend' or 'comfy'
# Loads guidance_utils robustly even if this module isn't treated as a package.

import importlib.util as _ilu
import sys as _sys
import os as _os

BACKEND = "reForge"

# --- reForge imports ---
from ldm_patched.ldm.modules.attention import optimized_attention
from ldm_patched.modules.model_patcher import ModelPatcher

# --- guidance_utils robust import ---
try:
    # Prefer relative import when package context is present
    from .guidance_utils import parse_unet_blocks, rescale_guidance  # type: ignore
except Exception:
    # Fallback: load by absolute file path next to this file
    _root = _os.path.dirname(__file__)
    _gpath = _os.path.join(_root, "guidance_utils.py")
    _spec = _ilu.spec_from_file_location("tpg_guidance_utils_fallback", _gpath)
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["tpg_guidance_utils_fallback"] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    parse_unet_blocks = _mod.parse_unet_blocks
    rescale_guidance = _mod.rescale_guidance

def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)

def _shuffle_kv(k, v):
    import torch
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    return k[:, :, perm, :], v[:, :, perm, :]

def tpg_replace(scale: float, s0: float, s1: float, rescale: float, mode: str):
    # signature: fn(q,k,v,extra_options)->z
    def _fn(q, k, v, extra_options):
        z = optimized_attention(q, k, v, extra_options)
        if scale == 0.0:
            return z
        sigma = float(extra_options.get("sigma", 0.0))
        if not _within_sigma(sigma, s0, s1):
            return z
        k_p, v_p = _shuffle_kv(k, v)
        z_p = optimized_attention(q, k_p, v_p, extra_options)
        out = z + scale * (z - z_p)
        if rescale > 0.0:
            out = rescale_guidance(out, z, rescale, mode)
        return out
    return _fn

class TokenPerturbationGuidance:
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

        # Parse blocks (optional filter); tolerate parse failures
        blocks = block_names = None
        try:
            if unet_block_list:
                blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2")
        except Exception:
            blocks = block_names = None

        from ldm_patched.ldm.modules.attention import BasicTransformerBlock
        for name, module in inner_model.diffusion_model.named_modules():
            parts = name.split(".")
            block_name = parts[0]
            if block_names and block_name not in block_names:
                continue
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

                if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                    m.set_model_attn2_replace(
                        tpg_replace(scale, sigma_start, sigma_end, rescale, rescale_mode),
                        block_name, block_id, t_idx
                    )

        return (m,)
