
from contextlib import suppress
from typing import Optional, Tuple

import torch

from .guidance_utils import parse_unet_blocks, rescale_guidance

BACKEND = None

try:
    from ldm_patched.ldm.modules.attention import optimized_attention
    from ldm_patched.modules.model_patcher import ModelPatcher
    BACKEND = "reForge"
except ImportError:
    from backend.attention import attention_function as optimized_attention
    from backend.patcher.base import ModelPatcher
    BACKEND = "Forge"

# ---- Robust imports for guidance utils (no package parent required) ----
# We try absolute, then file-path based import; else define safe fallbacks.
import importlib.util as _ilu, sys as _sys, os as _os
_guid_ok = False
try:
    # try absolute within extension
    from guidance_utils import parse_unet_blocks as _parse_blocks, rescale_guidance as _rescale_guidance
    _guid_ok = True
except Exception as _e1:
    try:
        _root = _os.path.dirname(__file__)
        _gpath = _os.path.join(_root, "guidance_utils.py")
        if _os.path.exists(_gpath):
            _spec = _ilu.spec_from_file_location("guidance_utils", _gpath)
            _gmod = _ilu.module_from_spec(_spec)
            _sys.modules["guidance_utils"] = _gmod
            _spec.loader.exec_module(_gmod)  # type: ignore[attr-defined]
            _parse_blocks = _gmod.parse_unet_blocks
            _rescale_guidance = _gmod.rescale_guidance
            _guid_ok = True
    except Exception as _e2:
        print(f"[TPG] WARNING: guidance_utils import failed: {_e1} / {_e2}. Using fallbacks.")
        _guid_ok = False

if not _guid_ok:
    def _parse_blocks(model, block_list: str, target: str):
        # Fallback: select all blocks by returning (None, None)
        return None, None
    def __rescale_guidance(out, base, rescale: float, mode: str):
        # Fallback: identity blend (no rescale)
        return out
# ------------------------------------------------------------------------


def within_sigma_range(sigma: float, start: float, end: float) -> bool:
    if start == float("inf") and end < 0:
        return True
    if end < 0:
        return sigma >= start
    return (sigma >= start) and (sigma <= end)


def _shuffle_kv_tokens(k: torch.Tensor, v: torch.Tensor):
    # k,v: (B,H,T,D)
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    return k[:, :, perm, :], v[:, :, perm, :]


def tpg_attn2_replace_wrapper(
    scale: float,
    sigma_start: float,
    sigma_end: float,
    rescale: float,
    rescale_mode: str,
):
    # Signature expected by Forge attention replace: fn(q,k,v,extra_options)->z
    def replace(q, k, v, extra_options):
        # Compute "clean" attention
        z = optimized_attention(q, k, v, extra_options)

        sigma = float(extra_options.get("sigma", 0.0))

        if not within_sigma_range(sigma, sigma_start, sigma_end) or scale == 0.0:
            return z

        # perturbed pass with shuffled KV (keep Q)
        k_p, v_p = _shuffle_kv_tokens(k, v)
        z_p = optimized_attention(q, k_p, v_p, extra_options)

        out = z + scale * (z - z_p)

        if rescale > 0.0:
            out = _rescale_guidance(out, z, rescale, rescale_mode)

        return out

    return replace


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
        inner = m.model

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        blocks, block_names = _parse_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        if BACKEND == "reForge":
            from ldm_patched.ldm.modules.attention import BasicTransformerBlock, CrossAttention
        else:
            from backend.nn.unet import BasicTransformer

        for name, module in inner.diffusion_model.named_modules():
            parts = name.split(".")
            block_name = parts[0]
            if block_names and block_name not in block_names:
                continue

            if BACKEND == "reForge":
                is_attn2 = hasattr(module, "attn2")
                if isinstance(module, BasicTransformerBlock) and is_attn2:
                    block_id = 0
                    with suppress(Exception):
                        block_id = int(parts[1])
                    t_idx = None
                    if "transformer_blocks" in parts:
                        t_pos = parts.index("transformer_blocks") + 1
                        with suppress(Exception):
                            t_idx = int(parts[t_pos])

                    if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                        patch = tpg_attn2_replace_wrapper(scale, sigma_start, sigma_end, rescale, rescale_mode)
                        m.set_model_attn2_replace(patch, block_name, block_id, t_idx)
            else:
                if isinstance(module, BasicTransformer) and getattr(module, "attn2", None) is not None:
                    block_id = 0
                    with suppress(Exception):
                        block_id = int(parts[1])
                    patch = tpg_attn2_replace_wrapper(scale, sigma_start, sigma_end, rescale, rescale_mode)
                    m.set_model_attn2_replace(patch, block_name, block_id, None)

        return m
