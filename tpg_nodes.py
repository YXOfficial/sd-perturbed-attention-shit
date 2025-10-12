
# tpg_nodes.py — reForge-only Token Perturbation Guidance using post-CFG two-pass (like PAG)
# This avoids any comfy/backend imports and computes guidance as (cond - perturbed_cond)*scale.

BACKEND = "reForge"

from ldm_patched.ldm.modules.attention import optimized_attention, BasicTransformerBlock
from ldm_patched.modules.model_patcher import ModelPatcher, set_model_options_patch_replace, set_model_options_post_cfg_function
from ldm_patched.modules.samplers import calc_cond_uncond_batch

# Robust utils import (relative or by path)
import importlib.util as _ilu, sys as _sys, os as _os
try:
    from .guidance_utils import parse_unet_blocks, rescale_guidance, snf_guidance
except Exception:
    _root = _os.path.dirname(__file__)
    _g = _os.path.join(_root, "guidance_utils.py")
    _spec = _ilu.spec_from_file_location("tpg_guidance_utils_fallback", _g)
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["tpg_guidance_utils_fallback"] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    parse_unet_blocks = _mod.parse_unet_blocks
    rescale_guidance = _mod.rescale_guidance
    snf_guidance = getattr(_mod, "snf_guidance", None)

import torch

def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)

def _shuffle_kv(k: torch.Tensor, v: torch.Tensor):
    # k, v: (B, H, T, D)
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    return k[:, :, perm, :], v[:, :, perm, :]

def tpg_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options, mask=None):
    # Baseline cross-attention
    z = optimized_attention(q, k, v, extra_options)
    # Shuffle K/V tokens to break text alignment
    k_p, v_p = _shuffle_kv(k, v)
    z_p = optimized_attention(q, k_p, v_p, extra_options)
    # Return perturbed result (we'll take difference outside)
    return z_p

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
        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        # Parse attn2 blocks; if empty, apply to all attn2 (parse returns None,None)
        blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        def post_cfg_function(args):
            """CFG + TPG"""
            model_ = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            # Gate by sigma window and zero scale
            if scale == 0.0 or not _within_sigma(sigma[0].item() if torch.is_tensor(sigma) else float(sigma[0]), sigma_start, sigma_end):
                return cfg_result

            # Install attn2 replace for selected blocks
            if blocks:
                for (layer, number, index) in blocks:
                    model_options = set_model_options_patch_replace(model_options, tpg_attention, "attn2", layer, number, index)
            else:
                # No explicit blocks -> patch all attn2 across UNet parts that exist; follow PAG pattern for attn1
                # We target typical groups; if a block doesn't exist, set_model_options silently ignores.
                for layer in ("input", "middle", "output"):
                    for number in range(0, 100):
                        model_options = set_model_options_patch_replace(model_options, tpg_attention, "attn2", layer, number, None)

            # Recompute conditional prediction with perturbed cross-attn
            (tpg_cond_pred, _) = calc_cond_uncond_batch(model_, cond, None, x, sigma, model_options)

            # Guidance = (cond - perturbed_cond) * scale
            guidance = (cond_pred - tpg_cond_pred) * scale

            # SNF special case: mimic PAG behaviour if requested
            if rescale_mode == "snf" and snf_guidance is not None:
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, guidance)
                return cfg_result + guidance

            # Default rescale
            return cfg_result + rescale_guidance(guidance, cond_pred, cfg_result, rescale, rescale_mode)

        # Register post-CFG
        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)
