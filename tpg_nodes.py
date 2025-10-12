# tpg_nodes.py — reForge Token Perturbation Guidance (post-CFG two-pass)
# Robustly imports guidance_utils with fallbacks (relative -> absolute -> by path).

BACKEND = "reForge"

# --- reForge imports ---
from ldm_patched.ldm.modules.attention import optimized_attention, BasicTransformerBlock
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.samplers import calc_cond_uncond_batch

# --- guidance_utils robust import ---
import importlib.util as _ilu, sys as _sys, os as _os
try:
    # Prefer relative if package context exists
    from .guidance_utils import (
        parse_unet_blocks,
        rescale_guidance,
        snf_guidance,
        set_model_options_patch_replace,
    )  # type: ignore
except Exception:
    try:
        # Absolute import (same folder on sys.path)
        from guidance_utils import (
            parse_unet_blocks,
            rescale_guidance,
            snf_guidance,
            set_model_options_patch_replace,
        )
    except Exception:
        # Load by file path (same dir as this file)
        _root = _os.path.dirname(__file__)
        _gpath = _os.path.join(_root, "guidance_utils.py")
        _spec = _ilu.spec_from_file_location("tpg_guidance_utils_fallback", _gpath)
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["tpg_guidance_utils_fallback"] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
        parse_unet_blocks = _mod.parse_unet_blocks
        rescale_guidance = _mod.rescale_guidance
        snf_guidance = getattr(_mod, "snf_guidance", None)
        set_model_options_patch_replace = _mod.set_model_options_patch_replace

import torch

def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)

def _tpg_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options, mask=None):
    # Baseline cross-attn result
    z = optimized_attention(q, k, v, extra_options)
    # Perturb K/V token order to break text alignment
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    z_p = optimized_attention(q, k[:, :, perm, :], v[:, :, perm, :], extra_options)
    # Return perturbed result; the outer post-CFG will compute diff vs cond_pred
    return z_p

class TokenPerturbationGuidance:
    def __init__(self):
        pass

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()
        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        # Determine target blocks like PAG/SEG
        single_block = (unet_block, int(unet_block_id), None)
        if unet_block_list:
            blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2")
        else:
            blocks, block_names = [single_block], None

        def post_cfg_function(args):
            model_ = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            model_options = args["model_options"].copy()

            # Gate by sigma window and zero scale
            s0, s1 = sigma_start, sigma_end
            sv = sigma[0] if isinstance(sigma, (list, tuple)) else sigma
            try:
                sv = sv.item() if hasattr(sv, "item") else float(sv)
            except Exception:
                sv = float(sv)
            if scale == 0.0 or not _within_sigma(sv, s0, s1):
                return cfg_result

            # Replace attn2 with TPG variant on selected blocks
            dbg = True
            for layer, number, index in blocks:
                if dbg:
                    print(f"[TPG] patch attn2 -> {layer}.{number}.{index}")
                    dbg = False
                model_options = set_model_options_patch_replace(
                    model_options, _tpg_attention, "attn2", layer, number, index
                )

            # Recompute cond with perturbed attn2
            (tpg_cond_pred, _) = calc_cond_uncond_batch(model_, cond, None, x, sigma, model_options)

            # Guidance = (cond - perturbed_cond) * scale
            guidance = (cond_pred - tpg_cond_pred) * scale

            # SNF mode like PAG
            if rescale_mode == "snf" and snf_guidance is not None:
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, guidance)
                return cfg_result + guidance

            # Default rescale
            return cfg_result + rescale_guidance(guidance, cond_pred, cfg_result, rescale, rescale_mode)

        # Register post-CFG hook
        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")
        return (m,)
