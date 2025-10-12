from functools import partial

BACKEND = None

try:
    # Comfy (not used in Test-ReForge but keep parity with PAG)
    from comfy.ldm.modules.attention import optimized_attention
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import calc_cond_batch
    from .guidance_utils import (
        parse_unet_blocks,
        rescale_guidance,
        snf_guidance,
    )
    from .guidance_utils import set_model_options_patch_replace
    BACKEND = "ComfyUI"
except ImportError:
    # reForge / Forge imports
    from .guidance_utils import (
        parse_unet_blocks,
        rescale_guidance,
        snf_guidance,
        set_model_options_patch_replace,
    )
    try:
        from ldm_patched.ldm.modules.attention import optimized_attention
        from ldm_patched.modules.model_patcher import ModelPatcher
        from ldm_patched.modules.samplers import calc_cond_uncond_batch
        BACKEND = "reForge"
    except ImportError:
        from backend.attention import attention_function as optimized_attention
        from backend.patcher.base import ModelPatcher
        from backend.sampling.sampling_function import calc_cond_uncond_batch
        BACKEND = "Forge"

import torch

def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)

def _tpg_attention(q, k, v, extra_options, mask=None):
    # base cross-attn
    z = optimized_attention(q, k, v, extra_options)
    # shuffle tokens in K/V
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    z_p = optimized_attention(q, k[:, :, perm, :], v[:, :, perm, :], extra_options)
    return z_p

class TokenPerturbationGuidance:
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
        single_block = (unet_block, int(unet_block_id), None)
        blocks, block_names = (
            parse_unet_blocks(model, unet_block_list, "attn2") if unet_block_list else ([single_block], None)
        )

        def post_cfg_function(args):
            model_ = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            # sigma gating
            s0 = sigma_start
            s1 = sigma_end
            sigma_val = sigma[0] if isinstance(sigma, (list, tuple)) else sigma
            try:
                sigma_val = sigma_val.item() if hasattr(sigma_val, "item") else float(sigma_val)
            except Exception:
                sigma_val = float(sigma_val)

            if scale == 0.0 or not _within_sigma(sigma_val, s0, s1):
                return cfg_result

            # Replace CROSS-attention (attn2) with TPG attention on selected blocks
            for layer, number, index in blocks:
                model_options = set_model_options_patch_replace(
                    model_options, _tpg_attention, "attn2", layer, number, index
                )

            # Recompute conditional with perturbed attn2
            if BACKEND == "ComfyUI":
                (tpg_cond_pred,) = calc_cond_batch(model_, [cond], x, sigma, model_options)
            else:
                (tpg_cond_pred, _) = calc_cond_uncond_batch(model_, cond, None, x, sigma, model_options)

            tpg = (cond_pred - tpg_cond_pred) * scale

            if rescale_mode == "snf" and snf_guidance is not None:
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, tpg)
                return cfg_result + tpg

            return cfg_result + rescale_guidance(tpg, cond_pred, cfg_result, rescale, rescale_mode)

        # Register post-CFG hook
        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")
        return (m,)
