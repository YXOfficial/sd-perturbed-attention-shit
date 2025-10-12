
# tpg_nodes.py — reForge TPG: DIRECT attn2 replacement (1-pass), no post-CFG required.
# This is a fallback to guarantee visible effect even if sampler post-CFG isn't invoked.

BACKEND = "reForge"

from ldm_patched.ldm.modules.attention import optimized_attention, BasicTransformerBlock
from ldm_patched.modules.model_patcher import ModelPatcher

# Robust guidance_utils import (for block parsing + rescale, although rescale here is optional)
import importlib.util as _ilu, sys as _sys, os as _os
try:
    from .guidance_utils import (
        parse_unet_blocks,
        rescale_guidance,
    )  # type: ignore
except Exception:
    try:
        from guidance_utils import (
            parse_unet_blocks,
            rescale_guidance,
        )
    except Exception:
        _root = _os.path.dirname(__file__)
        _gpath = _os.path.join(_root, "guidance_utils.py")
        _spec = _ilu.spec_from_file_location("tpg_guidance_utils_fallback", _gpath)
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["tpg_guidance_utils_fallback"] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
        parse_unet_blocks = _mod.parse_unet_blocks
        rescale_guidance = _mod.rescale_guidance

import torch

def _within_sigma(sigma: float, s0: float, s1: float) -> bool:
    if s0 == float("inf") and s1 < 0:
        return True
    if s1 < 0:
        return sigma >= s0
    return (sigma >= s0) and (sigma <= s1)

def _shuffle_kv(k: torch.Tensor, v: torch.Tensor):
    B, H, T, D = k.shape
    perm = torch.randperm(T, device=k.device)
    return k[:, :, perm, :], v[:, :, perm, :]

def tpg_replace(scale: float, s0: float, s1: float, rescale: float, mode: str):
    # direct attention replacement: return z + scale*(z - z_p)
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
        inner = m.model
        print("[TPG] DIRECT attn2 replace: begin")
        s0 = float("inf") if sigma_start < 0 else sigma_start
        s1 = sigma_end

        single = (unet_block, int(unet_block_id), None)
        try:
            blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else ([single], None)
        except Exception as e:
            print("[TPG] parse_unet_blocks failed, use single:", e)
            blocks, block_names = [single], None

        patched = 0
        # Walk diffusion model modules and install replace by (layer, id, t_idx) matching
        for name, module in inner.diffusion_model.named_modules():
            parts = name.split(".")
            layer = parts[0]
            if block_names and layer not in block_names:
                continue
            if isinstance(module, BasicTransformerBlock) and hasattr(module, "attn2"):
                try:
                    number = int(parts[1])
                except Exception:
                    number = 0
                t_idx = None
                if "transformer_blocks" in parts:
                    pos = parts.index("transformer_blocks") + 1
                    try:
                        t_idx = int(parts[pos])
                    except Exception:
                        t_idx = None
                # Match any listed block
                for (ly, num, tidx) in blocks:
                    if ly == layer and num == number and (tidx is None or t_idx == tidx):
                        m.set_model_attn2_replace(tpg_replace(scale, s0, s1, rescale, rescale_mode), layer, number, t_idx)
                        patched += 1
                        if patched <= 1:
                            print(f"[TPG] DIRECT: set_model_attn2_replace -> {layer}.{number}.{t_idx}")
                        break

        # Aggressive fallback: if nothing patched, try wide ranges
        if patched == 0:
            for layer in ("input", "middle", "output"):
                for number in range(0, 64):
                    m.set_model_attn2_replace(tpg_replace(scale, s0, s1, rescale, rescale_mode), layer, number, None)
                    patched += 1
                    if patched == 1:
                        print(f"[TPG] DIRECT fallback: set_model_attn2_replace -> {layer}.0.None")
                    if number >= 0:
                        break  # only ensure at least one

        print(f"[TPG] DIRECT attn2 replace: patched={patched}")
        return (m,)
