
# TPG WebUI Script (Test-ReForge) — unconditional registration
# If utils import fails, UI still shows and logs an error on apply.

import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion

opTPG = None
_backend = None
_err = None


# --- bootstrap import: load tpg_nodes by file path (no package needed) ---
import os as _os, sys as _sys, importlib.util as _ilu
_ext_root = _os.path.dirname(_os.path.dirname(__file__))
_utils_path = _os.path.join(_ext_root, "tpg_nodes.py")
if _os.path.exists(_utils_path):
    _spec = _ilu.spec_from_file_location("tpg_nodes", _utils_path)
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["tpg_nodes"] = _mod
    try:
        _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    except Exception as _e:
        print(f"[TPG] ERROR executing tpg_nodes.py: {_e}")
else:
    print(f"[TPG] ERROR: tpg_nodes.py not found at {_utils_path}")
# ------------------------------------------------------------------------------

try:
    import tpg_nodes_webui as tpg_nodes_reforge as tpg_nodes
    opTPG = tpg_nodes.TokenPerturbationGuidance()
    _backend = getattr(tpg_nodes, "BACKEND", None)
    print(f"[TPG] Loaded tpg_nodes, BACKEND={_backend}")
except Exception as e:
    _err = e
    print(f"[TPG] ERROR loading tpg_nodes: {e}")

class TPGScript(scripts.Script):
    def title(self):
        return "Token Perturbation Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("TPG", open=False):
            enabled = gr.Checkbox(label="Enable", value=False)
            with gr.Row():
                scale = gr.Slider(label="TPG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
            with gr.Row():
                rescale_tpg = gr.Slider(label="Rescale TPG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
            with gr.Row():
                hr_mode = gr.Dropdown(
                    choices=["Both", "HRFix Off", "HRFix Only"],
                    value="Both",
                    label="When to apply (HR)",
                    info="Control when TPG is active during generation",
                )

            with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                hr_scale = gr.Slider(label="TPG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                with gr.Row():
                    hr_rescale_tpg = gr.Slider(label="Rescale TPG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                    hr_rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")

            with gr.Row():
                block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                block_list = gr.Text(label="U-Net Blocks (optional)", value="", placeholder="e.g. input 0-3, middle, output 0-1 or d2.2-9")

            with gr.Row():
                sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

        self.infotext_fields = (
            (enabled, lambda p: gr.Checkbox.update(value="tpg_enabled" in p)),
            (scale, "tpg_scale"),
            (rescale_tpg, "tpg_rescale"),
            (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("tpg_rescale_mode", "full"))),
            (adaptive_scale, "tpg_adaptive_scale"),
            (hr_mode, "tpg_hr_mode"),
            (hr_override, lambda p: gr.Checkbox.update(value="tpg_hr_override" in p)),
            (hr_cfg, "tpg_hr_cfg"),
            (hr_scale, "tpg_hr_scale"),
            (hr_rescale_tpg, "tpg_hr_rescale"),
            (hr_rescale_mode, lambda p: gr.Dropdown.update(value=p.get("tpg_hr_rescale_mode", "full"))),
            (block, lambda p: gr.Dropdown.update(value=p.get("tpg_block", "middle"))),
            (block_id, "tpg_block_id"),
            (block_list, lambda p: gr.Text.update(value=p.get("tpg_block_list", ""))),
            (sigma_start, "tpg_sigma_start"),
            (sigma_end, "tpg_sigma_end"),
        )

        return (
            enabled,
            scale,
            rescale_tpg,
            rescale_mode,
            adaptive_scale,
            hr_mode,
            block,
            block_id,
            block_list,
            hr_override,
            hr_cfg,
            hr_scale,
            hr_rescale_tpg,
            hr_rescale_mode,
            sigma_start,
            sigma_end,
        )

    def process(self, p, *script_args, **kwargs):
        (
            enabled,
            scale,
            rescale_tpg,
            rescale_mode,
            adaptive_scale,
            hr_mode,
            block,
            block_id,
            block_list,
            hr_override,
            hr_cfg,
            hr_scale,
            hr_rescale_tpg,
            hr_rescale_mode,
            sigma_start,
            sigma_end,
        ) = script_args

        if not enabled:
            return

        if opTPG is None:
            print("[TPG] UI active but tpg_nodes failed to import; no patch applied.", _err)
            return

        unet = p.sd_model.forge_objects.unet
        hr_enabled = getattr(p, "enable_hr", False)
        is_hr_pass = getattr(p, "is_hr_pass", False)

        # hr_mode rule
        if hr_mode == "HRFix Off" and is_hr_pass:
            return
        elif hr_mode == "HRFix Only" and not is_hr_pass:
            return

        # Apply override during HR pass
        if hr_enabled and is_hr_pass and hr_override:
            p.cfg_scale_before_hr = p.cfg_scale
            p.cfg_scale = hr_cfg
            unet = opTPG.patch(unet, hr_scale, sigma_start, sigma_end, hr_rescale_tpg, hr_rescale_mode, block_list)
            print(f"[TPG] Patched UNet (HR override). Scale={hr_scale}, block_list='{block_list}'")
        else:
            unet = opTPG.patch(unet, scale, sigma_start, sigma_end, rescale_tpg, rescale_mode, block_list)
            print(f"[TPG] Patched UNet. Scale={scale}, block_list='{block_list}'")

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(
            dict(
                tpg_enabled=enabled,
                tpg_scale=scale,
                tpg_rescale=rescale_tpg,
                tpg_rescale_mode=rescale_mode,
                tpg_adaptive_scale=adaptive_scale,
                tpg_hr_mode=hr_mode,
                tpg_block=block,
                tpg_block_id=block_id,
                tpg_block_list=block_list,
                tpg_sigma_start=sigma_start,
                tpg_sigma_end=sigma_end,
            )
        )
        return
