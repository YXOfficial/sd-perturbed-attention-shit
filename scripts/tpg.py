# scripts/tpg.py — pattern giống hệt PAG/SEG

try:
    import tpg_nodes

    if tpg_nodes.BACKEND in {"Forge", "reForge"}:
        import gradio as gr
        from modules import scripts
        from modules.ui_components import InputAccordion

        opTPG = tpg_nodes.TokenPerturbationGuidance()

        class TPGScript(scripts.Script):
            def title(self):
                return "Token Perturbation Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="TPG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                    with gr.Row():
                        rescale_tpg = gr.Slider(label="Rescale TPG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                    hr_mode = gr.Dropdown(choices=["Both", "HRFix Off", "HRFix Only"], value="Both", label="When to apply (HR)")
                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="TPG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                        with gr.Row():
                            hr_rescale_tpg = gr.Slider(label="Rescale TPG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            hr_rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                        block_list = gr.Text(label="U-Net Blocks (optional)", value="", placeholder="e.g. input 0-3, middle, output 0-1")
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

                self.infotext_fields = (
                    (enabled, lambda p: gr.Checkbox.update(value="tpg_enabled" in p)),
                    (scale, "tpg_scale"),
                    (rescale_tpg, "tpg_rescale"),
                    (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("tpg_rescale_mode", "full"))),
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

                return (enabled, scale, rescale_tpg, rescale_mode, hr_mode,
                        block, block_id, block_list,
                        hr_override, hr_cfg, hr_scale, hr_rescale_tpg, hr_rescale_mode,
                        sigma_start, sigma_end)

            def process(self, p, *script_args, **kwargs):
                (enabled, scale, rescale_tpg, rescale_mode, hr_mode,
                 block, block_id, block_list,
                 hr_override, hr_cfg, hr_scale, hr_rescale_tpg, hr_rescale_mode,
                 sigma_start, sigma_end) = script_args

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet
                hr_enabled = getattr(p, "enable_hr", False)
                is_hr_pass = getattr(p, "is_hr_pass", False)

                if hr_mode == "HRFix Off" and is_hr_pass:
                    return
                elif hr_mode == "HRFix Only" and not is_hr_pass:
                    return

                if hr_enabled and is_hr_pass and hr_override:
                    p.cfg_scale_before_hr = p.cfg_scale
                    p.cfg_scale = hr_cfg
                    unet = opTPG.patch(unet, hr_scale, sigma_start, sigma_end, hr_rescale_tpg, hr_rescale_mode, block_list)[0]
                else:
                    unet = opTPG.patch(unet, scale, sigma_start, sigma_end, rescale_tpg, rescale_mode, block_list)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        tpg_enabled=enabled,
                        tpg_scale=scale,
                        tpg_rescale=rescale_tpg,
                        tpg_rescale_mode=rescale_mode,
                        tpg_hr_mode=hr_mode,
                        tpg_block=block,
                        tpg_block_id=block_id,
                        tpg_block_list=block_list,
                        tpg_sigma_start=sigma_start,
                        tpg_sigma_end=sigma_end,
                    )
                )
                return

except ImportError:
    # Nếu import tpg_nodes fail hoàn toàn, bỏ qua script (giống style repo)
    pass
