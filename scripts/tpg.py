
try:
    import tpg_forge_utils

    if tpg_forge_utils.BACKEND in {"Forge", "reForge"}:
        import gradio as gr
        from modules import scripts
        from modules.ui_components import InputAccordion

        opTPG = tpg_forge_utils.TokenPerturbationGuidance()

        class TPGScript(scripts.Script):
            def title(self):
                return "Token Perturbation Guidance (TPG)"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, is_img2img):
                with gr.Accordion("TPG", open=False):
                    enabled = gr.Checkbox(label="Enable", value=False)
                    with gr.Row():
                        scale = gr.Slider(label="TPG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)
                    block_list = gr.Text(label="U-Net Blocks (optional)", value="", placeholder="e.g. input 0-3, middle, output 0-1 or d2.2-9")

                self.infotext_fields = (
                    (enabled, lambda p: gr.Checkbox.update(value="tpg_enabled" in p)),
                    (scale, "tpg_scale"),
                    (block_list, lambda p: gr.Text.update(value=p.get("tpg_block_list", ""))),
                    (sigma_start, "tpg_sigma_start"),
                    (sigma_end, "tpg_sigma_end"),
                )

                return (enabled, scale, block_list, sigma_start, sigma_end)

            def process(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    block_list,
                    sigma_start,
                    sigma_end,
                ) = script_args

                if not enabled:
                    return

                # patch unet via ModelPatcher
                unet = p.sd_model.forge_objects.unet
                unet = opTPG.patch(unet, scale, sigma_start, sigma_end, block_list)
                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        tpg_enabled=enabled,
                        tpg_scale=scale,
                        tpg_block_list=block_list,
                        tpg_sigma_start=sigma_start,
                        tpg_sigma_end=sigma_end,
                    )
                )

                return

except ImportError:
    pass
