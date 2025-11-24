try:
    import asag_nodes

    if asag_nodes.BACKEND in {"Forge", "reForge"}:
        import gradio as gr

        from modules import scripts
        from modules.ui_components import InputAccordion

        opASAG = asag_nodes.ASAGGuidance()

        class ASAGScript(scripts.Script):
            def title(self):
                return "ASAG (Adversarial Sinkhorn Attention Guidance)"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="ASAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=1.5)
                    sinkhorn_iters = gr.Slider(
                        label="Sinkhorn Iterations", minimum=1, maximum=8, step=1, value=2
                    )
                    with gr.Row():
                        rescale_asag = gr.Slider(label="Rescale ASAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")

                    hr_mode = gr.Radio(
                        show_label=False,
                        label="Hires Fix Mode",
                        choices=["Both", "HRFix Off", "HRFix Only"],
                        value="Both",
                        info="Control when ASAG is active during generation",
                    )

                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="ASAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=1.5)
                        hr_sinkhorn_iters = gr.Slider(
                            label="Sinkhorn Iterations", minimum=1, maximum=8, step=1, value=2
                        )
                        with gr.Row():
                            hr_rescale_asag = gr.Slider(
                                label="Rescale ASAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0
                            )
                            hr_rescale_mode = gr.Dropdown(
                                choices=["full", "partial", "snf"], value="full", label="Rescale Mode"
                            )
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                        block_list = gr.Text(label="U-Net Block List")
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="asag_enabled" in p)),
                        (scale, "asag_scale"),
                        (sinkhorn_iters, "asag_sinkhorn_iters"),
                        (rescale_asag, "asag_rescale"),
                        (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("asag_rescale_mode", "full"))),
                        (hr_mode, "asag_hr_mode"),
                        (hr_override, lambda p: gr.Checkbox.update(value="asag_hr_override" in p)),
                        (hr_cfg, "asag_hr_cfg"),
                        (hr_scale, "asag_hr_scale"),
                        (hr_sinkhorn_iters, "asag_hr_sinkhorn_iters"),
                        (hr_rescale_asag, "asag_hr_rescale"),
                        (hr_rescale_mode, lambda p: gr.Dropdown.update(value=p.get("asag_hr_rescale_mode", "full"))),
                        (block, lambda p: gr.Dropdown.update(value=p.get("asag_block", "middle"))),
                        (block_id, "asag_block_id"),
                        (block_list, lambda p: gr.Text.update(value=p.get("asag_block_list", ""))),
                        (sigma_start, "asag_sigma_start"),
                        (sigma_end, "asag_sigma_end"),
                    )

                return (
                    enabled,
                    scale,
                    sinkhorn_iters,
                    rescale_asag,
                    rescale_mode,
                    hr_mode,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_sinkhorn_iters,
                    hr_rescale_asag,
                    hr_rescale_mode,
                    sigma_start,
                    sigma_end,
                )

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    sinkhorn_iters,
                    rescale_asag,
                    rescale_mode,
                    hr_mode,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_sinkhorn_iters,
                    hr_rescale_asag,
                    hr_rescale_mode,
                    sigma_start,
                    sigma_end,
                ) = script_args

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
                    unet = opASAG.patch(
                        unet,
                        hr_scale,
                        hr_sinkhorn_iters,
                        block,
                        block_id,
                        sigma_start,
                        sigma_end,
                        hr_rescale_asag,
                        hr_rescale_mode,
                        block_list,
                    )[0]
                else:
                    unet = opASAG.patch(
                        unet,
                        scale,
                        sinkhorn_iters,
                        block,
                        block_id,
                        sigma_start,
                        sigma_end,
                        rescale_asag,
                        rescale_mode,
                        block_list,
                    )[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        asag_enabled=enabled,
                        asag_scale=scale,
                        asag_sinkhorn_iters=sinkhorn_iters,
                        asag_rescale=rescale_asag,
                        asag_rescale_mode=rescale_mode,
                        asag_block=block,
                        asag_block_id=block_id,
                        asag_block_list=block_list,
                    )
                )

                if hr_mode != "Both":
                    p.extra_generation_params["asag_hr_mode"] = hr_mode
                if hr_enabled:
                    p.extra_generation_params["asag_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                asag_hr_cfg=hr_cfg,
                                asag_hr_scale=hr_scale,
                                asag_hr_sinkhorn_iters=hr_sinkhorn_iters,
                                asag_hr_rescale=hr_rescale_asag,
                                asag_hr_rescale_mode=hr_rescale_mode,
                            )
                        )
                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            asag_sigma_start=sigma_start,
                            asag_sigma_end=sigma_end,
                        )
                    )

                return

            def post_sample(self, p, ps, *script_args):
                (
                    enabled,
                    scale,
                    sinkhorn_iters,
                    rescale_asag,
                    rescale_mode,
                    hr_mode,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_sinkhorn_iters,
                    hr_rescale_asag,
                    hr_rescale_mode,
                    sigma_start,
                    sigma_end,
                ) = script_args

                if not enabled:
                    return

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and hr_override:
                    p.cfg_scale = p.cfg_scale_before_hr

                return

except ImportError:
    pass
