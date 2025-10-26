# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — pre-CFG once per Generate (original formula: minimize ||eps_text - eps_uncond||_2)
#
# Pre-CFG call site in reForge:
#   /content/Test-reForge/Test-reForge-main/ldm_patched/modules/samplers.py  lines ~288-289:
#       for fn in model_options.get("sampler_pre_cfg_function", []):
#           model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)
#
# model_options source before sampling:
#   /content/Test-reForge/Test-reForge-main/modules/sd_samplers_cfg_denoiser.py  lines ~200-205
#       model_options = self.inner_model.inner_model.forge_objects.unet.model_options
#
# UNet cloning (why we install just-before-sampling):
#   /content/Test-reForge/Test-reForge-main/extensions-builtin/mahiro_reforge/scripts/mahiro_cfg_script.py  lines ~46-80
#       unet = p.sd_model.forge_objects.unet.clone()
#       p.sd_model.forge_objects.unet = unet
#
# Original algorithm reference (you uploaded):
#   /mnt/data/init_noise_orig/init_noise_diffusion_memorization-main/local_sd_pipeline.py
#   Lines 318–354: AdamW optimize latents at first step; Lines 338–343: loss = ||eps_text - eps_uncond||_2 .mean()

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # safe helper in reForge


def make_init_noise_pre_cfg_hook(iters=20, lr=0.05):
    """
    pre-CFG hook signature:
      fn(model, cond, uncond, x, timestep, model_options) -> (model, cond, uncond, x', timestep, model_options)

    Implements *original* formula:
        minimize  mean(|| eps_text - eps_uncond ||_2)
    using AdamW on latents x (first call only via closure state['done']).
    """
    import torch
    state = {"done": False}

    def _hook(model, cond, uncond, x, timestep, model_options):
        # Run exactly once at the first pre-CFG call for this Generate
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        # Clone current latents and optimize with AdamW (as in original code)
        xg = x.detach().clone().requires_grad_(True)
        opt = torch.optim.AdamW([xg], lr=float(lr))

        try:
            for _ in range(int(iters)):
                opt.zero_grad(set_to_none=True)

                # In original: they call unet on scaled input via scheduler.scale_model_input(...)
                # In reForge pre-CFG, sampler wrapper handles scaling before UNet; we call via helper:
                eps_text, eps_uncond = calc_cond_uncond_batch(
                    model=model,
                    cond=cond,
                    uncond=uncond,
                    x_in=xg,
                    timestep=timestep,
                    model_options=model_options,
                )

                # Original objective (local_sd_pipeline.py L338–343):
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred_text = noise_pred_text - noise_pred_uncond
                # loss = torch.norm(noise_pred_text, p=2).mean()
                diff = (eps_text - eps_uncond)

                # Reduce over all non-batch dims to match .mean() over batch in original
                if diff.ndim >= 2:
                    # norm per-sample then mean over batch
                    loss = diff.flatten(1).norm(p=2, dim=1).mean()
                else:
                    loss = diff.norm(p=2)

                loss.backward()
                opt.step()

            x_adj = xg.detach()

            with torch.no_grad():
                rms = ((x_adj - x).pow(2).mean()).sqrt().item()
            print(f"[InitNoise] adjusted x_T (orig-min, iters={iters}, lr={lr}, ||Δx||_rms={rms:.5f})")

        except Exception as e:
            # If backend forbids grad here, keep x unchanged (original code has no 'nudge' fallback)
            print(f"[InitNoise] skipped (no-grad or error: {e})")
            x_adj = x

        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options

    _hook.__init_noise_tag = True  # tag for cleanup
    return _hook


class ScriptInitNoise(scripts.Script):
    def title(self):
        # Change title to break any cached ON state; default OFF should apply on fresh UI
        return "[Forge/reForge] Init Noise (x_T) — original formula (pre-CFG once)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment (original formula)", value=False)
            iters   = gr.Slider(label="Iters", minimum=1, maximum=200, step=1, value=20)
            lr      = gr.Slider(label="AdamW LR", minimum=1e-4, maximum=0.5, step=1e-4, value=0.05)
        return [enabled, iters, lr]

    # Install hook *after* other scripts have cloned/assigned UNet for this run
    def process_before_every_sampling(self, p, enabled, iters, lr, **kwargs):
        if not enabled:
            return

        try:
            unet = p.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access unet:", e)
            return

        # Clean any previous InitNoise hooks on the current (cloned) UNet
        mo = dict(unet.model_options or {})
        pre = [fn for fn in list(mo.get("sampler_pre_cfg_function", [])) if not getattr(fn, "__init_noise_tag", False)]
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo

        # Install fresh hook (new closure -> state['done']=False for this Generate)
        hook = make_init_noise_pre_cfg_hook(int(iters), float(lr))
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre.insert(0, hook)  # run first
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo
        print("[InitNoise] pre-CFG hook installed at process_before_every_sampling() (original formula)")

        # Show params in infotext “Parameters” (like NAG/PAG)
        try:
            if hasattr(p, "extra_generation_params") and isinstance(p.extra_generation_params, dict):
                p.extra_generation_params["InitNoise (orig) iters"] = int(iters)
                p.extra_generation_params["InitNoise (orig) AdamW LR"] = float(lr)
        except Exception:
            pass
