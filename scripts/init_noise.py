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
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        import torch
        x0 = x.detach().clone()

        # --- NHÁNH 1: cố dùng "đúng công thức gốc" nếu grad khả dụng ---
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.AdamW([xg], lr=float(lr))   # lr = AdamW LR từ UI
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)
                    eps_text, eps_uncond = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    # Nếu backend thật sự no-grad, 2 tensor này sẽ không có grad_fn:
                    if not (eps_text.requires_grad or eps_uncond.requires_grad):
                        raise RuntimeError("no-grad backend")

                    diff = (eps_text - eps_uncond)
                    # minimize mean(||diff||_2) ~ đúng local_sd_pipeline.py L338–343
                    if diff.ndim >= 2:
                        loss = diff.flatten(1).norm(p=2, dim=1).mean()
                    else:
                        loss = diff.norm(p=2)

                    loss.backward()
                    opt.step()

                x_adj = xg.detach()
                mode = "orig-min(AdamW)"

        # --- NHÁNH 2: fallback "nudge-min" KHÔNG dùng grad (giảm ||Δε||) ---
        except Exception:
            xg = x.detach().clone()
            for _ in range(int(iters)):
                with torch.no_grad():
                    eps_text, eps_uncond = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    diff = (eps_text - eps_uncond)
                    # hướng giảm chuẩn: x := x - η * (Δε / ||Δε||)
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-8
                    xg = xg - float(lr) * (diff / n)  # dùng cùng "lr" làm step size cho nudge-min

            x_adj = xg.detach()
            mode = "nudge-min(no-grad)"

        with torch.no_grad():
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, lr={lr}, ||Δx||_rms={rms:.5f})")

        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options



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
