# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) pre-CFG (once per Generate) — minimal, robust

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # safe helper from Forge

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    """
    pre-CFG hook signature:
      fn(model, cond, uncond, x, timestep, model_options) -> same tuple
    Runs exactly once at the first pre-CFG call per Generate, via closure state["done"].
    """
    import torch
    state = {"done": False}

    def _hook(model, cond, uncond, x, timestep, model_options):
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        x0 = x.detach().clone()

        # Try autograd; if backend no_grad, fall back to vector nudge (no grad)
        use_grad = True
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.SGD([xg], lr=float(step_size))
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)

                    c, u = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    if not (c.requires_grad or u.requires_grad):
                        raise RuntimeError("no-grad backend")

                    diff = (c - u)
                    loss = -(diff.square().mean())  # maximize ||diff||^2
                    (-loss).backward()
                    opt.step()

                    with torch.no_grad():
                        if rho_clip and rho_clip > 0:
                            xg.clamp_(-float(rho_clip), float(rho_clip))
                        if gamma_scale and gamma_scale > 0:
                            xg.copy_(x0 + float(gamma_scale) * (xg - x0))

                x_adj = xg.detach()

        except Exception:
            use_grad = False
            xg = x.detach().clone()
            for _ in range(int(iters)):
                c, u = calc_cond_uncond_batch(
                    model=model, cond=cond, uncond=uncond,
                    x_in=xg, timestep=timestep, model_options=model_options
                )
                diff = (c - u)
                with torch.no_grad():
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1,1,1,1) + 1e-8
                    xg = xg + float(step_size) * (diff / n)
                    if rho_clip and rho_clip > 0:
                        xg.clamp_(-float(rho_clip), float(rho_clip))
                    if gamma_scale and gamma_scale > 0:
                        xg = x0 + float(gamma_scale) * (xg - x0)
            x_adj = xg.detach()

        with torch.no_grad():
            mode = "autograd" if use_grad else "nudge"
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale}, ||Δx||_rms={rms:.5f})")

        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options

    # tag to recognize & clean previous copies
    _hook.__init_noise_tag = True
    return _hook


class ScriptInitNoise(scripts.Script):
    def title(self):
        # Change title to break saved UI state; checkbox default False will actually apply.
        return "[Forge/reForge] Init Noise (x_T) — pre-CFG once (minimal)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Default OFF; if WebUI previously persisted ON, title change resets it.
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=False)
            iters   = gr.Slider(label="Iters",      minimum=1,    maximum=200, step=1,     value=20)
            step    = gr.Slider(label="Step size",  minimum=0.001,maximum=0.5,  step=0.001, value=0.05)
            rho     = gr.Slider(label="Rho clip",   minimum=0,    maximum=200,  step=1,     value=50)
            gamma   = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
        return [enabled, iters, step, rho, gamma]

    def process(self, p, enabled, iters, step, rho, gamma):
        try:
            unet = shared.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access shared.sd_model.forge_objects.unet:", e)
            return

        # Always clean old copies of THIS hook from model_options (maintenance only).
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre = [fn for fn in pre if not getattr(fn, "__init_noise_tag", False)]
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo  # assign back only for cleanup once

        if not enabled:
            print("[InitNoise] disabled via UI (cleaned)")
            return

        # Create a fresh hook (new closure -> state['done']=False for this Generate)
        hook = make_init_noise_pre_cfg_hook(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )

        # ✅ Register via UnetPatcher API (robust against other extensions reassigning model_options)
        try:
            unet.add_sampler_pre_cfg_function(hook, ensure_uniqueness=True)
            print("[InitNoise] pre-CFG hook installed on UnetPatcher (via API)")
        except Exception as e:
            # Fallback: manual insert to the very front (rarely needed)
            mo = dict(unet.model_options or {})
            pre = list(mo.get("sampler_pre_cfg_function", []))
            pre.insert(0, hook)
            mo["sampler_pre_cfg_function"] = pre
            unet.model_options = mo
            print(f"[InitNoise] pre-CFG hook installed by manual insert (reason: {e})")
