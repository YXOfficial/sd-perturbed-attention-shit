# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — one-shot per Generate, with pre-CFG + fallback modifier
# - Script title changed to reset WebUI saved state (default OFF really takes effect).
# - Installs BOTH:
#     (A) pre-CFG hook: preferred, runs once at first sigma
#     (B) conditioning modifier (fallback): runs once per job via closure
# - Always cleans previous hooks/modifiers tagged by __init_noise_tag to avoid duplicates.

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch

# ---------------------------
# One-shot pre-CFG hook
# ---------------------------
def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    import torch
    state = {"done": False}
    def _hook(model, cond, uncond, x, timestep, model_options):
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options
        x0 = x.detach().clone()
        use_grad = True
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.SGD([xg], lr=float(step_size))
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)
                    c, u = calc_cond_uncond_batch(model, cond, uncond, xg, timestep, model_options)
                    if not (c.requires_grad or u.requires_grad):
                        raise RuntimeError("no-grad backend")
                    diff = (c - u)
                    loss = -(diff.square().mean())
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
                c, u = calc_cond_uncond_batch(model, cond, uncond, xg, timestep, model_options)
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
    _hook.__init_noise_tag = True
    return _hook

# ---------------------------
# Fallback: conditioning modifier (also one-shot via closure)
# ---------------------------
def make_init_noise_modifier(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    import torch
    state = {"done": False}
    def _modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        if state["done"]:
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed
        x0 = x.detach().clone()
        use_grad = True
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.SGD([xg], lr=float(step_size))
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)
                    c, u = calc_cond_uncond_batch(model, cond, uncond, xg, timestep, model_options)
                    if not (c.requires_grad or u.requires_grad):
                        raise RuntimeError("no-grad backend")
                    diff = (c - u)
                    loss = -(diff.square().mean())
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
                c, u = calc_cond_uncond_batch(model, cond, uncond, xg, timestep, model_options)
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
        print(f"[InitNoise] adjusted x_T (fallback-{mode}, iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale}, ||Δx||_rms={rms:.5f})")
        state["done"] = True
        return model, x_adj, timestep, uncond, cond, cond_scale, model_options, seed
    _modifier.__init_noise_tag = True
    return _modifier


class ScriptInitNoise(scripts.Script):
    def title(self):
        # đổi tiêu đề → WebUI không áp giá trị checkbox cũ, trả về mặc định OFF thật sự
        return "[Forge/reForge] Init Noise (x_T) — one-shot (v2)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # default OFF; nếu WebUI từng lưu state theo title cũ, đổi title sẽ reset
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment (default OFF)", value=False)
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

        mo = dict(unet.model_options or {})

        # 1) Clean any previous pre-CFG hooks of this extension
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre = [fn for fn in pre if not getattr(fn, "__init_noise_tag", False)]

        # 2) Clean any previous conditioning modifiers of this extension
        mods = list(mo.get("conditioning_modifiers", []))
        mods = [m for m in mods if not getattr(m, "__init_noise_tag", False)]

        if not enabled:
            mo["sampler_pre_cfg_function"] = pre
            mo["conditioning_modifiers"] = mods
            unet.model_options = mo
            print("[InitNoise] disabled via UI (cleaned)")
            return

        # 3) install fresh one-shot pre-CFG hook (preferred)
        hook = make_init_noise_pre_cfg_hook(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        pre.insert(0, hook)  # run first
        mo["sampler_pre_cfg_function"] = pre

        # 4) install fallback conditioning modifier (will run once if pre-CFG path isn’t used)
        modifier = make_init_noise_modifier(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        mods.insert(0, modifier)
        mo["conditioning_modifiers"] = mods

        unet.model_options = mo
        print("[InitNoise] pre-CFG hook + fallback modifier installed")
