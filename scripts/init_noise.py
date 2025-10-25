# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — run exactly once per job via conditioning_modifiers

import os, time, hashlib
import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # Forge helper

# --- Session-level guards ---
# Enforce one-time "default OFF" on first run after WebUI boot, even if WebUI restores old state.
__SESSION_FORCE_FIRST_RUN_DISABLED__ = True
# Prevent duplicate installs if Script.process() is called multiple times for the same action.
__LAST_INSTALL_MARK__ = None

def _make_run_uid():
    s = f"{time.time()}:{os.getpid()}"
    return hashlib.md5(s.encode()).hexdigest()[:10]

def make_init_noise_modifier(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    """
    Modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    - Runs exactly ONCE per generate (per modifier instance).
    - Closure 'state' is reliable across steps in the same job; we do not rely on model_options persistence.
    """
    import torch
    state = {"done": False}  # remember for this job only (per modifier instance)

    def _modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        # Already adjusted for this job? skip.
        if state["done"]:
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        # Perform initial-noise adjustment (autograd if available; else nudge)
        x0 = x.detach().clone()
        use_grad = True
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.SGD([xg], lr=float(step_size))
                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)
                    cond_pred, uncond_pred = calc_cond_uncond_batch(
                        model=model, cond=cond, uncond=uncond,
                        x_in=xg, timestep=timestep, model_options=model_options
                    )
                    if not (cond_pred.requires_grad or uncond_pred.requires_grad):
                        raise RuntimeError("no-grad backend")
                    diff = (cond_pred - uncond_pred)
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
                cond_pred, uncond_pred = calc_cond_uncond_batch(
                    model=model, cond=cond, uncond=uncond,
                    x_in=xg, timestep=timestep, model_options=model_options
                )
                diff = (cond_pred - uncond_pred)
                with torch.no_grad():
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
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
        return model, x_adj, timestep, uncond, cond, cond_scale, model_options, seed

    _modifier.__init_noise_tag = True  # so we can cleanly remove old modifiers
    return _modifier


class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise (x_T)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Default ENABLED = False (we will also force the first-run to be disabled in process()).
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=False)
            iters   = gr.Slider(label="Iters",      minimum=1,    maximum=200, step=1,     value=20)
            step    = gr.Slider(label="Step size",  minimum=0.001,maximum=0.5,  step=0.001, value=0.05)
            rho     = gr.Slider(label="Rho clip",   minimum=0,    maximum=200,  step=1,     value=50)
            gamma   = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
        return [enabled, iters, step, rho, gamma]

    def process(self, p, enabled, iters, step, rho, gamma):
        global __SESSION_FORCE_FIRST_RUN_DISABLED__, __LAST_INSTALL_MARK__

        # Access proper UnetPatcher on Forge/reForge
        try:
            unet = shared.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access shared.sd_model.forge_objects.unet:", e)
            return

        # Snapshot + cleanup old modifiers
        mo = dict(unet.model_options or {})
        to = dict(mo.get("transformer_options", {}))
        mods = list(mo.get("conditioning_modifiers", []))
        mods = [m for m in mods if not getattr(m, "__init_noise_tag", False)]

        # Enforce first-run default OFF (once per WebUI boot), regardless of restored UI state.
        if __SESSION_FORCE_FIRST_RUN_DISABLED__:
            enabled = False
            __SESSION_FORCE_FIRST_RUN_DISABLED__ = False
            print("[InitNoise] first-run hard default OFF (session)")

        # If disabled -> clean flags and bail
        if not enabled:
            to.pop("init_noise_run_uid", None)
            mo["transformer_options"] = to
            mo["conditioning_modifiers"] = mods
            unet.model_options = mo
            __LAST_INSTALL_MARK__ = None
            print("[InitNoise] disabled via UI (cleaned)")
            return

        # Build a run mark to avoid duplicate installs for the same trigger
        run_mark = f"{time.time():.3f}:{os.getpid()}"
        if __LAST_INSTALL_MARK__ is not None:
            # If the last install happened within a very short window, skip duplicate install
            try:
                last_t = float(__LAST_INSTALL_MARK__.split(":")[0])
                now_t = time.time()
                if (now_t - last_t) < 0.25:
                    # Keep existing modifier; do not re-install mid-flight
                    print("[InitNoise] duplicate install call suppressed")
                    return
            except Exception:
                pass
        __LAST_INSTALL_MARK__ = run_mark

        # Enabled: install fresh modifier (new closure -> state reset)
        uid = _make_run_uid()
        to["init_noise_run_uid"] = uid
        mo["transformer_options"] = to

        modifier = make_init_noise_modifier(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        # Prepend so it runs before other conditioning modifiers/patches
        mods.insert(0, modifier)
        mo["conditioning_modifiers"] = mods

        unet.model_options = mo
        print(f"[InitNoise] modifier installed on {type(unet).__name__} (uid={uid})")
