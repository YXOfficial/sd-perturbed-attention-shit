# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) pre-CFG hook — runs ONCE at first sigma

import gradio as gr
from modules import scripts, shared

# ✅ Dùng helper chính thức của Forge (đúng chữ ký, tránh lỗi .forward)
from ldm_patched.modules.samplers import calc_cond_uncond_batch

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    import torch
    state = {"done": False}

    def _hook(model, cond, uncond, x, timestep, model_options):
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        x0 = x.detach().clone()

        # thử nhánh autograd trước; nếu backend khóa grad, chuyển sang nudge
        use_grad = True
        try:
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                opt = torch.optim.SGD([xg], lr=float(step_size))

                for _ in range(int(iters)):
                    opt.zero_grad(set_to_none=True)

                    cond_pred, uncond_pred = calc_cond_uncond_batch(
                        model=model,
                        cond=cond,
                        uncond=uncond,
                        x_in=xg,
                        timestep=timestep,
                        model_options=model_options
                    )

                    # nếu không có grad thì nhảy sang fallback
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
            # ---- FALLBACK: vector nudge, không dùng autograd ----
            use_grad = False
            xg = x.detach().clone()
            for _ in range(int(iters)):
                # không cần mở grad; chỉ cập nhật trực tiếp
                cond_pred, uncond_pred = calc_cond_uncond_batch(
                    model=model,
                    cond=cond,
                    uncond=uncond,
                    x_in=xg,
                    timestep=timestep,
                    model_options=model_options
                )
                diff = (cond_pred - uncond_pred)

                # chuẩn hóa theo L2 để bước ổn định
                with torch.no_grad():
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1,1,1,1)
                    n = n + 1e-8
                    step = float(step_size) * (diff / n)
                    xg = xg + step

                    if rho_clip and rho_clip > 0:
                        xg.clamp_(-float(rho_clip), float(rho_clip))
                    if gamma_scale and gamma_scale > 0:
                        xg = x0 + float(gamma_scale) * (xg - x0)

            x_adj = xg.detach()

        state["done"] = True
        mode = "autograd" if use_grad else "nudge"
        # thống kê nhanh để bạn thấy tác động
        with torch.no_grad():
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale}, ||Δx||_rms={rms:.5f})")

        return model, cond, uncond, x_adj, timestep, model_options

    return _hook


class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise (x_T) pre-CFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=True)
            iters   = gr.Slider(label="Iters",      minimum=1,    maximum=200, step=1,     value=20)
            step    = gr.Slider(label="Step size",  minimum=0.001,maximum=0.5,  step=0.001, value=0.05)
            rho     = gr.Slider(label="Rho clip",   minimum=0,    maximum=200,  step=1,     value=50)
            gamma   = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
        return [enabled, iters, step, rho, gamma]

    def process(self, p, enabled, iters, step, rho, gamma):
        if not enabled:
            print("[InitNoise] disabled via UI")
            return

        try:
            # ĐÚNG ĐƯỜNG DẪN TRONG Test-ReForge (forge_loader đã set forge_objects)
            unet = shared.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access shared.sd_model.forge_objects.unet:", e)
            return

        hook = make_init_noise_pre_cfg_hook(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        try:
            # API CHUẨN: modules_forge/unet_patcher.py
            unet.add_sampler_pre_cfg_function(hook, ensure_uniqueness=True)
            print(f"[InitNoise] hook installed on {type(unet).__name__}")
        except Exception as e:
            print("[InitNoise] failed to install hook:", e)
