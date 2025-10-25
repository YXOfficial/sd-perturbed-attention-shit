# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) pre-CFG hook — runs ONCE at first sigma

import os, time, hashlib
import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # helper chính thức của Forge

def _make_run_uid():
    # UID ngắn: theo thời điểm + PID (đủ phân biệt mỗi lần Generate)
    s = f"{time.time()}:{os.getpid()}"
    return hashlib.md5(s.encode()).hexdigest()[:10]

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    import torch
    state = {"done": False, "uid": None}

    def _hook(model, cond, uncond, x, timestep, model_options):
        # Đọc UID do process() đặt trong transformer_options
        to = model_options.get("transformer_options", {})
        uid = to.get("init_noise_run_uid", None)

        # UID thay đổi => reset state để chạy lại cho job mới
        if uid is not None and uid != state.get("uid"):
            state["uid"] = uid
            state["done"] = False

        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        x0 = x.detach().clone()

        # Thử autograd; nếu backend no_grad thì fallback sang 'nudge'
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

                # Vector nudge (không autograd), chuẩn hoá để bước ổn định
                with torch.no_grad():
                    n = diff.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
                    step = float(step_size) * (diff / n)
                    xg = xg + step

                    if rho_clip and rho_clip > 0:
                        xg.clamp_(-float(rho_clip), float(rho_clip))
                    if gamma_scale and gamma_scale > 0:
                        xg = x0 + float(gamma_scale) * (xg - x0)

            x_adj = xg.detach()

        state["done"] = True
        mode = "autograd" if use_grad else "nudge"
        with torch.no_grad():
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale}, ||Δx||_rms={rms:.5f})")

        return model, cond, uncond, x_adj, timestep, model_options

    # Cờ nhận diện để dọn hook cũ khi bật/tắt
    _hook.__init_noise_tag = True
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

        # Đúng đường dẫn Forge/reForge: forge_loader đã set forge_objects
        try:
            unet = shared.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access shared.sd_model.forge_objects.unet:", e)
            return

        # 1) Đặt run UID vào transformer_options (reset state cho lần Generate mới)
        uid = _make_run_uid()
        mo = dict(unet.model_options or {})
        to = dict(mo.get("transformer_options", {}))
        to["init_noise_run_uid"] = uid
        mo["transformer_options"] = to

        # 2) Dọn mọi pre-CFG hook cũ của InitNoise
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre = [fn for fn in pre if not getattr(fn, "__init_noise_tag", False)]

        # 3) Thêm hook mới *lên đầu* để chạy trước các patch khác
        hook = make_init_noise_pre_cfg_hook(
            iters=int(iters),
            step_size=float(step),
            rho_clip=float(rho),
            gamma_scale=float(gamma)
        )
        pre.insert(0, hook)
        mo["sampler_pre_cfg_function"] = pre

        # Ghi trả lại
        unet.model_options = mo
        print(f"[InitNoise] hook installed on {type(unet).__name__} (uid={uid})")
