# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — run exactly once per job via conditioning_modifiers

import os, time, hashlib
import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # Forge helper

def _make_run_uid():
    s = f"{time.time()}:{os.getpid()}"
    return hashlib.md5(s.encode()).hexdigest()[:10]

def make_init_noise_modifier(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    """
    Modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    Chạy đúng 1 lần / job (theo run_uid). Dùng closure để nhớ đã chạy, tránh lặp mỗi step.
    """
    import torch
    state = {"applied_uid": None}  # nhớ UID đã áp dụng trong lượt generate hiện tại

    def _modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        # 1) Lấy UID do process() đặt
        to = dict(model_options.get("transformer_options", {}))
        uid = to.get("init_noise_run_uid", None)
        if uid is None:
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        # 2) Nếu đã chạy cho UID này rồi → bỏ qua (không lặp ở các step sau)
        if state["applied_uid"] == uid:
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        # 3) Đánh dấu đã chạy cho UID này
        state["applied_uid"] = uid

        # 4) Điều chỉnh x_T (autograd nếu có; nếu backend no_grad thì dùng nudge)
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

        # Không cần ghi “applied” vào model_options (bị reload mỗi step); closure đã đảm bảo 1 lần/UID
        return model, x_adj, timestep, uncond, cond, cond_scale, model_options, seed

    # Cờ để dọn modifier cũ khi bật/tắt
    _modifier.__init_noise_tag = True
    return _modifier


class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise (x_T)"

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
        # Lấy UNet patcher đúng chuẩn Forge/reForge
        try:
            unet = shared.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access shared.sd_model.forge_objects.unet:", e)
            return

        mo = dict(unet.model_options or {})
        to = dict(mo.get("transformer_options", {}))
        mods = list(mo.get("conditioning_modifiers", []))

        # Luôn dọn mọi modifier cũ của InitNoise trước
        mods = [m for m in mods if not getattr(m, "__init_noise_tag", False)]

        if not enabled:
            # Xóa luôn các flag để ngăn chạy ngoài ý muốn
            to.pop("init_noise_run_uid", None)
            to.pop("init_noise_applied_uid", None)
            mo["transformer_options"] = to
            mo["conditioning_modifiers"] = mods
            unet.model_options = mo
            print("[InitNoise] disabled via UI (cleaned)")
            return

        # Enabled: tạo run UID cho job hiện tại
        uid = _make_run_uid()
        to["init_noise_run_uid"] = uid
        to.pop("init_noise_applied_uid", None)  # không dùng nữa, tránh sót
        mo["transformer_options"] = to

        # Thêm modifier mới *lên đầu* để chạy trước các patch khác
        modifier = make_init_noise_modifier(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        mods.insert(0, modifier)
        mo["conditioning_modifiers"] = mods

        # Ghi trả lại vào UNet
        unet.model_options = mo
        print(f"[InitNoise] modifier installed on {type(unet).__name__} (uid={uid})")
