# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — one-shot at first pre-CFG call per Generate

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # safe helper

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    """
    pre-CFG hook signature:
      fn(model, cond, uncond, x, timestep, model_options) -> same tuple
    Chạy đúng 1 lần ở call đầu tiên trong lượt Generate nhờ closure state["done"].
    """
    import torch
    state = {"done": False}

    def _hook(model, cond, uncond, x, timestep, model_options):
        # đã chạy 1 lần trong job này -> bỏ qua
        if state["done"]:
            return model, cond, uncond, x, timestep, model_options

        x0 = x.detach().clone()

        # thử autograd; nếu backend no_grad thì dùng "nudge" (không cần grad)
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
                    # nếu không có grad → rơi sang nudge
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

        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options

    # gắn thẻ để có thể xoá sạch các hook cũ trước khi thêm mới
    _hook.__init_noise_tag = True
    return _hook


class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise (x_T) pre-CFG (once)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # ⚠️ default False trong UI; WebUI có thể nhớ trạng thái cũ của bạn
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

        # lấy snapshot options & dọn mọi pre-CFG hook cũ của InitNoise
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre = [fn for fn in pre if not getattr(fn, "__init_noise_tag", False)]

        if not enabled:
            mo["sampler_pre_cfg_function"] = pre
            unet.model_options = mo
            print("[InitNoise] disabled via UI (cleaned)")
            return

        # thêm hook mới (closure mới → state['done']=False) và đặt LÊN ĐẦU
        hook = make_init_noise_pre_cfg_hook(
            iters=int(iters), step_size=float(step), rho_clip=float(rho), gamma_scale=float(gamma)
        )
        pre.insert(0, hook)
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo
        print(f"[InitNoise] pre-CFG hook installed on {type(unet).__name__}")
