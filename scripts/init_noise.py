# -*- coding: utf-8 -*-
# Init Noise (x_T) — pre-CFG once, installed at process_before_every_sampling
import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch

def make_init_noise_pre_cfg_hook(iters, step_size, rho_clip, gamma_scale):
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
                        raise RuntimeError("no-grad")
                    diff = (c - u); loss = -(diff.square().mean())
                    (-loss).backward(); opt.step()
                    with torch.no_grad():
                        if rho_clip > 0: xg.clamp_(-float(rho_clip), float(rho_clip))
                        if gamma_scale > 0: xg.copy_(x0 + float(gamma_scale) * (xg - x0))
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
                    if rho_clip > 0: xg.clamp_(-float(rho_clip), float(rho_clip))
                    if gamma_scale > 0: xg = x0 + float(gamma_scale) * (xg - x0)
            x_adj = xg.detach()

        with torch.no_grad():
            mode = "autograd" if use_grad else "nudge"
            rms = ((x_adj - x0).pow(2).mean()).sqrt().item()
        print(f"[InitNoise] adjusted x_T ({mode}, iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale}, ||Δx||_rms={rms:.5f})")
        state["done"] = True
        return model, cond, uncond, x_adj, timestep, model_options
    _hook.__init_noise_tag = True
    return _hook

class ScriptInitNoise(scripts.Script):
    def title(self): return "[Forge/reForge] Init Noise (x_T) — pre-CFG once"
    def show(self, is_img2img): return scripts.AlwaysVisible
    def ui(self, is_img2img):
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=False)
            iters   = gr.Slider("Iters", 1, 200, 1, 20)
            step    = gr.Slider("Step size", 0.001, 0.5, 0.001, 0.05)
            rho     = gr.Slider("Rho clip", 0, 200, 1, 50)
            gamma   = gr.Slider("Gamma (pull to init)", 0.0, 1.0, 0.05, 0.7)
        return [enabled, iters, step, rho, gamma]

    # ❗ Chuyển sang hook này (điểm móc đúng thời gian) ❗
    def process_before_every_sampling(self, p, enabled, iters, step, rho, gamma, **kwargs):
        if not enabled:
            return
        try:
            unet = p.sd_model.forge_objects.unet  # UNet hiện tại, đã clone xong bởi các script khác
        except Exception as e:
            print("[InitNoise] cannot access unet:", e); return

        # Dọn mọi hook InitNoise cũ khỏi UNet hiện tại
        mo = dict(unet.model_options or {})
        pre = [fn for fn in list(mo.get("sampler_pre_cfg_function", [])) if not getattr(fn, "__init_noise_tag", False)]
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo

        # Cài hook MỚI (closure mới → state['done']=False) vào UNet hiện tại
        hook = make_init_noise_pre_cfg_hook(int(iters), float(step), float(rho), float(gamma))
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre.insert(0, hook)  # chạy trước
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo
        print("[InitNoise] pre-CFG hook installed at process_before_every_sampling()")

        # Hiển thị tham số trong “Parameters”
        try:
            if hasattr(p, "extra_generation_params") and isinstance(p.extra_generation_params, dict):
                p.extra_generation_params["InitNoise iters"] = int(iters)
                p.extra_generation_params["InitNoise step"]  = float(step)
                p.extra_generation_params["InitNoise rho"]   = float(rho)
                p.extra_generation_params["InitNoise gamma"] = float(gamma)
        except: pass
