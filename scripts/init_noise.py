# -*- coding: utf-8 -*-
# [Forge/reForge] Init Noise (x_T) — pre-CFG once per Generate (installed just before sampling)

import gradio as gr
from modules import scripts, shared
from ldm_patched.modules.samplers import calc_cond_uncond_batch  # PATH: Test-reForge-main/ldm_patched/modules/samplers.py

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7):
    """
    pre-CFG hook signature:
      fn(model, cond, uncond, x, timestep, model_options) -> (model, cond, uncond, x', timestep, model_options)
    Chạy đúng 1 lần ở call đầu nhờ closure state['done'].
    """
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

                    # >>> pre-CFG được gọi tại:
                    # /content/Test-reForge/Test-reForge-main/ldm_patched/modules/samplers.py
                    # lines ~288-289:
                    # for fn in model_options.get("sampler_pre_cfg_function", []):
                    #     model, cond, uncond_, x, timestep, model_options = fn(...)
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

    _hook.__init_noise_tag = True
    return _hook


class ScriptInitNoise(scripts.Script):
    def title(self):
        # đổi title để phá cache UI cũ nếu có
        return "[Forge/reForge] Init Noise (x_T) — pre-CFG once (PBS)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Init Noise (x_T)", open=False):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=False)
            iters   = gr.Slider(label="Iters", minimum=1, maximum=200, step=1, value=20)
            step    = gr.Slider(label="Step size", minimum=0.001, maximum=0.5, step=0.001, value=0.05)
            rho     = gr.Slider(label="Rho clip", minimum=0, maximum=200, step=1, value=50)
            gamma   = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
        return [enabled, iters, step, rho, gamma]

    # >>> Móc vào đúng thời điểm:
    # /content/Test-reForge/Test-reForge-main/modules/scripts.py calls this before sampling starts
    # Sau các extension khác (PAG/NAG/AMS…) đã clone/gán UNet:
    # /content/Test-reForge/Test-reForge-main/extensions-builtin/mahiro_reforge/scripts/mahiro_cfg_script.py
    # lines ~46-80 -> unet = p.sd_model.forge_objects.unet.clone(); p.sd_model.forge_objects.unet = unet
    def process_before_every_sampling(self, p, enabled, iters, step, rho, gamma, **kwargs):
        if not enabled:
            return
        try:
            # TẠI THỜI ĐIỂM NÀY, UNet đã là bản clone cuối cùng sẽ dùng để sample:
            # /content/Test-reForge/Test-reForge-main/modules/sd_samplers_cfg_denoiser.py
            # line ~200: model_options = self.inner_model.inner_model.forge_objects.unet.model_options
            unet = p.sd_model.forge_objects.unet
        except Exception as e:
            print("[InitNoise] cannot access unet:", e)
            return

        # Dọn hook cũ của chính InitNoise
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre = [fn for fn in pre if not getattr(fn, "__init_noise_tag", False)]
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo

        # Cài hook MỚI cho lượt này (closure mới ⇒ state['done']=False)
        hook = make_init_noise_pre_cfg_hook(int(iters), float(step), float(rho), float(gamma))
        mo = dict(unet.model_options or {})
        pre = list(mo.get("sampler_pre_cfg_function", []))
        pre.insert(0, hook)  # chạy trước các hook khác
        mo["sampler_pre_cfg_function"] = pre
        unet.model_options = mo
        print("[InitNoise] pre-CFG hook installed at process_before_every_sampling()")

        # Xuất tham số vào infotext “Parameters”
        try:
            if hasattr(p, "extra_generation_params") and isinstance(p.extra_generation_params, dict):
                p.extra_generation_params["InitNoise iters"] = int(iters)
                p.extra_generation_params["InitNoise step"]  = float(step)
                p.extra_generation_params["InitNoise rho"]   = float(rho)
                p.extra_generation_params["InitNoise gamma"] = float(gamma)
        except Exception:
            pass
