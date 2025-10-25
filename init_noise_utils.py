# reForge/Forge-compatible "init noise" for first step only
from typing import Dict, Any, Callable
import torch

def _grad_step_on_latents(model, x, sigma, cond, uncond, model_options,
                          iters: int, step_size: float, rho_clip: float, gamma_scale: float):
    """
    Do a few gradient steps on x (latents) at the very first sigma.
    Heuristic goal: push x along (eps_c - eps_u) direction with size control.

    - model: reForge wrapped unet callable -> model(x, sigma, cond=..., model_options=...)
    - x: latents tensor [B,4,H/8,W/8]
    - sigma: current sigma (FloatTensor)
    """
    device = x.device
    x0 = x.detach().clone()

    # keep optimizer simple to avoid overhead
    x = x.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([x], lr=step_size)

    for _ in range(max(1, iters)):
        opt.zero_grad(set_to_none=True)
        # predict eps for cond/uncond
        eps_c = model(x, sigma, cond=cond, model_options=model_options)
        eps_u = model(x, sigma, cond=uncond, model_options=model_options)

        # target direction: eps_tilde = eps_c - eps_u
        eps_tilde = eps_c - eps_u

        # maximize ||eps_tilde|| -> minimize negative L2
        # (use mean to keep scale stable)
        loss = -(eps_tilde.square().mean())

        (-loss).backward()
        opt.step()

        with torch.no_grad():
            # soft clamp x so it doesn't explode; rho_clip ~ 50 (latents are ~N(0,1))
            if rho_clip is not None and rho_clip > 0:
                x.clamp_(-rho_clip, rho_clip)

            # gentle pull toward init latent to keep distributional shape
            if gamma_scale and gamma_scale > 0:
                x.copy_(x0 + gamma_scale * (x - x0))

    return x.detach()

def make_init_noise_pre_cfg_hook(iters: int = 20,
                                 step_size: float = 0.05,
                                 rho_clip: float = 50.0,
                                 gamma_scale: float = 0.7) -> Callable:
    """
    Factory -> a function(model, cond, uncond, x, sigma, model_options)
               returning (model, cond, uncond, x, sigma, model_options)
    Runs once at the very first call (tracked in model_options).
    """
    state = {"done": False, "first_sigma": None}

    def _hook(model, cond, uncond, x, sigma, model_options: Dict[str, Any]):
        # if already adjusted, bypass
        if state["done"]:
            return model, cond, uncond, x, sigma, model_options

        # record first sigma and only run once there
        if state["first_sigma"] is None:
            state["first_sigma"] = float(sigma[0].item()) if torch.is_tensor(sigma) else float(sigma)

        # run only at first encountered sigma
        curr = float(sigma[0].item()) if torch.is_tensor(sigma) else float(sigma)
        if abs(curr - state["first_sigma"]) > 1e-7:
            # safety: if sampler calls us later first, still run once
            state["first_sigma"] = curr

        # perform the adjustment
        x_adj = _grad_step_on_latents(
            model=model, x=x, sigma=sigma, cond=cond, uncond=uncond,
            model_options=model_options, iters=iters, step_size=step_size,
            rho_clip=rho_clip, gamma_scale=gamma_scale
        )

        state["done"] = True
        return model, cond, uncond, x_adj, sigma, model_options

    return _hook

def set_init_noise_hook(model_options: Dict[str, Any],
                        iters: int = 20, step_size: float = 0.05,
                        rho_clip: float = 50.0, gamma_scale: float = 0.7) -> Dict[str, Any]:
    """
    Append our pre-CFG hook into model_options (non-destructive).
    """
    to = model_options.get("transformer_options", {}).copy()
    model_options = model_options.copy()
    model_options["transformer_options"] = to

    hooks = model_options.get("sampler_pre_cfg_function", [])
    hooks = list(hooks) if isinstance(hooks, (list, tuple)) else []
    hooks.append(make_init_noise_pre_cfg_hook(
        iters=iters, step_size=step_size, rho_clip=rho_clip, gamma_scale=gamma_scale
    ))
    model_options["sampler_pre_cfg_function"] = hooks
    return model_options
