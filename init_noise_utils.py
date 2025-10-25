# init_noise_utils.py
from typing import Dict, Any, Callable
import torch

def _grad_step_on_latents(model, x, sigma, cond, uncond, model_options,
                          iters: int, step_size: float, rho_clip: float, gamma_scale: float):
    x0 = x.detach().clone()
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([x], lr=step_size)
        for _ in range(max(1, iters)):
            opt.zero_grad(set_to_none=True)
            eps_c = model(x, sigma, cond=cond,   model_options=model_options)
            eps_u = model(x, sigma, cond=uncond, model_options=model_options)
            eps_tilde = eps_c - eps_u
            # maximize ||eps_tilde||^2  <=> minimize negative mean
            loss = -(eps_tilde.square().mean())
            (-loss).backward()
            opt.step()
            with torch.no_grad():
                if rho_clip and rho_clip > 0:
                    x.clamp_(-rho_clip, rho_clip)
                if gamma_scale and gamma_scale > 0:
                    x.copy_(x0 + gamma_scale * (x - x0))
    return x.detach()

def make_init_noise_pre_cfg_hook(iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7) -> Callable:
    state = {"done": False, "first_sigma": None}

    def _hook(model, cond, uncond, x, sigma, model_options: Dict[str, Any]):
        if state["done"]:
            return model, cond, uncond, x, sigma, model_options

        # detect first step and only run once
        if state["first_sigma"] is None:
            state["first_sigma"] = float(sigma[0].item()) if torch.is_tensor(sigma) else float(sigma)

        print(f"[InitNoise] running at sigma={state['first_sigma']:.6f} "
              f"(iters={iters}, step={step_size}, rho={rho_clip}, gamma={gamma_scale})")
        x_adj = _grad_step_on_latents(
            model=model, x=x, sigma=sigma, cond=cond, uncond=uncond,
            model_options=model_options, iters=iters, step_size=step_size,
            rho_clip=rho_clip, gamma_scale=gamma_scale
        )
        state["done"] = True
        return model, cond, uncond, x_adj, sigma, model_options

    return _hook

def _as_list_or_compose(existing, new_hook, insert_at_front, ensure_list_semantics):
    if ensure_list_semantics:
        lst = []
        if existing is None:
            pass
        elif isinstance(existing, (list, tuple)):
            lst = list(existing)
        else:
            # was single callable → keep it, but our hook should run first/last per flag
            lst = [existing]
        if insert_at_front:
            lst = [new_hook] + lst
        else:
            lst = lst + [new_hook]
        return lst
    else:
        # legacy: always append to list (or create single)
        if existing is None:
            return [new_hook]
        if isinstance(existing, (list, tuple)):
            return list(existing) + [new_hook]
        # compose single→single
        def composed(model, cond, uncond, x, sigma, model_options):
            a = new_hook(model, cond, uncond, x, sigma, model_options)
            return existing(*a)  # expects same signature
        return composed

def set_init_noise_hook(model_options: Dict[str, Any],
                        iters=20, step_size=0.05, rho_clip=50.0, gamma_scale=0.7,
                        insert_at_front=True, ensure_list_semantics=True) -> Dict[str, Any]:
    model_options = dict(model_options or {})
    key = "sampler_pre_cfg_function"
    hooks_existing = model_options.get(key, None)
    hook = make_init_noise_pre_cfg_hook(iters, step_size, rho_clip, gamma_scale)
    model_options[key] = _as_list_or_compose(
        hooks_existing, hook, insert_at_front, ensure_list_semantics
    )
    return model_options
