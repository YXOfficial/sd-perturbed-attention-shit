# scripts/init_noise.py
import os, sys
import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.samplers import CFGNoisePredictor  # ensure reForge backend present

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)
from init_noise_utils import set_init_noise_hook  # absolute import

class ScriptInitNoise(scripts.Script):
    def title(self):
        return "[Forge/reForge] Init Noise Adjust (first step)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Init Noise (x_T)"):
            enabled = gr.Checkbox(label="Enable init-noise adjustment", value=True)
            iters   = gr.Slider(label="Iters", minimum=1, maximum=200, step=1, value=20)
            step    = gr.Slider(label="Step size", minimum=0.001, maximum=0.5, step=0.001, value=0.05)
            rho     = gr.Slider(label="Rho clip", minimum=0, maximum=200, step=1, value=50)
            gamma   = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
        return [enabled, iters, step, rho, gamma]

    def process(self, p, enabled, iters, step, rho, gamma):
        if not enabled:
            print("[InitNoise] disabled via UI")
            return

        # --- tìm model_options theo kiểu 'duck-typing' (không lệ thuộc ModelPatcher) ---
        target_obj = None
        mo = None

        # A) trực tiếp trên sd_model
        if hasattr(p, "sd_model") and hasattr(p.sd_model, "model_options"):
            target_obj = p.sd_model
            mo = p.sd_model.model_options

        # B) một số bản để dưới .model
        elif hasattr(p, "sd_model") and hasattr(getattr(p.sd_model, "model", None), "model_options"):
            target_obj = p.sd_model.model
            mo = p.sd_model.model_options

        # C) fallback: sampler mang model_options (một số fork làm vậy)
        elif hasattr(p, "sampler") and hasattr(p.sampler, "model_options"):
            target_obj = p.sampler
            mo = p.sampler.model_options

        # D) reForge: thử lấy qua model_management (nếu có)
        if mo is None:
            try:
                from ldm_patched.modules import model_management as _mm
                cur = getattr(_mm, "current_loaded_model", None)
                if cur is not None and hasattr(cur, "model_options"):
                    target_obj = cur
                    mo = cur.model_options
            except Exception:
                pass

        if mo is None or target_obj is None:
            print("[InitNoise] backend exposes no `model_options`; cannot install hook. "
                  "Please update Forge/reForge or keep this extension disabled.")
            return

        # --- gắn hook; để chạy TRƯỚC các hook khác ---
        from init_noise_utils import set_init_noise_hook
        new_mo = set_init_noise_hook(
            mo,
            iters=int(iters),
            step_size=float(step),
            rho_clip=float(rho),
            gamma_scale=float(gamma),
            insert_at_front=True,
            ensure_list_semantics=True,
        )

        # ghi trả lại đúng chỗ đã lấy ra
        try:
            target_obj.model_options = new_mo
        except Exception as e:
            print("[InitNoise] failed to assign model_options back:", e)
            return

        print(f"[InitNoise] hook installed on {type(target_obj).__name__} "
              f"(iters={iters}, step={step}, rho={rho}, gamma={gamma})")

