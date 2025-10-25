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
        if not isinstance(p.sd_model, ModelPatcher):
            print("[InitNoise] not a ModelPatcher backend; skip")
            return

        mo = p.sd_model.model_options
        # put our hook at the FRONT to run before others (e.g., NAG/PAG)
        mo = set_init_noise_hook(
            mo,
            iters=int(iters),
            step_size=float(step),
            rho_clip=float(rho),
            gamma_scale=float(gamma),
            insert_at_front=True,       # NEW
            ensure_list_semantics=True  # NEW
        )
        p.sd_model.model_options = mo
        print(f"[InitNoise] hook installed (iters={iters}, step={step}, rho={rho}, gamma={gamma})")
