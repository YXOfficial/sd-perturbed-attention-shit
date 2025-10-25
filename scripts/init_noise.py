try:
    import gradio as gr
    from modules import scripts
    from modules.ui_components import InputAccordion
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules.samplers import CFGNoisePredictor  # just to ensure reForge backend present

    import os, sys

    # thêm đường dẫn gốc của extension vào sys.path để import tuyệt đối
    EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if EXT_DIR not in sys.path:
        sys.path.insert(0, EXT_DIR)

    from init_noise_utils import set_init_noise_hook  # <-- TUYỆT ĐỐI, không dùng '..'

    class ScriptInitNoise(scripts.Script):
        def title(self):
            return "[Forge/reForge] Init Noise Adjust (first step)"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            with InputAccordion(False, label="Init Noise (x_T)"):
                with gr.Group():
                    iters = gr.Slider(label="Iters", minimum=1, maximum=200, step=1, value=20)
                    step = gr.Slider(label="Step size", minimum=0.001, maximum=0.5, step=0.001, value=0.05)
                    rho = gr.Slider(label="Rho clip", minimum=0, maximum=200, step=1, value=50)
                    gamma = gr.Slider(label="Gamma (pull to init)", minimum=0.0, maximum=1.0, step=0.05, value=0.7)
            return [iters, step, rho, gamma]

        def process(self, p, iters, step, rho, gamma):
            # p.sd_model is a ModelPatcher in reForge
            if isinstance(p.sd_model, ModelPatcher):
                mo = p.sd_model.model_options
                mo = set_init_noise_hook(mo, iters=int(iters), step_size=float(step),
                                         rho_clip=float(rho), gamma_scale=float(gamma))
                p.sd_model.model_options = mo

except Exception as e:
    # allow the rest of the extension to work even if this script fails to load
    print("[InitNoise] UI disabled:", e)
