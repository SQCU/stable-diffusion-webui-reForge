import modules.scripts as scripts
from modules import extra_networks
from modules.processing import StableDiffusionProcessing
import gradio as gr
from loractl.lib import utils, plot, lora_ctl_network, network_patch
from modules_forge.forge_sampler import sampling_prepare

class LoraCtlScript(scripts.Script):
    def __init__(self):
        self.original_network = None
        super().__init__()

    def title(self):
        return "Dynamic Lora Weights (reForge)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Dynamic Lora Weights", open=False):
                opt_enable = gr.Checkbox(
                    value=False, label="Enable Dynamic Lora Weights")
                opt_plot_lora_weight = gr.Checkbox(
                    value=False, label="Plot the LoRA weight in all steps")
        return [opt_enable, opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        if opt_enable and type(extra_networks.extra_network_registry["lora"]) != lora_ctl_network.LoraCtlNetwork:
            self.original_network = extra_networks.extra_network_registry["lora"]
            network = lora_ctl_network.LoraCtlNetwork()
            extra_networks.register_extra_network(network)
            extra_networks.register_extra_network_alias(network, "loractl")
        elif not opt_enable and type(extra_networks.extra_network_registry["lora"]) != lora_ctl_network.LoraCtlNetwork.__bases__[0]:
            extra_networks.register_extra_network(self.original_network)
            self.original_network = None
        network_patch.apply()
        utils.set_hires(False)
        utils.set_active(opt_enable)
        lora_ctl_network.reset_weights()
        plot.reset_plot()

    def before_hr(self, p, *args):
        utils.set_hires(True)

    def postprocess(self, p, processed, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        if opt_plot_lora_weight and opt_enable:
            processed.images.extend([plot.make_plot()])

    def process_before_every_sampling(self, p: StableDiffusionProcessing, cond, mask, *args, **kwargs):
        temp = p.sampler.callback_state
        def callback_state_inner(d):
            step=d['i']
            lora_ctl_network.step = step
            extra_networks.activate(p, p.extra_network_data)
            sampling_prepare(p.sampler.model_wrap.inner_model.forge_objects_after_applying_lora.unet, d['x'])
            return temp(d)
        p.sampler.callback_state = callback_state_inner

    def postprocess_image(self, p, pp: scripts.PostprocessImageArgs, *args):
        if isinstance(extra_networks.extra_network_registry["lora"], lora_ctl_network.LoraCtlNetwork):
            extra_networks.extra_network_registry["lora"].clear()
        return super().postprocess_image(p, pp, *args)
        