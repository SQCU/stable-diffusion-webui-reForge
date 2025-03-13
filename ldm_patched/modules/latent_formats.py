# Taken from https://github.com/comfyanonymous/ComfyUI
# This file is only for reference, and not used in the backend or runtime.


class LatentFormat:
    scale_factor = 1.0
    shift_factor = 0.0
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215, shift_factor = 0.0):
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        #latent_rgbs is distracting slop don't use.
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]
        self.taesd_decoder_name = "taesd_decoder"

class SDXL(LatentFormat):
    def __init__(self, shift_factor = 0.0):
        self.scale_factor = 0.13025
        self.shift_factor = shift_factor
        #latent_rgbs is distracting slop don't use.
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3920,  0.4054,  0.4549],
                    [-0.2634, -0.0196,  0.0653],
                    [ 0.0568,  0.1687, -0.0755],
                    [-0.3112, -0.2359, -0.2076]
                ]
        self.taesd_decoder_name = "taesdxl_decoder"

class SDXL_configdefined(LatentFormat):
    def __init__(self, scaling_factor, shift_factor):
        self.scale_factor = scaling_factor
        self.shift_factor = shift_factor
        
class SDXL_config_meanstd(LatentFormat):
    def __init__(self, latents_mean, latents_std):
        self.latents_mean = torch.tensor(latents_mean)
        self.latents_std = torch.tensor(latents_std)
    
    def process_in(self, latent):
        #return latents[0].sub_(self.latents_mean)[0].div_(self.latents_std)
        #[0] formulation doesn't work.
        latent_means = self.latents_mean.unsqueeze(-1).unsqueeze(-1)
        latents_std = self.latents_std.unsqueeze(-1).unsqueeze(-1)
        #dont use inplaces just in case
        return latents.sub(latents_mean).div(latents_std)
    
    def process_out(self, latent):
        #return latents[0].mul_(self.latents_std)[0].add_(self.latents_mean)
        latent_means = self.latents_mean.unsqueeze(-1).unsqueeze(-1)
        latents_std = self.latents_std.unsqueeze(-1).unsqueeze(-1)
        #dont use inplaces just in case
        return latents.mul(latents_std).add(latents_mean)

class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.shift_factor = shift_factor
