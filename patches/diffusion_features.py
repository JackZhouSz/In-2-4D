"""Stable Diffusion UNet features on modern diffusers.

The feature map is the output of unet.up_blocks[i], read with a forward hook, using the SAME generator seed for every frame
(VAE posterior sample and noise), so identical frames give identical features.
"""
import numpy as np
import torch
from PIL import Image


class _Stop(Exception):
    """Raised by the hook once the requested block ran; the rest of the UNet is skipped."""


class DiffusionFeatures:
    def __init__(self, sd_id, t=261, up_ft_index=1, ensemble=8, res=768, seed=0, dtype="float32", device="cuda",
                 chunk=None):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device, self.dtype = device, getattr(torch, dtype)
        self.t, self.ensemble, self.res, self.seed = t, ensemble, res, seed
        self.chunk = chunk or ensemble
        self.unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=self.dtype).to(device).eval()
        self.vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae", torch_dtype=self.dtype).to(device).eval()
        self.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        tok = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
        enc = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder", torch_dtype=self.dtype).to(device)
        ids = tok("", padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.null_embeds = enc(ids.to(device))[0]
        del enc

        self._feat = None
        self.unet.up_blocks[up_ft_index].register_forward_hook(self._hook)

    def to(self, device):
        """Park the models on CPU between analyses (frees the GPU that Wan shares)."""
        self.unet.to(device)
        self.vae.to(device)
        self.null_embeds = self.null_embeds.to(device)
        return self

    def _hook(self, module, inputs, output):
        self._feat = output.detach()
        raise _Stop

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> [C, h, w] ensemble-averaged features (1280 x 48 x 48 at res 768, index 1)."""
        img = np.asarray(Image.fromarray(rgb[..., :3]).resize((self.res, self.res)))
        x = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1.0).permute(2, 0, 1)[None]
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        params = self.vae.encode(x.to(self.device, self.dtype)).latent_dist.parameters
        posterior = DiagonalGaussianDistribution(params.repeat(self.ensemble, 1, 1, 1))
        g = torch.Generator(self.device).manual_seed(self.seed)
        latents = posterior.sample(generator=g) * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=g, device=self.device, dtype=latents.dtype)
        t = torch.tensor(self.t, dtype=torch.long, device=self.device)
        x_t = self.scheduler.add_noise(latents, noise, t)
        feat = 0
        for x_c in x_t.split(self.chunk):
            try:
                self.unet(x_c, t, encoder_hidden_states=self.null_embeds.repeat(len(x_c), 1, 1))
            except _Stop:
                pass
            feat = feat + self._feat.float().sum(0)
        return feat / self.ensemble
