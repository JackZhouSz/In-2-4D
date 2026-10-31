"""First+last-frame conditioning for Wan2.2-I2V-A14B.

`generate_flf` is a copy of wan/image2video.py::WanI2V.generate with exactly these changes:
  * the end image is encoded into the last latent frame (mask 1 on the first AND last frame),
  * both images are resized to the identical (h, w) that Wan computes from the start image,
  * the retiming hook receives the normalized timestep before the two model calls,
  * `sample_solver='euler'` (Wan2.2-Lightning's solver, patches/video_distill_lora.py) and no unconditional
    model call when the guide scale is 1 (as the Lightning fork does).
Install with `install()`, which adds the method to WanI2V without editing the submodule.
"""
import gc
import math
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .video_rope_retime import RETIME


def generate_flf(self, input_prompt, img, img_end, max_area=720 * 1280, frame_num=81, shift=5.0,
                 sample_solver='unipc', sampling_steps=40, guide_scale=5.0, n_prompt="", seed=-1,
                 offload_model=True):
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
    img_end = TF.to_tensor(img_end).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(16, (F - 1) // self.vae_stride[0] + 1, lat_h, lat_w, dtype=torch.float32,
                        generator=seed_g, device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:-1] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    rs = lambda x: torch.nn.functional.interpolate(x[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1)
    y = self.vae.encode([torch.concat([rs(img), torch.zeros(3, F - 2, h, w), rs(img_end)], dim=1).to(self.device)])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
    no_sync_high_noise = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

    with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
    ):
        boundary = self.boundary * self.num_train_timesteps

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        elif sample_solver == 'euler':
            from .video_distill_lora import EulerScheduler
            sample_scheduler = EulerScheduler(self.num_train_timesteps, shift=shift)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise
        arg_c = {'context': [context[0]], 'seq_len': max_seq_len, 'y': [y]}
        arg_null = {'context': context_null, 'seq_len': max_seq_len, 'y': [y]}

        if offload_model:
            torch.cuda.empty_cache()

        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(self.device)]
            timestep = [t]
            timestep = torch.stack(timestep).to(self.device)

            model = self._prepare_model_for_timestep(t, boundary, offload_model)
            sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

            RETIME.set_timestep(t.item() / self.num_train_timesteps)
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            if offload_model:
                torch.cuda.empty_cache()
            if abs(sample_guide_scale - 1.0) > 1e-6:
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                                            return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            x0 = [latent]
            del latent_model_input, timestep

        if offload_model:
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent, x0
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def install():
    from wan.image2video import WanI2V

    WanI2V.generate_flf = generate_flf
