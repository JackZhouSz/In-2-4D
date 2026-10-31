"""Stage 2 — first+last-frame generation of the root video; video.backbone: wan | ltx.

Outputs: video/generated/root/{video.mp4, frames/%04d.png, alpha/%04d.png, meta.json}
`make_generator` and `generate_clip` are reused by Stage 4 for every fragment regeneration.
"""
import numpy as np
import torch
from PIL import Image

from core.io import composite_white, read_image, read_json, write_frames, write_json, write_video
from core.matting import matte_video
from core.metrics import masked_lpips
from core.paths import WAN, resolve, use_repo
from core.stage import StageContext, stage_parser


class FLFGenerator:
    """Wan first+last-frame video model. backend: wan22_patched (default) | wan21_flf2v (diffusers fallback)."""

    def __init__(self, cfg):
        self.cfg = cfg.wan
        use_repo(WAN)
        from patches import video_flf, video_rope_retime

        video_flf.install()
        video_rope_retime.install()
        self.split = len(self.cfg.devices) > 1
        if self.cfg.backend == "wan22_patched":
            import copy

            import wan
            from wan.configs import WAN_CONFIGS

            from patches import video_multigpu

            config = copy.deepcopy(WAN_CONFIGS["i2v-A14B"])
            config.param_dtype = getattr(torch, self.cfg.dtype)
            if self.cfg.attention == "sdpa":
                video_multigpu.install_sdpa(config.param_dtype)
            self.model = wan.WanI2V(config=config, checkpoint_dir=str(resolve(cfg.paths.wan_ckpt)),
                                    device_id=self.cfg.devices[0], rank=0, t5_cpu=self.cfg.t5_cpu,
                                    convert_model_dtype=self.cfg.convert_model_dtype)
            if self.cfg.lora_dir:
                from patches.video_distill_lora import merge_into_wan
                merge_into_wan(self.model, resolve(self.cfg.lora_dir), self.cfg.lora_strength)
            if self.split:
                video_multigpu.enable_split(self.model, self.cfg.devices, self.cfg.block_split)
        elif self.cfg.backend == "wan21_flf2v":
            from diffusers import WanImageToVideoPipeline

            self.model = WanImageToVideoPipeline.from_pretrained(self.cfg.wan21_flf2v_id, torch_dtype=torch.bfloat16)
            self.model.enable_model_cpu_offload(gpu_id=self.cfg.devices[0])
        else:
            raise ValueError(f"unknown wan.backend '{self.cfg.backend}'")

    def __call__(self, start_rgb: np.ndarray, end_rgb: np.ndarray, prompt: str, neg: str, frame_num: int,
                 seed: int) -> np.ndarray:
        """RGB uint8 inputs (same size) -> (frame_num, H, W, 3) uint8."""
        assert start_rgb.shape == end_rgb.shape, "start and end must share one (h, w)"
        assert (frame_num - 1) % 4 == 0, "frame_num must be 4n+1"
        c = self.cfg
        img, img_end = Image.fromarray(start_rgb), Image.fromarray(end_rgb)
        if c.backend == "wan22_patched":
            video = self.model.generate_flf(prompt, img, img_end, max_area=c.max_area, frame_num=frame_num,
                                            shift=c.shift, sample_solver=c.solver, sampling_steps=c.steps,
                                            guide_scale=tuple(c.guide), n_prompt=neg, seed=seed,
                                            offload_model=c.offload_model and not self.split)
            if self.split:
                self.model.release()
            return ((video.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
        side = int(round(np.sqrt(c.max_area) / 16)) * 16
        frames = self.model(image=img, last_image=img_end, prompt=prompt, negative_prompt=neg, height=side,
                            width=side, num_frames=frame_num, num_inference_steps=c.steps,
                            guidance_scale=float(c.guide[0]), output_type="np",
                            generator=torch.Generator("cuda").manual_seed(seed)).frames[0]
        return (np.asarray(frames) * 255).round().astype(np.uint8)


def make_generator(cfg):
    if cfg.video.backbone == "ltx":
        from core.ltx_video import LTXGenerator
        return LTXGenerator(cfg)
    if cfg.video.backbone == "wan":
        return FLFGenerator(cfg)
    raise ValueError(f"unknown video.backbone '{cfg.video.backbone}'")


def generate_clip(gen, start_rgba, end_rgba, prompt, neg, frame_num, seed, cfg, keyframes=()):
    """Generate, resize to the framed size, re-matte (seeded with the start mask) and measure FLF fidelity.
    keyframes = [(frame_index, rgba)] pinned inside the clip (multi-condition backbones only)."""
    size = cfg.framing.size
    extra = {"keyframes": [(f, composite_white(k)) for f, k in keyframes]} if keyframes else {}
    frames = gen(composite_white(start_rgba), composite_white(end_rgba), prompt, neg, frame_num, seed, **extra)
    if frames.shape[1:3] != (size, size):
        frames = np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.LANCZOS)) for f in frames])
    alphas = matte_video(frames, start_rgba[..., 3] / 255.0, cfg)
    s_alpha, e_alpha = start_rgba[..., 3] / 255.0, end_rgba[..., 3] / 255.0
    fidelity = {
        "lpips_first": masked_lpips(frames[0], composite_white(start_rgba), np.maximum(alphas[0], s_alpha)),
        "lpips_last": masked_lpips(frames[-1], composite_white(end_rgba), np.maximum(alphas[-1], e_alpha)),
    }
    return frames, alphas, fidelity


def check_flf(check, name, fidelity, cfg):
    """last frame vs img_end < 0.15, first frame vs img < 0.10 (masked LPIPS)."""
    check(name, fidelity["lpips_last"] < cfg.video.flf_lpips_end and fidelity["lpips_first"] < cfg.video.flf_lpips_start,
          **fidelity, hint="if this fails consistently, try another seed (video.seed) or backbone (video.backbone)")


def save_clip(out_dir, frames, alphas, meta, fps):
    write_frames(frames, out_dir / "frames", alphas, out_dir / "alpha")
    write_video(frames, out_dir / "video.mp4", fps)
    write_json(meta, out_dir / "meta.json")


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 2)
    root = ctx.out / "root"
    if ctx.done(root / "meta.json"):
        return
    cfg = ctx.cfg
    cap = read_json(ctx.dirs.s1 / "caption.json")
    start = read_image(ctx.dirs.s0 / "start.png", "RGBA")
    end = read_image(ctx.dirs.s0 / "end.png", "RGBA")

    gen = make_generator(cfg)
    frames, alphas, fid = generate_clip(gen, start, end, cap["prompt"], cap["negative_prompt"],
                                        cfg.video.frame_num, cfg.video.seed, cfg)
    check_flf(ctx.check, "flf_fidelity_root", fid, cfg)
    save_clip(root, frames, alphas, {"frame_num": cfg.video.frame_num, "seed": cfg.video.seed, "fps": cfg.video.fps,
                                     "prompt": cap["prompt"], "backbone": cfg.video.backbone, **fid}, cfg.video.fps)


if __name__ == "__main__":
    main()
