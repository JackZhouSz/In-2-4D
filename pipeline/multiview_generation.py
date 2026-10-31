"""Stage 5 — Era3D ortho multi-view per fragment frame with EMA attention injection.

Outputs: multiview/frag_XX/mv/tTTT_vV.png (rgb), tTTT_vV_normal.png, tTTT_vV_alpha.png, multiview/frag_XX/meta.json
         (with eval.injection_ablation: multiview/frag_XX/mv_noinj/ — same, injection off)
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from core.cameras import ERA3D_VIEWS
from core.framing import era3d_place
from core.io import read_frames, read_json, read_masks, write_image, write_json, write_mask
from core.matting import matte_white_threshold
from core.paths import ERA3D, ERA3D_PROMPT_EMBEDS, use_repo
from core.stage import StageContext, stage_parser
from patches.multiview_injection import Inj


def load_pipeline(cfg):
    """As Era3D/test_mvdiffusion_unclip.py::load_era3d_pipeline, after installing the injection patch."""
    use_repo(ERA3D)
    from patches import multiview_injection
    multiview_injection.install()
    from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline

    era3d_cfg = OmegaConf.load(ERA3D / cfg.era3d.config)
    assert not era3d_cfg.regress_elevation and not era3d_cfg.regress_focal_length, "use the ortho model"
    # a local folder, as Era3D's scripts use: diffusers' Hub download path rejects the custom
    # `mvdiffusion.models.unet_mv2d_condition` class named in model_index.json
    ckpt = Path(cfg.paths.era3d_ckpt)
    if not ckpt.is_dir():
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(cfg.paths.era3d_ckpt)
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(str(ckpt), torch_dtype=torch.float16)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_slicing()
    pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


class Era3DRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = load_pipeline(cfg)
        normal = torch.load(ERA3D_PROMPT_EMBEDS / "normal_embeds.pt")
        color = torch.load(ERA3D_PROMPT_EMBEDS / "clr_embeds.pt")
        self.prompt_embeds = torch.cat([normal, color], 0)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        """512x512 RGB uint8 on white -> (normals, colors) each (6, 3, H, W) in [0, 1]."""
        e = self.cfg.era3d
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        imgs_in = x[None].repeat(len(ERA3D_VIEWS) * 2, 1, 1, 1)
        g = torch.Generator(device=self.pipe.unet.device).manual_seed(e.seed)   # re-created for EVERY frame
        with torch.autocast("cuda"):
            out = self.pipe(imgs_in, None, prompt_embeds=self.prompt_embeds, generator=g,
                            guidance_scale=e.cfg, output_type="pt", num_images_per_prompt=1,
                            num_inference_steps=e.steps, eta=e.eta,
                            callback=Inj.step_callback, callback_steps=1).images
        n = out.shape[0] // 2
        return out[:n].float().cpu(), out[n:].float().cpu()


def run_fragment(runner: Era3DRunner, frames, alphas, era3d, out_dir, cfg, check, frag_id, inject=True):
    inj = cfg.injection
    Inj.s, Inj.ema = inj.s, inj.mode == "ema"
    use_inj = inject and inj.mode != "off"
    for t in range(len(frames)):
        rgb, _ = era3d_place(frames[t], alphas[t], era3d, cfg.era3d.img_size)
        mode = ("record" if t == 0 else "inject") if use_inj else "off"
        Inj.start_frame(mode, cfg.era3d.steps, inj.inject_steps_frac)
        normals, colors = runner(rgb)
        if mode != "off":
            check(f"injection_alignment_frag{frag_id:02d}", Inj.idx == len(Inj.cache), t=t, idx=Inj.idx,
                  cache=len(Inj.cache))
        for v in range(len(ERA3D_VIEWS)):
            col = (colors[v].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            nrm = (normals[v].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            alpha = matte_white_threshold(col, cfg.era3d.alpha_threshold, cfg.era3d.alpha_erode_px)
            write_image(col, out_dir / f"t{t:03d}_v{v}.png")
            write_image(nrm, out_dir / f"t{t:03d}_v{v}_normal.png")
            write_mask(alpha, out_dir / f"t{t:03d}_v{v}_alpha.png")
    Inj.mode, Inj.cache = "off", []


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 5)
    cfg = ctx.cfg
    frags = read_json(ctx.dirs.fragments_json)["fragments"]
    era3d = read_json(ctx.dirs.framing_json)["era3d"]
    runner = None
    for f in frags:
        out = ctx.dirs.frag_dir(5, f["id"])
        if not ctx.mine(f["id"]) or ctx.done(out / "meta.json"):
            continue
        runner = runner or Era3DRunner(cfg)
        src = ctx.dirs.root / f["dir"]
        frames, alphas = read_frames(src / "frames"), read_masks(src / "alpha")
        run_fragment(runner, frames, alphas, era3d, out / "mv", cfg, ctx.check, f["id"])
        if cfg.eval.injection_ablation:
            run_fragment(runner, frames, alphas, era3d, out / "mv_noinj", cfg, ctx.check, f["id"], inject=False)
        write_json({"num_frames": len(frames), "views": ERA3D_VIEWS, "era3d": era3d,
                    "injection": cfg.injection.to_dict(), "seed": cfg.era3d.seed}, out / "meta.json")


if __name__ == "__main__":
    main()
