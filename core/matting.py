"""Foreground alpha for single images and for videos (temporal stability matters)."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_REMBG_SESSION = None


def matte_white_threshold(rgb: np.ndarray, thresh: float, erode_px: int = 0) -> np.ndarray:
    """Foreground = pixels that differ from white; largest-component cleanup + optional erosion."""
    from scipy import ndimage

    diff = np.abs(1.0 - rgb[..., :3].astype(np.float32) / 255.0).max(-1)
    fg = ndimage.binary_closing(diff > thresh, iterations=2)
    fg = ndimage.binary_fill_holes(fg)
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    if erode_px > 0:
        fg = ndimage.binary_erosion(fg, iterations=erode_px)
    return fg.astype(np.float32)


def matte_rembg(rgb: np.ndarray, model: str = "birefnet-general") -> np.ndarray:
    """model: any rembg session name, e.g. u2net, isnet-general-use, birefnet-general (keeps thin structures)."""
    global _REMBG_SESSION
    from rembg import new_session, remove

    if _REMBG_SESSION is None or _REMBG_SESSION.model_name != model:
        _REMBG_SESSION = new_session(model)
    rgba = remove(rgb[..., :3], session=_REMBG_SESSION)
    return np.asarray(rgba)[..., 3].astype(np.float32) / 255.0


def release_rembg():
    """Drop the cached rembg session so onnxruntime returns its CUDA arena (needed before a one-GPU video model)."""
    global _REMBG_SESSION
    _REMBG_SESSION = None
    import gc
    gc.collect()


def matte_image(rgb: np.ndarray, cfg) -> np.ndarray:
    if cfg.matting.image == "rembg":
        return matte_rembg(rgb, cfg.matting.rembg_model)
    if cfg.matting.image == "white_threshold":
        return matte_white_threshold(rgb, cfg.matting.white_threshold)
    raise ValueError(f"unknown matting.image '{cfg.matting.image}'")


def matte_video(frames: np.ndarray, first_alpha: np.ndarray, cfg) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, H, W) alpha. SAM2 is seeded with the start mask (preferred)."""
    method = cfg.matting.video
    if method == "sam2":
        try:
            return _matte_sam2(frames, first_alpha, cfg)
        except ImportError as e:
            print(f"[matting] WARNING: SAM2 unavailable ({e}); falling back to per-frame rembg")
            method = "rembg"
    if method == "rembg":
        return np.stack([matte_rembg(f, cfg.matting.rembg_model) for f in frames])
    if method == "white_threshold":
        return np.stack([matte_white_threshold(f, cfg.matting.white_threshold) for f in frames])
    raise ValueError(f"unknown matting.video '{method}'")


def _matte_sam2(frames, first_alpha, cfg):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    from .paths import resolve

    predictor = build_sam2_video_predictor(cfg.matting.sam2_cfg, str(resolve(cfg.matting.sam2_ckpt)))
    out = np.zeros(frames.shape[:3], np.float32)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for t, f in enumerate(frames):
            Image.fromarray(f).save(Path(tmp) / f"{t:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=tmp)
        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=first_alpha > 0.5)
        for t, _, mask_logits in predictor.propagate_in_video(state):
            out[t] = (mask_logits[0, 0] > 0.0).float().cpu().numpy()
    del predictor
    torch.cuda.empty_cache()
    return out
