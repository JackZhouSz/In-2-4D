"""Read/write frames, masks and json. Images are uint8 numpy arrays (H, W, C); masks are float32 (H, W) in [0, 1]."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_image(path, mode=None) -> np.ndarray:
    img = Image.open(path)
    if mode is not None:
        img = img.convert(mode)
    return np.asarray(img).copy()


def write_image(arr: np.ndarray, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def read_mask(path) -> np.ndarray:
    return read_image(path, "L").astype(np.float32) / 255.0


def write_mask(mask: np.ndarray, path):
    write_image((np.clip(mask, 0, 1) * 255).round().astype(np.uint8), path)


def to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb[..., :3], (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)[..., None]], -1)


def composite_white(rgba: np.ndarray) -> np.ndarray:
    """RGBA uint8 -> RGB uint8 on a white background."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
    return rgb.round().clip(0, 255).astype(np.uint8)


def write_frames(frames, frame_dir, alphas=None, alpha_dir=None):
    """frames: (T, H, W, 3) uint8, written as %04d.png; alphas likewise as masks."""
    for t, f in enumerate(frames):
        write_image(f, Path(frame_dir) / f"{t:04d}.png")
    if alphas is not None:
        for t, a in enumerate(alphas):
            write_mask(a, Path(alpha_dir) / f"{t:04d}.png")


def read_frames(frame_dir) -> np.ndarray:
    files = sorted(Path(frame_dir).glob("[0-9][0-9][0-9][0-9].png"))
    if not files:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    return np.stack([read_image(f, "RGB") for f in files])


def read_masks(mask_dir) -> np.ndarray:
    files = sorted(Path(mask_dir).glob("[0-9][0-9][0-9][0-9].png"))
    if not files:
        raise FileNotFoundError(f"no masks in {mask_dir}")
    return np.stack([read_mask(f) for f in files])


def write_video(frames, path, fps):
    """Best effort mp4 (needs imageio-ffmpeg); frames on disk are the source of truth."""
    try:
        import imageio.v2 as imageio
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)
    except Exception as e:  # noqa: BLE001 - optional artifact
        print(f"[io] could not write {path}: {e}")
