"""One global framing per case.

Coordinates:
  * source  — pixel coordinates of the original input images.
  * framed  — the global square crop of the source, resized to `framing.size` (Wan resolution).
              Every Wan output frame already lives in framed coordinates.
  * era3d   — a 512x512 white canvas; framed images are cropped by the union bbox over all frames
              of the case and scaled by ONE factor (`era3d_scale`), never per frame.
"""
import numpy as np
from PIL import Image


def alpha_bbox(alpha: np.ndarray, thresh=0.0):
    """(x0, y0, x1, y1), exclusive max, of pixels with alpha > thresh."""
    ys, xs = np.nonzero(alpha > thresh)
    if len(xs) == 0:
        raise ValueError("empty alpha mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def union_bbox(alphas, thresh=0.0):
    boxes = np.array([alpha_bbox(a, thresh) for a in alphas])
    return int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())


def square_framing(alphas, margin: float, size: int) -> dict:
    """Global square around the union bbox of the given (source-space) alphas."""
    x0, y0, x1, y1 = union_bbox(alphas)
    side = int(np.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"x0": int(round(cx - side / 2)), "y0": int(round(cy - side / 2)), "side": side, "size": size,
            "src_hw": list(alphas[0].shape[:2])}


def _crop_padded(rgba: np.ndarray, x0, y0, w, h) -> np.ndarray:
    """Crop that may extend past the image; outside pixels are transparent white."""
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = 255
    H, W = rgba.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgba[sy0:sy1, sx0:sx1]
    return out


def apply_framing(rgba: np.ndarray, framing: dict) -> np.ndarray:
    """Source RGBA -> framed RGBA (size x size)."""
    crop = _crop_padded(rgba, framing["x0"], framing["y0"], framing["side"], framing["side"])
    s = framing["size"]
    return np.asarray(Image.fromarray(crop).resize((s, s), Image.LANCZOS))


def era3d_params(framed_alphas, crop_size: int) -> dict:
    """Union bbox over ALL frames of the case (framed coords) and the single scale mapping its longest side to crop_size."""
    x0, y0, x1, y1 = union_bbox(framed_alphas, thresh=0.5)
    return {"bbox": [x0, y0, x1, y1], "scale": float(crop_size / max(x1 - x0, y1 - y0)), "crop_size": crop_size}


def era3d_place(rgb: np.ndarray, alpha: np.ndarray, era3d: dict, canvas: int = 512):
    """Framed frame -> (rgb uint8 on white, alpha float) in the Era3D canvas, with the case-global scale.

    Mirrors the last step of Era3D's SingleImageDataset.load_image (crop, rescale, center with add_margin,
    composite on white) but with one crop and one scale for every frame of the case.
    """
    x0, y0, x1, y1 = era3d["bbox"]
    s = era3d["scale"]
    w, h = int(round((x1 - x0) * s)), int(round((y1 - y0) * s))
    rgba = np.concatenate([rgb[..., :3], (alpha * 255).round().astype(np.uint8)[..., None]], -1)
    crop = Image.fromarray(np.ascontiguousarray(rgba[y0:y1, x0:x1])).resize((w, h), Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
    out.paste(crop, ((canvas - w) // 2, (canvas - h) // 2))
    out = np.asarray(out).astype(np.float32) / 255.0
    a = out[..., 3]
    rgb_white = out[..., :3] * a[..., None] + (1 - a[..., None])
    return (rgb_white * 255).round().astype(np.uint8), a
