"""Image metrics shared by several stages: masked LPIPS and sharpness."""
import numpy as np

_LPIPS = {}


def lpips_model(device="cuda", net="alex"):
    import lpips

    key = (device, net)
    if key not in _LPIPS:
        _LPIPS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LPIPS[key]


def to_lpips_tensor(rgb: np.ndarray, device):
    import torch

    x = torch.from_numpy(rgb[..., :3].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    return x.to(device)


def masked_lpips(a: np.ndarray, b: np.ndarray, mask: np.ndarray, device="cuda") -> float:
    """LPIPS restricted to `mask`: both images are set to white outside the mask, then cropped to its bbox."""
    import torch

    m = mask > 0.5
    if not m.any():
        return 0.0
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ims = []
    for img in (a, b):
        img = np.where(m[..., None], img[..., :3], 255).astype(np.uint8)[y0:y1, x0:x1]
        ims.append(to_lpips_tensor(img, device))
    with torch.no_grad():
        return float(lpips_model(device)(ims[0], ims[1]).item())


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image (higher = sharper)."""
    from scipy import ndimage

    grey = rgb[..., :3].astype(np.float32).mean(-1)
    return float(ndimage.laplace(grey).var())


def mask_iou(a: np.ndarray, b: np.ndarray, thresh=0.5) -> float:
    a, b = a > thresh, b > thresh
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
