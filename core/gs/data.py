"""Per-fragment multi-view video in Blender-style (transforms_*.json) format.

View 0 (front) is the Wan fragment frame, framed with the case-global Era3D crop/scale; views 1-5 are
Era3D's outputs from Stage 5. With merge.overlap_frames > 1, non-last fragments also get the first
overlap-1 frames of the next fragment, reference view only.
"""
import numpy as np
import torch

from core.cameras import ERA3D_VIEWS, era3d_camera, far_c2w, far_fov, write_transforms
from core.framing import era3d_place
from core.io import read_frames, read_image, read_json, read_masks, to_rgba, write_image


def write_dataset(dirs, frags, k: int, cfg, out_dir):
    """Build 4d/reconstruction/frag_XX/data/: images/tTTT_vV.png (RGBA) + transforms_{train,test}.json."""
    frag = frags[k]
    era3d = read_json(dirs.framing_json)["era3d"]
    src = dirs.root / frag["dir"]
    frames, alphas = read_frames(src / "frames"), read_masks(src / "alpha")
    T = len(frames)
    extra = []
    if cfg.merge.overlap_frames > 1 and k + 1 < len(frags):
        nxt = dirs.root / frags[k + 1]["dir"]
        nf, na = read_frames(nxt / "frames"), read_masks(nxt / "alpha")
        extra = [(nf[m], na[m]) for m in range(1, cfg.merge.overlap_frames)]
    T_train = T + len(extra)
    D, size = cfg.gs.D_far, cfg.era3d.img_size
    mv = dirs.frag_dir(5, frag["id"]) / "mv"

    records = []
    for t in range(T_train):
        time = t / (T_train - 1)
        views = range(len(ERA3D_VIEWS)) if t < T else [0]
        for v in views:
            if v == 0:
                rgb_src, a_src = (frames[t], alphas[t]) if t < T else extra[t - T]
                rgb, a = era3d_place(rgb_src, a_src, era3d, size)
            else:
                rgb, a = read_image(mv / f"t{t:03d}_v{v}.png", "RGB"), read_image(mv / f"t{t:03d}_v{v}_alpha.png", "L") / 255.0
            name = f"images/t{t:03d}_v{v}"
            write_image(to_rgba(rgb, a), out_dir / f"{name}.png")
            records.append({"file_path": name, "transform_matrix": far_c2w(ERA3D_VIEWS[v], D).tolist(),
                            "time": time, "time_index": t, "view_index": v})
    write_transforms(out_dir / "transforms_train.json", records, far_fov(D))
    write_transforms(out_dir / "transforms_test.json", [r for r in records if r["view_index"] == 0], far_fov(D))
    return {"num_frames": T, "T_train": T_train}


class FragmentData:
    """Images as GPU tensors: rgb[t][v] (3, H, W) on white, alpha[t][v] (1, H, W); None where unavailable."""

    def __init__(self, data_dir, cfg, device="cuda"):
        meta = read_json(data_dir / "transforms_train.json")
        self.T = 1 + max(f["time_index"] for f in meta["frames"])
        V = len(ERA3D_VIEWS)
        self.rgb = [[None] * V for _ in range(self.T)]
        self.alpha = [[None] * V for _ in range(self.T)]
        for f in meta["frames"]:
            rgba = read_image(data_dir / f"{f['file_path']}.png", "RGBA").astype(np.float32) / 255.0
            t, v = f["time_index"], f["view_index"]
            self.rgb[t][v] = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).to(device)
            self.alpha[t][v] = torch.from_numpy(rgba[..., 3:]).permute(2, 0, 1).to(device)
        size = self.rgb[0][0].shape[-1]
        self.cams = [era3d_camera(v, cfg.gs.D_far, size, device=device) for v in ERA3D_VIEWS]

    def time(self, t: float) -> float:
        return t / (self.T - 1)

    def cam(self, v: int, t: float):
        return self.cams[v].at_time(self.time(t))

    def views_at(self, t: int):
        return [v for v in range(len(self.cams)) if self.rgb[t][v] is not None]
