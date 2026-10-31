"""Novel-view videos of a trained case (in24d_4dgs env; needs Stage 6+ checkpoints).

  python render_video.py --cfg configs/t4x4_distilled.yaml --case flower
  ./run.sh video flower --azimuths 0 90 180 270 --elevations 0 30 --boomerang

Writes outputs/CASE/novel_view_videos/:
  grid.mp4   the animation from every (elevation, azimuth) camera at once, one labelled tile per camera
             (rows = elevations, columns = azimuths), optionally next to the input video (--with_input)
  orbit.mp4  one camera circling the object (--orbit_turns per play-through) while the animation plays
Cameras are orbit cameras around the origin as in Stage 9 (azimuth 0 = the input view, elevation up > 0);
the radius / FOV default to eval.radius / eval.fov_deg. Times are global frames resampled to --frames.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from core.config import load_cfg
from core.io import read_frames
from core.paths import CaseDirs
from pipeline.render_eval import SequenceRenderer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    p.add_argument("--azimuths", type=float, nargs="+", default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--elevations", type=float, nargs="+", default=[15.0])
    p.add_argument("--radius", type=float, default=None, help="default: eval.radius")
    p.add_argument("--fov", type=float, default=None, help="degrees; default: eval.fov_deg")
    p.add_argument("--size", type=int, default=320, help="tile size in pixels (multiple of 16)")
    p.add_argument("--frames", type=int, default=None, help="time samples; default: the case's global frames")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--boomerang", action="store_true", help="play forward then backward")
    p.add_argument("--with_input", action="store_true", help="add the Stage 2 input video as the first tile")
    p.add_argument("--orbit_turns", type=float, default=1.0, help="0 disables orbit.mp4")
    p.add_argument("--orbit_elevation", type=float, default=15.0)
    p.add_argument("--orbit_size", type=int, default=512)
    p.add_argument("--out", default=None, help="default: outputs/CASE/novel_view_videos")
    return p.parse_args()


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def label(tile: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(tile)
    ImageDraw.Draw(im).text((6, 4), text, fill=(60, 60, 60))
    return np.asarray(im)


def tile_grid(tiles, cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return canvas


def write_video(frames, path, fps):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as w:
        for f in frames:
            w.append_data(f)
    print(f"[video] {path}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


def input_tiles(dirs, size: int, times):
    """Stage 2 root clip (the generated input video), resampled to the render times."""
    clip = read_frames(dirs.s2 / "root" / "frames")
    out = []
    for p in times:
        f = clip[min(int(round(p)), len(clip) - 1)]
        out.append(label(np.asarray(Image.fromarray(f).convert("RGB").resize((size, size))), "input video"))
    return out


@torch.no_grad()
def main():
    a = parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dirs = CaseDirs(cfg.paths.outputs, a.case)
    out = dirs.videos if a.out is None else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    radius, fov = a.radius or cfg.eval.radius, a.fov or cfg.eval.fov_deg

    r = SequenceRenderer(dirs, cfg, load_data=False)
    n = a.frames or r.seq.total
    times = [i / max(n - 1, 1) * (r.seq.total - 1) for i in range(n)]
    order = list(range(n)) + (list(range(n - 2, 0, -1)) if a.boomerang else [])

    views = [(el, az) for el in a.elevations for az in a.azimuths]
    grid = [[] for _ in range(n)]
    r.reset_smoothing()
    for i, p in enumerate(times):
        k, local, m, st, blend = r.state_at(p)
        for el, az in views:
            cam = r.orbit_camera(m, local, az, el, radius, fov, a.size)
            grid[i].append(label(to_uint8(r.draw(cam, m, st, blend)), f"az {az:g}  el {el:g}"))
    cols = len(a.azimuths)
    if a.with_input:
        blank = np.full((a.size, a.size, 3), 255, np.uint8)
        for i, t in enumerate(input_tiles(dirs, a.size, times)):
            rows = [grid[i][j * cols:(j + 1) * cols] for j in range(len(a.elevations))]
            grid[i] = [x for j, row in enumerate(rows) for x in [t if j == 0 else blank] + row]
        cols += 1
    write_video([tile_grid(grid[i], cols) for i in order], out / "grid.mp4", a.fps)

    if a.orbit_turns > 0:
        frames = []
        r.reset_smoothing()
        for j, i in enumerate(order):
            k, local, m, st, blend = r.state_at(times[i])
            az = 360.0 * a.orbit_turns * j / len(order)
            cam = r.orbit_camera(m, local, az, a.orbit_elevation, radius, fov, a.orbit_size)
            frames.append(to_uint8(r.draw(cam, m, st, blend)))
        write_video(frames, out / "orbit.mp4", a.fps)


if __name__ == "__main__":
    main()
