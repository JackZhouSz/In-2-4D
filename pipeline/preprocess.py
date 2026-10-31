"""Stage 0 — alpha + one global square framing for the whole case.

Outputs: video/inputs/{start.png, end.png} (RGBA, framed, framing.size^2), video/inputs/framing.json
"""
import numpy as np
from PIL import Image

from core import framing
from core.io import write_image, write_json
from core.matting import matte_image
from core.stage import StageContext, stage_parser


def load_rgba(path, cfg, size_hw=None) -> np.ndarray:
    img = Image.open(path)
    if size_hw is not None and img.size != (size_hw[1], size_hw[0]):
        print(f"[s0] resizing {path} from {img.size} to {(size_hw[1], size_hw[0])} to share one framing")
        img = img.resize((size_hw[1], size_hw[0]), Image.LANCZOS)
    if img.mode == "RGBA" and np.asarray(img)[..., 3].min() < 255:
        return np.asarray(img).copy()
    rgb = np.asarray(img.convert("RGB"))
    alpha = matte_image(rgb, cfg)
    return np.concatenate([rgb, (alpha * 255).round().astype(np.uint8)[..., None]], -1)


def main():
    p = stage_parser(__doc__)
    p.add_argument("--start", required=True, help="start state image I_s")
    p.add_argument("--end", required=True, help="end state image I_e")
    ctx = StageContext(p.parse_args(), 0)
    if ctx.done(ctx.dirs.framing_json):
        return

    start = load_rgba(ctx.args.start, ctx.cfg)
    end = load_rgba(ctx.args.end, ctx.cfg, size_hw=start.shape[:2])
    alphas = [start[..., 3] / 255.0, end[..., 3] / 255.0]
    fr = framing.square_framing(alphas, ctx.cfg.framing.margin, ctx.cfg.framing.size)
    fr["inputs"] = {"start": str(ctx.args.start), "end": str(ctx.args.end)}

    write_image(framing.apply_framing(start, fr), ctx.out / "start.png")
    write_image(framing.apply_framing(end, fr), ctx.out / "end.png")
    write_json(fr, ctx.dirs.framing_json)
    print(f"[s0] framing {fr}")


if __name__ == "__main__":
    main()
