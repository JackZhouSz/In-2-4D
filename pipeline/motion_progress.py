"""Stage 3 — diffusion features, motion progress curve and segmented least squares on the root video.

Outputs: video/motion_progress/root/{features.pt, dist.npz, progress.npy, progress_raw.npy, segments.json, progress_plot.png}
`ProgressAnalyzer` is reused by Stage 4 for every regenerated clip.
"""
import numpy as np
import torch

from core import progress
from core.io import read_frames, read_masks, write_json
from core.stage import StageContext, stage_parser
from patches.diffusion_features import DiffusionFeatures


class ProgressAnalyzer:
    def __init__(self, cfg):
        d = cfg.features
        self.cfg = cfg
        self.extractor = DiffusionFeatures(d.sd_id, t=d.t, up_ft_index=d.up_ft_index, ensemble=d.ensemble, res=d.res,
                               seed=d.seed, dtype=d.dtype, chunk=d.chunk)
        self.siglip = None
        if cfg.progress.w_local < 1.0:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(cfg.progress.siglip_model,
                                                                         pretrained=cfg.progress.siglip_pretrained)
            self.siglip = (model.cuda().eval(), preprocess)

    def _siglip(self, frames):
        from PIL import Image
        model, preprocess = self.siglip
        x = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).cuda()
        with torch.no_grad():
            z = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
        return z.cpu().numpy()

    def analyze(self, frames: np.ndarray, alphas: np.ndarray) -> dict:
        c = self.cfg
        self.extractor.to(self.extractor.device)
        try:
            feats = [self.extractor(f) for f in frames]
        finally:
            if c.features.offload:
                self.extractor.to("cpu")
                torch.cuda.empty_cache()
        tokens = [progress.foreground_tokens(f, a) for f, a in zip(feats, alphas)]
        glob = self._siglip(frames) if self.siglip else None
        pairs, sims, d = progress.pair_distances(tokens, c.progress.pair_window, c.progress.p, glob, c.progress.w_local)
        S_raw = progress.solve(len(frames), pairs, d, c.progress.sigma, c.progress.lam)
        S = progress.monotonize(S_raw)
        bounds = progress.segmented_least_squares(S, c.seg.c_pen)
        slopes, dS = progress.segment_stats(S, bounds)
        adjacent = {int(i): float(s) for (i, j), s in zip(pairs, sims) if i - j == 1}
        return {"feats": feats, "pairs": pairs, "sims": sims, "dist": d, "S_raw": S_raw, "S": S,
                "boundaries": bounds, "slopes": slopes, "dS": dS, "adjacent_sim": adjacent,
                "lin": progress.linearity(S), "r2": progress.r2_vs_ramp(S)}


def save_analysis(out_dir, res: dict, save_feats=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_feats:
        torch.save(torch.stack(res["feats"]).half().cpu(), out_dir / "features.pt")
    np.savez(out_dir / "dist.npz", pairs=res["pairs"], sims=res["sims"], dist=res["dist"])
    np.save(out_dir / "progress_raw.npy", res["S_raw"])
    np.save(out_dir / "progress.npy", res["S"])
    _plot(out_dir / "progress_plot.png", res)
    write_json({"boundaries": res["boundaries"], "slopes": res["slopes"], "dS": res["dS"], "lin": res["lin"],
                "r2": res["r2"], "adjacent_sim": res["adjacent_sim"]}, out_dir / "segments.json")
    print(f"[s3] {out_dir.name}: boundaries {res['boundaries']}  lin {res['lin']:.3f}  R2 {res['r2']:.3f}")


def load_analysis(out_dir) -> dict:
    from core.io import read_json
    seg = read_json(out_dir / "segments.json")
    dist = np.load(out_dir / "dist.npz")
    return {**seg, "S": np.load(out_dir / "progress.npy"), "S_raw": np.load(out_dir / "progress_raw.npy"),
            "pairs": dist["pairs"], "sims": dist["sims"], "dist": dist["dist"],
            "adjacent_sim": {int(k): v for k, v in seg["adjacent_sim"].items()}}


def _plot(path, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    S, b = res["S"], res["boundaries"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res["S_raw"], color="0.7", label="S (raw)")
    ax.plot(S, label="S (monotone)")
    ax.plot(b, S[b], "o-", color="C3", label="segments")
    ax.plot([0, len(S) - 1], [0, 1], ":", color="0.5", label="linear")
    ax.set_xlabel("frame"), ax.set_ylabel("semantic progress"), ax.legend()
    ax.set_title(f"lin={res['lin']:.3f}  R2={res['r2']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 3)
    out = ctx.out / "root"
    if ctx.done(out / "segments.json"):
        return
    clip = ctx.dirs.s2 / "root"
    res = ProgressAnalyzer(ctx.cfg).analyze(read_frames(clip / "frames"), read_masks(clip / "alpha"))
    save_analysis(out, res)


if __name__ == "__main__":
    main()
