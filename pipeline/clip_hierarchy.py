"""Stage 4 — keyframes, optional retiming, hierarchical fragment regeneration.

hier.mode: fragments (Wan: split at keyframes, one clip per segment) | pinned (multi-condition backbones such as
LTX: one clip with every keyframe pinned at its linear-progress frame, always a single fragment).

Outputs: video/clips/{fragments.json, kf_XXX.png, frag_XX/{frames,alpha}/%04d.png + meta.json,
         nodes/<node>/} (cached clip + progress curve per tree node, makes the stage resumable),
         video/inputs/framing.json gains "era3d" (one bbox + one scale for the whole case).
"""
import shutil

import numpy as np

from core import framing, progress
from core.io import (composite_white, read_frames, read_image, read_json, read_masks, to_rgba, write_frames,
                             write_image, write_json)
from core.matting import matte_image, release_rembg
from core.metrics import masked_lpips, sharpness
from core.stage import StageContext, stage_parser
from patches.video_rope_retime import RETIME, band_alphas
from pipeline.video_generation import make_generator, check_flf, generate_clip, save_clip
from pipeline.motion_progress import ProgressAnalyzer, load_analysis, save_analysis


class Clip:
    def __init__(self, frames, alphas, res):
        self.frames, self.alphas, self.res = frames, alphas, res

    def rgba(self, k):
        return to_rgba(self.frames[k], self.alphas[k])


class Hierarchy:
    def __init__(self, ctx, prompt, neg):
        self.ctx, self.cfg, self.prompt, self.neg = ctx, ctx.cfg, prompt, neg
        self.I_s = read_image(ctx.dirs.s0 / "start.png", "RGBA")
        self.I_e = read_image(ctx.dirs.s0 / "end.png", "RGBA")
        self._gen = self._analyzer = None

    @property
    def gen(self):
        self._gen = self._gen or make_generator(self.cfg)
        return self._gen

    @property
    def analyzer(self):
        self._analyzer = self._analyzer or ProgressAnalyzer(self.cfg)
        return self._analyzer

    def _load(self, d):
        if (d / "meta.json").exists() and (d / "segments.json").exists() and not self.ctx.args.force:
            return Clip(read_frames(d / "frames"), read_masks(d / "alpha"), load_analysis(d))
        return None

    def _generate(self, node, start, end, T, keyframes=()):
        frames, alphas, fid = generate_clip(self.gen, start, end, self.prompt, self.neg, T, self.cfg.video.seed,
                                            self.cfg, keyframes)
        check_flf(self.ctx.check, f"flf_fidelity_{node}", fid, self.cfg)
        return frames, alphas, fid

    def clip(self, node, start, end, T, keyframes=()) -> Clip:
        d = self.ctx.out / "nodes" / node
        cached = self._load(d)
        if cached is not None:
            return cached
        if node == "r" and (self.ctx.dirs.s2 / "root" / "meta.json").exists():
            src2, src3 = self.ctx.dirs.s2 / "root", self.ctx.dirs.s3 / "root"
            c = Clip(read_frames(src2 / "frames"), read_masks(src2 / "alpha"), None)
            c.res = load_analysis(src3) if (src3 / "segments.json").exists() else self.analyzer.analyze(c.frames, c.alphas)
            meta = read_json(src2 / "meta.json")
        else:
            frames, alphas, fid = self._generate(node, start, end, T, keyframes)
            c = Clip(frames, alphas, self.analyzer.analyze(frames, alphas))
            meta = {"frame_num": T, "seed": self.cfg.video.seed, "pins": [f for f, _ in keyframes], **fid}
        if self.cfg.retime.enabled and self.cfg.video.backbone == "wan" and (node == "r" or self.cfg.retime.apply_to == "all"):
            c, meta["retime"] = self.retime(node, start, end, T, c)
        save_clip(d, c.frames, c.alphas, meta, self.cfg.video.fps)
        save_analysis(d, c.res, save_feats=False)
        return c

    def retime(self, node, start, end, T, clip: Clip):
        """Warp Wan's temporal RoPE until the progress curve is linear (max `iters` regenerations)."""
        rc = self.cfg.retime
        alphas = band_alphas(rc.alpha_low, rc.alpha_high, rc.kappa)
        RETIME.reset(T)
        log = [{"iter": 0, "lin": clip.res["lin"]}]
        try:
            for n in range(1, rc.iters + 1):
                if clip.res["lin"] >= self.cfg.hier.lin_target:
                    break
                delta = progress.inverse(clip.res["S"], np.linspace(0, 1, T)) - np.arange(T)
                RETIME.update(delta, alphas)
                RETIME.enabled = True
                frames, alphas_v, _ = self._generate(f"{node}_retime{n}", start, end, T)
                RETIME.enabled = False
                clip = Clip(frames, alphas_v, self.analyzer.analyze(frames, alphas_v))
                log.append({"iter": n, "lin": clip.res["lin"]})
        finally:
            RETIME.enabled = False
        print(f"[s4] retime {node}: {log}")
        return clip, log

    def keyframe_candidates(self, clip: Clip):
        T = len(clip.frames)
        if self.cfg.keyframe_mode == "progress_segments":
            return list(clip.res["boundaries"][1:-1])
        if self.cfg.keyframe_mode == "feature_threshold":
            low = [j for j in range(1, T) if clip.res["adjacent_sim"][j] < self.cfg.tau_feat]
            runs, cands = [], []
            for j in low:
                if runs and runs[-1][-1] == j - 1:
                    runs[-1].append(j)
                else:
                    runs.append([j])
            for run in runs:
                cands.append(min(run, key=lambda j: clip.res["adjacent_sim"][j]))
            return cands
        raise ValueError(f"unknown keyframe_mode '{self.cfg.keyframe_mode}'")

    def refine_keyframes(self, clip: Clip):
        """Search [b-r, b+r] for the frame most faithful to the inputs, sharpness as tie-break."""
        T, r = len(clip.frames), self.cfg.refine_radius
        sign = 1.0 if self.cfg.keyframe_fidelity_sign == "best" else -1.0
        ref_s, ref_e = composite_white(self.I_s), composite_white(self.I_e)
        a_s, a_e = self.I_s[..., 3] / 255.0, self.I_e[..., 3] / 255.0
        chosen = []
        for b in self.keyframe_candidates(clip):
            scored = []
            for f in range(max(1, b - r), min(T - 2, b + r) + 1):
                fr, al = clip.frames[f], clip.alphas[f]
                score = min(masked_lpips(fr, ref_s, np.maximum(al, a_s)), masked_lpips(fr, ref_e, np.maximum(al, a_e)))
                scored.append((round(sign * score, 3), -sharpness(fr), f))
            if scored:
                chosen.append(min(scored)[2])
        return [0] + sorted(set(chosen) - {0, T - 1}) + [T - 1]

    def keyframe_rgba(self, clip: Clip, k):
        if self.cfg.hier.rematte_keyframes:
            return to_rgba(clip.frames[k], matte_image(clip.frames[k], self.cfg))
        return clip.rgba(k)

    def build_pinned(self, node, start, end, T, depth):
        """Multi-condition backbones: keyframes pinned where linear progress puts them, one clip, one fragment."""
        h = self.cfg.hier
        clip = self.clip(node, start, end, T)
        log = [{"node": node, "lin": clip.res["lin"]}]
        while depth < h.max_depth and clip.res["lin"] < h.lin_target:
            kfs = self.refine_keyframes(clip)[1:-1]
            if not kfs:
                break
            S, n = clip.res["S"], len(clip.frames)
            pins, last = [], 0
            for b in kfs:
                f = int(round((T - 1) * (S[b] - S[0]) / (S[-1] - S[0])))
                f = int(round(f / h.pin_grid)) * h.pin_grid
                if last < f < T - 1:
                    pins.append((f, self.keyframe_rgba(clip, b)))
                    last = f
            if not pins:
                break
            release_rembg()
            depth += 1
            node = f"{node}.p{depth}"
            clip = self.clip(node, start, end, T, keyframes=pins)
            log.append({"node": node, "lin": clip.res["lin"], "pins": [f for f, _ in pins]})
        print(f"[s4] pinned keyframes: {log}")
        return [self._leaf(node, clip, start, end, depth)]

    def build(self, node, start, end, T, depth):
        h = self.cfg.hier
        clip = self.clip(node, start, end, T)
        S, segs = clip.res["S"], clip.res["boundaries"]
        if depth == h.max_depth or clip.res["lin"] >= h.lin_target or len(segs) == 2:
            return [self._leaf(node, clip, start, end, depth)]
        kfs = self.refine_keyframes(clip)
        if len(kfs) == 2:
            return [self._leaf(node, clip, start, end, depth)]
        leaves = []
        for i, (a, b) in enumerate(zip(kfs[:-1], kfs[1:])):
            T_k = int(np.clip(progress.round_to_4n1(h.T_total * (S[b] - S[a])), h.T_min, h.T_max))
            s_img = start if a == 0 else self.keyframe_rgba(clip, a)
            e_img = end if b == len(clip.frames) - 1 else self.keyframe_rgba(clip, b)
            leaves += self.build(f"{node}.{i}", s_img, e_img, T_k, depth + 1)
        return leaves

    @staticmethod
    def _leaf(node, clip, start, end, depth):
        """Boundary frames are replaced by the exact conditioning keyframes so neighbours share them."""
        frames, alphas = clip.frames.copy(), clip.alphas.copy()
        frames[0], alphas[0] = composite_white(start), start[..., 3] / 255.0
        frames[-1], alphas[-1] = composite_white(end), end[..., 3] / 255.0
        return {"node": node, "depth": depth, "frames": frames, "alphas": alphas, "start": start, "end": end,
                "lin": clip.res["lin"]}


def write_fragments(ctx, leaves):
    out = ctx.out
    for old in list(out.glob("frag_*")) + list(out.glob("kf_*.png")):
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    entries, g = [], 0
    rel = out.relative_to(ctx.dirs.root).as_posix()
    for i, leaf in enumerate(leaves):
        n = len(leaf["frames"])
        d = out / f"frag_{i:02d}"
        write_frames(leaf["frames"], d / "frames", leaf["alphas"], d / "alpha")
        write_image(leaf["start"], out / f"kf_{i:03d}.png")
        write_image(leaf["end"], out / f"kf_{i + 1:03d}.png")
        e = {"id": i, "global_start": g, "global_end": g + n - 1, "num_frames": n, "canonical_local_idx": 0,
             "start_keyframe": f"{rel}/kf_{i:03d}.png", "end_keyframe": f"{rel}/kf_{i + 1:03d}.png",
             "dir": f"{rel}/frag_{i:02d}", "node": leaf["node"], "depth": leaf["depth"], "lin": leaf["lin"]}
        write_json(e, d / "meta.json")
        entries.append(e)
        g += n - 1
    for a, b in zip(entries[:-1], entries[1:]):
        assert a["global_end"] == b["global_start"]
    return {"fps": ctx.cfg.video.fps, "total_frames": g + 1, "fragments": entries}


def main():
    ctx = StageContext(stage_parser(__doc__).parse_args(), 4)
    if ctx.done(ctx.dirs.fragments_json):
        return
    cap = read_json(ctx.dirs.s1 / "caption.json")
    hier = Hierarchy(ctx, cap["prompt"], cap["negative_prompt"])
    build = hier.build_pinned if ctx.cfg.hier.mode == "pinned" else hier.build
    leaves = build("r", hier.I_s, hier.I_e, ctx.cfg.hier.T_total, depth=0)
    frags = write_fragments(ctx, leaves)

    fr = read_json(ctx.dirs.framing_json)
    fr["era3d"] = framing.era3d_params([a for leaf in leaves for a in leaf["alphas"]], ctx.cfg.era3d.crop_size)
    write_json(fr, ctx.dirs.framing_json)
    ctx.check("framing_single_era3d_scale", isinstance(fr["era3d"]["scale"], float), **fr["era3d"])

    write_json(frags, ctx.dirs.fragments_json)
    print(f"[s4] {len(leaves)} fragments, {frags['total_frames']} frames: "
          f"{[(f['global_start'], f['global_end']) for f in frags['fragments']]}")


if __name__ == "__main__":
    main()
