"""Wan2.2-Lightning (lightx2v): 4-step distilled LoRA for Wan2.2-I2V-A14B, no CFG pass.

Reproduces the two pieces of the ModelTC/Wan2.2-Lightning fork that matter for inference:
  * `merge_lora`: wan/utils/utils.py::load_and_merge_lora_weight (W += alpha/rank * up @ down, blocks only),
    computed in fp32 and cast back to the expert's dtype;
  * `EulerScheduler`: wan/utils/fm_solvers_euler.py, the solver the LoRA was distilled with
    (timesteps linspace(1000, 0, steps+1) shifted, x <- x + (sigma_next - sigma) * v).
The CFG skip (guide_scale == 1 -> one model call per step) lives in patches/video_flf.generate_flf.
"""
from pathlib import Path

import torch
from safetensors import safe_open


@torch.no_grad()
def merge_lora(model: torch.nn.Module, path, strength: float = 1.0) -> int:
    """Merge a Wan-native LoRA file (`diffusion_model.<param>.lora_{up,down}.weight`, `.alpha`) into `model`."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}
    prefix = "diffusion_model." if any(k.startswith("diffusion_model.") for k in lora) else ""
    merged = 0
    for name, p in model.named_parameters():
        base = prefix + name[: -len(".weight")] if name.endswith(".weight") else None
        if base is None or base + ".lora_down.weight" not in lora:
            continue
        down, up = lora[base + ".lora_down.weight"].float(), lora[base + ".lora_up.weight"].float()
        scale = float(lora[base + ".alpha"]) / down.shape[0] if base + ".alpha" in lora else 1.0
        delta = (up @ down) * (scale * strength)
        p.copy_((p.float() + delta.to(p.device)).to(p.dtype))
        merged += 1
    n_lora = sum(k.endswith(".lora_down.weight") for k in lora)
    assert merged == n_lora, f"{Path(path).name}: merged {merged} of {n_lora} LoRA layers (key mismatch)"
    return merged


def merge_into_wan(wan_i2v, lora_dir, strength: float = 1.0):
    """Both experts of a WanI2V (call before they are placed on GPUs)."""
    for expert in ("high_noise_model", "low_noise_model"):
        n = merge_lora(getattr(wan_i2v, expert), Path(lora_dir) / f"{expert}.safetensors", strength)
        print(f"[lightning] merged {n} LoRA layers into {expert}")


class EulerScheduler:
    """Flow-matching Euler with the fork's shifted uniform timesteps; same step() call shape as Wan's solvers."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.T, self.shift = num_train_timesteps, shift

    def set_timesteps(self, steps: int, device=None):
        s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
        self.sigmas = (self.shift * s / (1 + (self.shift - 1) * s)).to(device)
        self.timesteps = self.sigmas[:-1] * self.T
        self._i = 0

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        sigma, sigma_next = self.sigmas[self._i], self.sigmas[self._i + 1]
        self._i += 1
        return (sample.float() + (sigma_next - sigma) * model_output.float(),)
