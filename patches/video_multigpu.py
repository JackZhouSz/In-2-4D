"""Running Wan2.2-I2V-A14B on small GPUs without flash-attn (e.g. 4x Tesla T4, 15 GB, sm75).

* `sdpa_attention` replaces wan.modules.model.flash_attention (flash-attn 2 needs sm80+). Upstream calls it
  with k_lens = full sequence length for self-attention and context_lens=None for cross-attention, so no
  mask is needed in practice; a key-padding mask is still applied if shorter k_lens ever arrive.
* `place` splits one expert's 40 transformer blocks over several GPUs (embeddings/head on the first);
  forward pre-hooks move a module's tensor inputs to its device, so WanModel.forward runs unchanged.
* `enable_split` replaces WanI2V._prepare_model_for_timestep: the needed expert is placed across the GPUs
  and the idle one waits in CPU RAM (one swap per generation, at the high/low-noise boundary).
"""
import types

import torch
import torch.nn.functional as F


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                   causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.float16, version=None):
    """Same signature and layout as wan.modules.attention.flash_attention: q/k/v [B, L, N, C]."""
    assert tuple(window_size) == (-1, -1), "sliding-window attention is not supported by the SDPA fallback"
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    mask = None
    lk = k.size(1)
    if k_lens is not None and int(k_lens.min()) < lk:
        mask = (torch.arange(lk, device=q.device)[None] < k_lens.to(q.device)[:, None])[:, None, None, :]
    half = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
    q, k, v = (x.transpose(1, 2).to(half) for x in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=causal,
                                         scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def install_sdpa(dtype=torch.float16):
    import wan.modules.model as model

    model.flash_attention = lambda *a, **kw: sdpa_attention(*a, **{**kw, "dtype": dtype})



def _to_module_device(module, args, kwargs):
    dev = next(module.parameters()).device
    mv = lambda t: t.to(dev) if torch.is_tensor(t) and t.is_cuda and t.device != dev else t
    return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}


def block_assignment(n_blocks: int, n_devices: int, split=None):
    split = list(split) if split else [n_blocks // n_devices + (i < n_blocks % n_devices) for i in range(n_devices)]
    assert len(split) == n_devices and sum(split) == n_blocks, f"block_split {split} must sum to {n_blocks}"
    return [d for d, n in enumerate(split) for _ in range(n)]


def place(model, devices, split=None):
    """Put a WanModel on several GPUs: blocks by `split`, everything else on devices[0]."""
    devs = [torch.device(f"cuda:{d}") if isinstance(d, int) else torch.device(d) for d in devices]
    for name, child in model.named_children():
        if name != "blocks":
            child.to(devs[0])
    for block, d in zip(model.blocks, block_assignment(len(model.blocks), len(devs), split)):
        block.to(devs[d])
    if not getattr(model, "_in24d_hooks", False):
        for module in [*model.blocks, model.head]:
            module.register_forward_pre_hook(_to_module_device, with_kwargs=True)
        model._in24d_hooks = True
    return model


def _prepare_split(self, t, boundary, offload_model):
    """Replacement for WanI2V._prepare_model_for_timestep (same selection rule as upstream)."""
    name = "high_noise_model" if t.item() >= boundary else "low_noise_model"
    if self._in24d_placed != name:
        self.release()
        place(getattr(self, name), self._in24d_devices, self._in24d_split)
        self._in24d_placed = name
    return getattr(self, name)


def _release(self):
    """Move both experts back to CPU RAM (frees the GPUs for feature extraction / Era3D between generations)."""
    self.low_noise_model.to("cpu")
    self.high_noise_model.to("cpu")
    self._in24d_placed = None
    torch.cuda.empty_cache()


def enable_split(wan_i2v, devices, split=None):
    wan_i2v._in24d_devices, wan_i2v._in24d_split, wan_i2v._in24d_placed = list(devices), split, None
    wan_i2v._prepare_model_for_timestep = types.MethodType(_prepare_split, wan_i2v)
    wan_i2v.release = types.MethodType(_release, wan_i2v)
    return wan_i2v
