"""Spatial KV-latent blending with EMA in Era3D's attn1.

`patched_forward` is a verbatim copy of
Era3D/mvdiffusion/models/transformer_mv2d_self_rowwise.py::BasicMVTransformerBlock.forward
where ONLY the attn1 call changes: in 'inject' mode, K and V come from norm1(blend) with
blend = s * cache[idx] + (1 - s) * hidden_states; Q and the residual are untouched.
Never touches attn_mv (cross-view attention) or attn2 (cross-attention).
"""
import math

import torch


class Inj:
    mode = 'off'
    s = 0.5
    ema = True
    cache, idx = [], 0
    step, active_steps = 0, None

    @classmethod
    def start_frame(cls, mode, num_steps, steps_frac):
        cls.mode, cls.idx, cls.step = mode, 0, 0
        cls.active_steps = math.ceil(num_steps * steps_frac)
        if mode == 'record':
            cls.cache = []

    @classmethod
    def step_callback(cls, i, t, latents):
        """Pipeline callback, called after denoising step i."""
        cls.step = i + 1

    @classmethod
    def active(cls):
        return cls.mode in ('record', 'inject') and cls.step < cls.active_steps


def _norm1(blk, h, timestep, class_labels):
    if blk.use_ada_layer_norm:
        return blk.norm1(h, timestep)
    if blk.use_ada_layer_norm_zero:
        return blk.norm1(h, timestep, class_labels, hidden_dtype=h.dtype)[0]
    return blk.norm1(h)


def patched_forward(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    timestep=None,
    cross_attention_kwargs=None,
    class_labels=None,
    dino_feature=None
):
    assert attention_mask is None
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

    kv = None
    if Inj.active() and Inj.mode == 'record':
        Inj.cache.append(hidden_states.detach().to('cpu', torch.float16)); Inj.idx += 1
    elif Inj.active() and Inj.mode == 'inject':
        ref = Inj.cache[Inj.idx].to(hidden_states.device, hidden_states.dtype)
        assert ref.shape == hidden_states.shape, (ref.shape, hidden_states.shape)
        blend = Inj.s * ref + (1 - Inj.s) * hidden_states
        if Inj.ema:
            Inj.cache[Inj.idx] = blend.detach().to('cpu', torch.float16)
        kv = _norm1(self, blend, timestep, class_labels); Inj.idx += 1
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=kv if kv is not None else
            (encoder_hidden_states if self.only_cross_attention else None),
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    if self.rowwise_attention:
        norm_hidden_states = (
            self.norm_mv(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_mv(hidden_states)
        )
        attn_output = self.attn_mv(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            num_views=self.num_views,
            multiview_attention=self.multiview_attention,
            cd_attention_mid=self.cd_attention_mid,
            **cross_attention_kwargs,
            )
        hidden_states = attn_output + hidden_states

    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self._chunk_size is not None:
        if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
            raise ValueError(
                f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
            )

        num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
        ff_output = torch.cat(
            [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
            dim=self._chunk_dim,
        )
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states


def install():
    """Call before building the pipeline."""
    from mvdiffusion.models.transformer_mv2d_self_rowwise import BasicMVTransformerBlock

    BasicMVTransformerBlock.forward = patched_forward
