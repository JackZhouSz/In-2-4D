"""LTX-Video multi-condition backbone: one prompt plus images pinned to frame indices (start, keyframes, end)."""
import copy

import numpy as np
import torch
from PIL import Image

from .paths import LTX, resolve, use_repo


class LTXGenerator:
    def __init__(self, cfg):
        c = self.c = cfg.ltx
        self.fps = cfg.video.fps
        use_repo(LTX)
        from ltx_video.inference import create_latent_upsampler, load_pipeline_config
        from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
        from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
        from ltx_video.models.transformers.transformer3d import Transformer3DModel
        from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline, LTXVideoPipeline
        from ltx_video.schedulers.rf import RectifiedFlowScheduler
        from safetensors import safe_open
        from transformers import T5EncoderModel, T5Tokenizer

        ckdir = resolve(c.ckpt_dir)
        self.pcfg = load_pipeline_config(str(LTX / "configs" / c.pipeline_config))
        ckpt = ckdir / self.pcfg["checkpoint_path"]
        with safe_open(str(ckpt), framework="pt") as f:
            import json
            allowed = json.loads(f.metadata()["config"]).get("allowed_inference_steps")
        self.dev, self.dtype = torch.device(f"cuda:{c.device}"), getattr(torch, c.dtype)
        self.t5_dev = torch.device(c.t5_device)
        self.tokenizer = T5Tokenizer.from_pretrained(ckdir, subfolder="tokenizer")
        self.t5 = T5EncoderModel.from_pretrained(ckdir, subfolder="text_encoder",
                                                 torch_dtype=torch.float32 if c.t5_device == "cpu" else self.dtype)
        self.t5 = self.t5.to(self.t5_dev).eval()
        class Pipeline(LTXVideoPipeline):
            @property
            def _execution_device(self):
                return dev

        dev = torch.device(f"cuda:{c.device}")
        pipe = Pipeline(
            transformer=Transformer3DModel.from_pretrained(ckpt).to(self.dtype),
            patchifier=SymmetricPatchifier(patch_size=1), text_encoder=None, tokenizer=self.tokenizer,
            scheduler=RectifiedFlowScheduler.from_pretrained(ckpt),
            vae=CausalVideoAutoencoder.from_pretrained(ckpt).to(self.dtype),
            prompt_enhancer_image_caption_model=None, prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None, allowed_inference_steps=allowed,
        ).to(self.dev)
        self.base = pipe
        if self.pcfg.get("pipeline_type") == "multi-scale":
            up = create_latent_upsampler(str(ckdir / self.pcfg["spatial_upscaler_model_path"]), self.dev)
            pipe = LTXMultiScalePipeline(pipe, latent_upsampler=up)
        self.pipe = pipe

    @torch.no_grad()
    def encode(self, text: str):
        text = self.base._text_preprocessing(text)
        tok = self.tokenizer(text, padding="max_length", max_length=256, truncation=True, add_special_tokens=True,
                             return_tensors="pt")
        emb = self.t5(tok.input_ids.to(self.t5_dev), attention_mask=tok.attention_mask.to(self.t5_dev))[0]
        return emb.to(self.dev, self.dtype), tok.attention_mask.to(self.dev)

    def _image(self, rgb: np.ndarray, h: int, w: int):
        from ltx_video.inference import load_image_to_tensor_with_resize_and_crop
        return load_image_to_tensor_with_resize_and_crop(Image.fromarray(rgb), h, w)

    @torch.no_grad()
    def __call__(self, start_rgb: np.ndarray, end_rgb: np.ndarray, prompt: str, neg: str, frame_num: int, seed: int,
                 keyframes=()) -> np.ndarray:
        """RGB uint8 start/end (+ [(frame_index, rgb)] keyframes) -> (frame_num, H, W, 3) uint8."""
        from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

        c = self.c
        h, w = (int(np.ceil(s / 32)) * 32 for s in start_rgb.shape[:2])
        n = ((frame_num - 2) // 8 + 1) * 8 + 1
        items = [(0, start_rgb, 1.0)] + [(int(f), rgb, c.keyframe_strength) for f, rgb in keyframes] \
            + [(frame_num - 1, end_rgb, 1.0)]
        cond = [ConditioningItem(self._image(rgb, h, w).to(self.dev), f, s) for f, rgb, s in items]
        pe, pm = self.encode(prompt)
        ne, nm = self.encode(neg or "")
        pc = copy.deepcopy(self.pcfg)
        for k in ("checkpoint_path", "spatial_upscaler_model_path", "text_encoder_model_name_or_path", "precision",
                  "sampler", "prompt_enhancement_words_threshold", "prompt_enhancer_image_caption_model_name_or_path",
                  "prompt_enhancer_llm_model_name_or_path", "pipeline_type", "stg_mode"):
            pc.pop(k, None)
        out = self.pipe(**pc, skip_layer_strategy=SkipLayerStrategy.AttentionValues,
                        generator=torch.Generator(device=self.dev).manual_seed(seed), output_type="pt",
                        height=h, width=w, num_frames=n, frame_rate=self.fps, prompt=None, negative_prompt=None,
                        prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne,
                        negative_prompt_attention_mask=nm, conditioning_items=cond, is_video=True,
                        vae_per_channel_normalize=True, image_cond_noise_scale=c.image_cond_noise_scale,
                        mixed_precision=False, offload_to_cpu=c.offload, device=self.dev, enhance_prompt=False).images
        video = out[0, :, :frame_num].permute(1, 2, 3, 0)
        return (video.float().clamp(0, 1) * 255).round().byte().cpu().numpy()
