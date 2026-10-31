#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="$HERE/checkpoints"
DISTILLED=0
LTX=0
for arg in "$@"; do
  case "$arg" in
    --distilled) DISTILLED=1 ;;
    --ltx) LTX=1 ;;
    -h|--help) cat <<'EOF'
Downloads every pretrained model the pipeline uses (files already present are skipped).
  bash download_weights.sh                 required models
  bash download_weights.sh --distilled     + the 4-step Wan2.2-Lightning LoRA (configs/t4x4_distilled.yaml)
  bash download_weights.sh --ltx           + LTX-Video 2B distilled, the alternative video backbone (configs/t4x4_ltx.yaml)
Needs the conda envs (./run.sh setup).
  checkpoints/                   Wan2.2-I2V-A14B, SAM 2.1, Wan2.2-Lightning, LTX-Video
  ~/.cache/huggingface/hub       Qwen2.5-VL-7B, Stable Diffusion 2.1, Era3D, Depth-Anything-V2
  ~/.cache/torch/hub             CoTracker3, LPIPS backbones
  ~/.u2net                       rembg matting models
EOF
      exit 0 ;;
    *) echo "unknown option $arg" >&2; exit 1 ;;
  esac
done

in_env() { local env="$1"; shift; conda run --no-capture-output -n "$env" "$@"; }
hf_get() { in_env in24d_wan hf download "$@"; }
step() { echo; echo "=== $*"; }

step "Wan2.2-I2V-A14B (video generation, ~126 GB)"
hf_get Wan-AI/Wan2.2-I2V-A14B --local-dir "$CKPT/Wan2.2-I2V-A14B" --max-workers 8

step "SAM 2.1 hiera-large (video matting)"
hf_get facebook/sam2.1-hiera-large sam2.1_hiera_large.pt --local-dir "$CKPT"

if [[ $DISTILLED == 1 ]]; then
  step "Wan2.2-Lightning 4-step LoRA (configs/t4x4_distilled.yaml)"
  hf_get lightx2v/Wan2.2-Lightning --include "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/*.safetensors" \
    --local-dir "$CKPT/Wan2.2-Lightning"
fi

if [[ $LTX == 1 ]]; then
  step "LTX-Video 2B distilled + spatial upscaler + T5-XXL text encoder stored in bf16 (~16 GB)"
  in_env in24d_wan python - "$CKPT/LTX-Video" <<'PY'
import gc, json, os, shutil, sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
out, repo = Path(sys.argv[1]), "Lightricks/LTX-Video"
for f in ("ltxv-2b-0.9.8-distilled.safetensors", "ltxv-spatial-upscaler-0.9.8.safetensors",
          "tokenizer/added_tokens.json", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
          "tokenizer/spiece.model", "text_encoder/config.json", "text_encoder/model.safetensors.index.json"):
    hf_hub_download(repo, f, local_dir=out)
tmp = out.parent / "_ltx_tmp"
for shard in sorted(set(json.load(open(out / "text_encoder/model.safetensors.index.json"))["weight_map"].values())):
    dst = out / "text_encoder" / shard
    if not dst.exists():
        src = hf_hub_download(repo, f"text_encoder/{shard}", local_dir=tmp)
        save_file({k: v.to(torch.bfloat16) for k, v in load_file(src).items()}, dst, metadata={"format": "pt"})
        os.remove(src)
        gc.collect()
cfg = json.load(open(out / "text_encoder/config.json"))
cfg["torch_dtype"] = "bfloat16"
json.dump(cfg, open(out / "text_encoder/config.json", "w"), indent=1)
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(out / ".cache", ignore_errors=True)
print("ok")
PY
fi

step "Qwen2.5-VL-7B-Instruct (motion caption)"
hf_get Qwen/Qwen2.5-VL-7B-Instruct

step "Stable Diffusion 2.1 (diffusion features; sd2-community mirror, only the UNet/VAE/text parts)"
hf_get sd2-community/stable-diffusion-2-1 --include model_index.json "scheduler/*" "tokenizer/*" \
  "text_encoder/config.json" "text_encoder/model.safetensors" \
  "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
  "vae/config.json" "vae/diffusion_pytorch_model.safetensors"

step "Depth-Anything-V2 Large (trajectory smoothing, post_ema_smoothing.mode=tracked)"
hf_get depth-anything/Depth-Anything-V2-Large-hf

step "Era3D 512 6-view ortho (multi-view generation)"
hf_get pengHTYX/MacLab-Era3D-512-6view-ortho

step "rembg matting models + LPIPS backbones"
in_env in24d_wan python -c "
from rembg import new_session
import lpips
for m in ('u2net', 'birefnet-general'):
    new_session(m)
for net in ('alex', 'vgg'):
    lpips.LPIPS(net=net, verbose=False)
print('ok')"

step "CoTracker3 offline (track refinement)"
in_env in24d_4dgs python -c "
import torch
torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline', trust_repo=True)
print('ok')"

echo; echo "All weights downloaded."
