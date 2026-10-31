#!/usr/bin/env bash
set -eo pipefail   # no -u: conda (de)activation scripts of gxx_linux-64 read unset variables
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/third_party"
git -C "$HERE" submodule update --init
eval "$(conda shell.bash hook)"

new_env() { conda create -y -n "$1" python="$2"; conda activate "$1"; }

setup_wan() {
  new_env in24d_wan 3.10
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  # flash_attn is left out: it needs sm80+ (T4 is sm75) and the T4 profile uses SDPA instead
  grep -v "^flash_attn" "$TP/Wan2.2/requirements.txt" > /tmp/in24d_wan_req.txt
  pip install -r /tmp/in24d_wan_req.txt
  pip install einops decord librosa peft qwen-vl-utils lpips rembg onnxruntime-gpu scipy pyyaml matplotlib open_clip_torch sentencepiece timm av
  pip install "git+https://github.com/facebookresearch/sam2.git"
}

setup_era3d() {
  new_env in24d_era3d 3.9
  pip install -r "$TP/Era3D/requirements.txt"
  pip install pyyaml "huggingface_hub==0.25.2"   # diffusers 0.26 imports cached_download (removed in hub 0.26)
}

setup_4dgs() {
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
  new_env in24d_4dgs 3.10
  # cu121 so the CUDA extensions build with the system nvcc 12.x
  # (torch refuses a CUDA major-version mismatch between itself and nvcc)
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
  conda install -y -c conda-forge "gxx_linux-64=11"
  export CUDA_HOME="$CONDA_PREFIX" CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export MAX_JOBS="${MAX_JOBS:-16}"
  grep -vE "submodules/generative-models|^mmcv|^open3d" "$TP/EG4D/requirements.txt" > /tmp/in24d_gs_req.txt
  pip install --no-build-isolation -r /tmp/in24d_gs_req.txt opencv-python-headless
  # pymeshlab (imported by the 4DGS code) needs libGL.so.1; expose only the GL
  # libs, so the env's CUDA 12.1 toolkit libs do not shadow torch's own on LD_LIBRARY_PATH
  conda install -y -c conda-forge libgl
  mkdir -p "$CONDA_PREFIX/lib/gl_only"
  for l in libGL.so.1 libGLdispatch.so.0 libGLX.so.0 libX11.so.6 libxcb.so.1 libXau.so.6 libXdmcp.so.6; do
    ln -sf "$CONDA_PREFIX/lib/$l" "$CONDA_PREFIX/lib/gl_only/$l"
  done
  conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib/gl_only"
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
  pip install --no-build-isolation git+https://github.com/lxxue/prefix_sum.git git+https://github.com/lxxue/FRNN.git \
    || echo "FRNN build failed; tracking loss will use pytorch3d knn_points"
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  pip install pyyaml scipy open_clip_torch imageio-ffmpeg
}

for target in "${@:-wan era3d 4dgs}"; do
  for t in $target; do "setup_$t"; done
done
echo "Envs ready. Next: bash download_weights.sh"
