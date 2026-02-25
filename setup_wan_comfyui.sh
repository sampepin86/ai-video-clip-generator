#!/bin/bash
# ============================================================
# WAN Models Setup Script for ComfyUI on RunPod
# Supports: WAN 2.1 I2V (480p/720p), WAN 2.2 MoE, WAN 2.1 T2V 1.3B
# RTX 6000 Ada | CUDA 12.4 | ComfyUI 0.15+
# ============================================================
set -e

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
VENV_PY="$COMFY_DIR/.venv/bin/python"
VENV_PIP="$COMFY_DIR/.venv/bin/pip"
MODELS_DIR="$COMFY_DIR/models"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
HF_CACHE="/workspace/hf_cache"

export HF_HOME="$HF_CACHE"
mkdir -p "$HF_CACHE"

echo "============================================"
echo " STEP 1 — Custom Nodes"
echo "============================================"

# ComfyUI-VideoHelperSuite (video loading/saving)
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite"
    $VENV_PIP install -r "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite/requirements.txt" -q
    echo "[OK] ComfyUI-VideoHelperSuite installed"
else
    echo "[SKIP] ComfyUI-VideoHelperSuite already present"
fi

# ComfyUI-KJNodes (WanVideoTeaCacheKJ for acceleration)
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-KJNodes" ]; then
    git clone https://github.com/kijai/ComfyUI-KJNodes.git \
        "$CUSTOM_NODES_DIR/ComfyUI-KJNodes"
    $VENV_PIP install -r "$CUSTOM_NODES_DIR/ComfyUI-KJNodes/requirements.txt" -q 2>/dev/null || true
    echo "[OK] ComfyUI-KJNodes installed"
else
    echo "[SKIP] ComfyUI-KJNodes already present"
fi

# Disable WanVideoWrapper if present (conflicts with native WAN nodes)
if [ -d "$CUSTOM_NODES_DIR/ComfyUI-WanVideoWrapper" ]; then
    mv "$CUSTOM_NODES_DIR/ComfyUI-WanVideoWrapper" "$CUSTOM_NODES_DIR/ComfyUI-WanVideoWrapper.disabled"
    echo "[OK] WanVideoWrapper disabled (conflicts with native nodes)"
fi

echo ""
echo "============================================"
echo " STEP 2 — Download WAN models"
echo "============================================"

# Ensure huggingface_hub is available
$VENV_PIP install -q huggingface_hub --upgrade

# Helper: download only if file doesn't exist
dl() {
    local DEST="$1"
    local REPO="$2"
    local FILE="$3"
    if [ -f "$DEST" ] && [ -s "$DEST" ]; then
        echo "[SKIP] $(basename "$DEST") already downloaded"
    else
        mkdir -p "$(dirname "$DEST")"
        echo "[DL]  $(basename "$DEST") ..."
        $VENV_PY -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='$REPO', filename='$FILE', local_dir='/tmp/hf_dl')
shutil.copy(path, '$DEST')
print('  -> saved to $DEST')
"
    fi
}

# ── Shared: Text Encoder, VAE, CLIP Vision ───────────────────────────────────
dl "$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
   "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
   "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

dl "$MODELS_DIR/vae/wan_2.1_vae.safetensors" \
   "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
   "split_files/vae/wan_2.1_vae.safetensors"

dl "$MODELS_DIR/clip_vision/clip_vision_h.safetensors" \
   "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
   "split_files/clip_vision/clip_vision_h.safetensors"

# ── WAN 2.1 I2V 480p (default — recommended) ─────────────────────────────────
dl "$MODELS_DIR/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors" \
   "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
   "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"

# ── WAN 2.1 T2V 1.3B (ultra-fast, text-to-video only) ───────────────────────
dl "$MODELS_DIR/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors" \
   "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
   "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"

# ── WAN 2.2 MoE (high + low noise pair) ──────────────────────────────────────
dl "$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
   "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
   "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

dl "$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
   "Comfy-Org/Wan_2.2_ComfyUI_repackaged" \
   "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

echo ""
echo "============================================"
echo " STEP 3 — Verify downloads"
echo "============================================"
echo "diffusion_models/:"
ls -lh "$MODELS_DIR/diffusion_models/" 2>/dev/null || echo "  (empty)"
echo "text_encoders/:"
ls -lh "$MODELS_DIR/text_encoders/" 2>/dev/null || echo "  (empty)"
echo "vae/:"
ls -lh "$MODELS_DIR/vae/" 2>/dev/null || echo "  (empty)"
echo "clip_vision/:"
ls -lh "$MODELS_DIR/clip_vision/" 2>/dev/null || echo "  (empty)"

echo ""
echo "============================================"
echo " SETUP COMPLETE — Models ready"
echo "  ComfyUI: $COMFY_DIR"
echo "  Start:   bash start_comfyui.sh"
echo "============================================"
