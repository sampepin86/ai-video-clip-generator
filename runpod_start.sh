#!/bin/bash
# ============================================================
# RunPod Startup Script - AI Video Clip Generator
# Use as "Docker Start Command" in a RunPod template.
# Idempotent: safe to re-run on pod restart.
# ============================================================
set -e

COMFY_DIR="/workspace/runpod-slim/ComfyUI"
VENV_PY="$COMFY_DIR/.venv/bin/python"
VENV_PIP="$COMFY_DIR/.venv/bin/pip"
PROJECT_DIR="/workspace/ai-video-clip-generator"
LOG="/workspace/comfyui.log"

echo "============================================"
echo " AI Video Clip Generator - RunPod Startup"
echo "============================================"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Project not found at $PROJECT_DIR"
    echo "        Upload project files or set up git repo."
    exit 1
fi

cd "$PROJECT_DIR"

# -- 1. Setup models if not present --
MODELS_DIR="$COMFY_DIR/models"
if [ ! -f "$MODELS_DIR/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors" ]; then
    echo "[SETUP] Models not found - running setup_wan_comfyui.sh..."
    bash "$PROJECT_DIR/setup_wan_comfyui.sh"
else
    echo "[OK] Models already present"
fi

# -- 2. Start ComfyUI in background --
if pgrep -f "main.py --listen" > /dev/null 2>&1; then
    echo "[OK] ComfyUI already running"
else
    echo "[START] Launching ComfyUI..."
    cd "$COMFY_DIR"
    nohup $VENV_PY main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --preview-method auto \
        > "$LOG" 2>&1 &
    cd "$PROJECT_DIR"
fi

# -- 3. Wait for ComfyUI to be ready --
echo "[WAIT] Waiting for ComfyUI on localhost:8188..."
MAX_WAIT=120
WAITED=0
while ! curl -s -o /dev/null http://127.0.0.1:8188/ 2>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[ERROR] ComfyUI did not start within ${MAX_WAIT}s"
        tail -20 "$LOG"
        exit 1
    fi
    printf "."
done
echo ""
echo "[OK] ComfyUI is ready (took ~${WAITED}s)"

# -- 4. Install ffmpeg if not present --
if ! command -v ffmpeg &> /dev/null; then
    echo "[SETUP] Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
    echo "[OK] ffmpeg installed"
else
    echo "[OK] ffmpeg already installed"
fi

# -- 5. Install project Python dependencies --
echo "[SETUP] Installing project requirements..."
$VENV_PIP install -q -r "$PROJECT_DIR/requirements.txt"
echo "[OK] Dependencies installed"

# -- 6. Export environment variables --
# Load .env if present
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a

export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

[ -z "$GEMINI_API_KEY" ] && echo "[WARN] GEMINI_API_KEY is not set - scenario generation will fail"
[ -z "$RUNPOD_POD_ID" ] && echo "[WARN] RUNPOD_POD_ID is not set - proxy URLs disabled (localhost OK on-pod)"

echo ""
echo "============================================"
echo " Launching Gradio UI on 0.0.0.0:7860"
echo " Access via: https://${RUNPOD_POD_ID}-7860.proxy.runpod.net"
echo "============================================"

# -- 7. Launch Gradio UI (blocking, foreground) --
cd "$PROJECT_DIR"
exec python ui.py --server-name 0.0.0.0 --port 7860 --no-browser
