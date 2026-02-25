#!/bin/bash
# ============================================================
# Start ComfyUI + Gradio UI on RunPod
# Run after setup_wan_comfyui.sh has completed
# ============================================================

APP_DIR="${APP_DIR:-/workspace/ai-video-clip-generator}"
COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
VENV_PY="$COMFY_DIR/.venv/bin/python"

# Load .env if present
[ -f "$APP_DIR/.env" ] && set -a && source "$APP_DIR/.env" && set +a

# ── Start ComfyUI ─────────────────────────────────────────────────────────────
echo "Checking ComfyUI..."
if pgrep -f "main.py --listen" > /dev/null; then
    echo "[OK] ComfyUI already running"
else
    echo "Starting ComfyUI..."
    cd "$COMFY_DIR"
    nohup $VENV_PY main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --preview-method auto \
        > /workspace/comfyui.log 2>&1 &
    sleep 5
    if pgrep -f "main.py --listen" > /dev/null; then
        echo "[OK] ComfyUI started on port 8188"
    else
        echo "[ERROR] ComfyUI failed to start — see /workspace/comfyui.log"
        tail -10 /workspace/comfyui.log
    fi
fi

# ── Start Gradio UI ──────────────────────────────────────────────────────────
echo ""
echo "Checking Gradio UI..."
if pgrep -f "ui.py" > /dev/null; then
    echo "[OK] Gradio UI already running"
else
    echo "Starting Gradio UI..."
    cd "$APP_DIR"
    nohup python ui.py --server-name 0.0.0.0 --port 7860 --no-browser \
        > /workspace/gradio.log 2>&1 &
    sleep 3
    if pgrep -f "ui.py" > /dev/null; then
        echo "[OK] Gradio UI started on port 7860"
    else
        echo "[ERROR] Gradio UI failed to start — see /workspace/gradio.log"
        tail -10 /workspace/gradio.log
    fi
fi

echo ""
echo "============================================"
echo " Services running:"
echo "  ComfyUI: https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
echo "  Gradio:  https://${RUNPOD_POD_ID}-7860.proxy.runpod.net"
echo ""
echo " Logs:"
echo "  tail -f /workspace/comfyui.log"
echo "  tail -f /workspace/gradio.log"
echo "============================================"
