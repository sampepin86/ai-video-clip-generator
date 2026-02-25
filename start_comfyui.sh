#!/bin/bash
# ============================================================
# Démarre ComfyUI sur RunPod (à relancer si le pod redémarre)
# ============================================================
COMFY_DIR="/workspace/runpod-slim/ComfyUI"
VENV_PY="$COMFY_DIR/.venv/bin/python"
LOG="/workspace/comfyui.log"

echo "Vérification ComfyUI..."
if pgrep -f "main.py --listen" > /dev/null; then
    echo "[OK] ComfyUI déjà en cours d'exécution"
    echo "     Logs: tail -f $LOG"
    echo "     URL:  https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
    exit 0
fi

echo "Démarrage ComfyUI..."
cd "$COMFY_DIR"
nohup $VENV_PY main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --preview-method auto \
    2>&1 | tee "$LOG" &

sleep 5
if pgrep -f "main.py --listen" > /dev/null; then
    echo "[OK] ComfyUI démarré"
    echo "     URL:  https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
    echo "     Logs: tail -f $LOG"
else
    echo "[ERREUR] ComfyUI n'a pas démarré, voir $LOG"
    tail -20 "$LOG"
fi
