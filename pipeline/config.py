"""
Configuration centralisée — tous les modèles WAN disponibles.
Utilisé par le pipeline et l'interface graphique.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import os, urllib.request, json
from pathlib import Path

# ── Load .env if present (before reading any env var) ─────────────────────────
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── RunPod ────────────────────────────────────────────────────────────────────
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID", "")
_ON_RUNPOD = os.path.exists("/workspace")

if _ON_RUNPOD:
    # On RunPod: ComfyUI is on localhost
    COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
    COMFYUI_WS_URL   = os.environ.get("COMFYUI_WS_URL",   "ws://127.0.0.1:8188/ws")
else:
    # Local dev: use RunPod proxy
    COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", f"https://{RUNPOD_POD_ID}-8188.proxy.runpod.net")
    COMFYUI_WS_URL   = os.environ.get("COMFYUI_WS_URL",   f"wss://{RUNPOD_POD_ID}-8188.proxy.runpod.net/ws")

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Encodeurs communs aux deux versions ──────────────────────────────────────
TEXT_ENCODER_FP8  = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
TEXT_ENCODER_FP16 = "umt5_xxl_fp16.safetensors"
CLIP_VISION_H     = "clip_vision_h.safetensors"


@dataclass
class WanModel:
    """Définition complète d'un modèle WAN."""
    id: str                          # Identifiant unique (id dans les menus)
    label: str                       # Nom affiché dans l'UI
    version: Literal["2.1", "2.2"]  # Version WAN
    filename: str                    # Nom du fichier safetensors (low_noise pour 2.2)
    vae: str                         # VAE à utiliser
    description: str                 # Description courte
    # Résolution recommandée
    default_width: int = 832
    default_height: int = 480
    # Frames recommandées (multiples de 4+1)
    default_frames: int = 81
    frame_options: list[int] = field(default_factory=lambda: [33, 49, 65, 81, 97])
    # Paramètres sampler recommandés
    default_steps: int = 20
    default_cfg: float = 6.0
    # Comportement: I2V supporte la continuité inter-scènes
    supports_i2v: bool = True
    # Qualité indicative (pour trier dans l'UI)
    quality_score: int = 5           # 1-10
    # Taille approximative sur disque
    size_gb: float = 16.0
    # Taille du modèle (pour TeaCache coefficients)
    model_params: Literal["1.3B", "14B"] = "14B"
    # ── WAN 2.2 MoE spécifique ────────────────────────────────────────────────
    high_noise_filename: str | None = None  # Fichier high-noise (MoE pair)
    split_step: int = 10                    # Étape de transition high→low noise
    shift: float = 5.0                      # ModelSamplingSD3 shift

    @property
    def teacache_coefficients(self) -> str:
        """Coefficient TeaCache adapté à la taille du modèle."""
        if self.model_params == "1.3B":
            return "1.3B"
        # 14B: i2v_480 pour 480p, i2v_720 pour 720p
        if self.default_height >= 720:
            return "i2v_720"
        return "i2v_480"

    @property
    def teacache_threshold(self) -> float:
        """Seuil TeaCache adapté au modèle."""
        if self.model_params == "1.3B":
            return 0.03  # ~10x plus petit pour 1.3B
        return 0.275


# ── CATALOGUE COMPLET ─────────────────────────────────────────────────────────
MODELS: dict[str, WanModel] = {

    # ── WAN 2.1 T2V 1.3B (ultra-rapide, text-to-video uniquement) ────────────
    "wan21_t2v_1.3B": WanModel(
        id="wan21_t2v_1.3B",
        label="WAN 2.1 T2V 1.3B ⚡ (ultra-rapide)",
        version="2.1",
        filename="wan2.1_t2v_1.3B_bf16.safetensors",
        vae="wan_2.1_vae.safetensors",
        description="T2V 1.3B ultra-rapide (~5x plus vite que 14B). Text-to-video seulement, pas de continuité inter-scènes.",
        default_width=832, default_height=480,
        default_frames=81,
        frame_options=[33, 49, 65, 81, 97, 113],
        default_steps=20,
        default_cfg=6.0,
        supports_i2v=False,
        quality_score=4,
        size_gb=2.84,
        model_params="1.3B",
    ),

    # ── WAN 2.1 I2V 480p ─────────────────────────────────────────────────────
    "wan21_i2v_480p_fp8_scaled": WanModel(
        id="wan21_i2v_480p_fp8_scaled",
        label="WAN 2.1 I2V 480p fp8 (recommandé)",
        version="2.1",
        filename="wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        description="I2V 480p optimisé fp8. Rapide, bonne cohérence.",
        default_width=832, default_height=480,
        default_frames=81,
        frame_options=[33, 49, 65, 81, 97, 113],
        default_steps=12,
        default_cfg=6.0,
        quality_score=7,
        size_gb=16.0,
    ),
    "wan21_i2v_480p_fp8_e4m3fn": WanModel(
        id="wan21_i2v_480p_fp8_e4m3fn",
        label="WAN 2.1 I2V 480p fp8 e4m3fn",
        version="2.1",
        filename="wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        vae="wan_2.1_vae.safetensors",
        description="I2V 480p fp8 e4m3fn. Légèrement plus précis que scaled.",
        default_width=832, default_height=480,
        default_frames=81,
        frame_options=[33, 49, 65, 81, 97, 113],
        quality_score=7,
        size_gb=16.0,
    ),
    "wan21_i2v_720p_fp8_scaled": WanModel(
        id="wan21_i2v_720p_fp8_scaled",
        label="WAN 2.1 I2V 720p fp8 scaled",
        version="2.1",
        filename="wan2.1_i2v_720p_14B_fp8_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        description="I2V 720p. Meilleure résolution, ~25% plus lent.",
        default_width=1280, default_height=720,
        default_frames=81,
        frame_options=[33, 49, 65, 81, 97],
        default_cfg=5.0,
        quality_score=9,
        size_gb=16.0,
    ),
    "wan21_i2v_720p_fp16": WanModel(
        id="wan21_i2v_720p_fp16",
        label="WAN 2.1 I2V 720p fp16 (qualité max)",
        version="2.1",
        filename="wan2.1_i2v_720p_14B_fp16.safetensors",
        vae="wan_2.1_vae.safetensors",
        description="I2V 720p pleine précision fp16. Meilleure qualité, plus lent.",
        default_width=1280, default_height=720,
        default_frames=65,
        frame_options=[33, 49, 65, 81],
        default_steps=25,
        default_cfg=5.0,
        quality_score=10,
        size_gb=28.0,
    ),

    # ── WAN 2.2 I2V — MoE dual-model (high + low noise) ──────────────────────
    # Ref officielle: https://docs.comfy.org/tutorials/video/wan/wan2_2
    # WAN 2.2 charge DEUX modèles: high_noise pour les premiers steps,
    # low_noise pour les derniers (architecture MoE split-denoising).
    # VAE = wan_2.1_vae (pas wan2.2_vae !)
    "wan22_i2v_fp8": WanModel(
        id="wan22_i2v_fp8",
        label="WAN 2.2 I2V MoE fp8 ✨",
        version="2.2",
        filename="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        high_noise_filename="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        vae="wan_2.1_vae.safetensors",
        description=(
            "WAN 2.2 I2V MoE (high+low noise). Architecture split-denoising "
            "pour une qualité cinématique. Réf: workflow officiel ComfyUI."
        ),
        default_width=832, default_height=480,
        default_frames=81,
        frame_options=[33, 49, 65, 81, 97],
        default_steps=20,
        default_cfg=3.5,
        split_step=10,
        shift=5.0,
        quality_score=9,
        size_gb=32.0,
    ),
}

# Modèle par défaut
DEFAULT_MODEL_ID = "wan21_i2v_480p_fp8_scaled"

# Stratégie de sélection automatique par scène
AUTO_MODEL_STRATEGY = {
    "dynamic":    "wan22_i2v_fp8",              # MoE haute qualité
    "subtle":     "wan22_i2v_fp8",              # MoE haute qualité
    "quality":    "wan22_i2v_fp8",              # MoE haute qualité
    "fast":       "wan21_i2v_480p_fp8_scaled",  # Prototypage rapide
    "ultra_fast": "wan21_t2v_1.3B",          # Ultra-rapide 1.3B (T2V uniquement)
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_model(model_id: str) -> WanModel:
    if model_id not in MODELS:
        raise ValueError(f"Modèle inconnu: {model_id}. Options: {list(MODELS.keys())}")
    return MODELS[model_id]


def list_models_for_ui() -> list[tuple[str, str]]:
    """Retourne [(label, id), ...] pour les composants Gradio."""
    return [(m.label, m.id) for m in MODELS.values()]


def check_model_on_runpod(model: WanModel) -> bool:
    """Vérifie si le fichier modèle existe sur RunPod."""
    _UA = "Mozilla/5.0 (compatible; AI-Video-Generator/1.0)"
    try:
        models_url = f"{COMFYUI_BASE_URL}/models/diffusion_models"
        req = urllib.request.Request(models_url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=5) as r:
            available = json.loads(r.read())
            return model.filename in available
    except Exception:
        return False
