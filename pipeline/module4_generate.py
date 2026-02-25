"""
Module 4 — Boucle de génération I2V.
Génère les clips séquentiellement: la dernière frame de scène N
devient l'image d'init de scène N+1 (continuité visuelle).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable

from module3_comfyui_client import ComfyUIClient
from config import get_model, DEFAULT_MODEL_ID, GEMINI_API_KEY


def generate_all_scenes(
    scenes: list[dict],
    output_dir: str | Path = "output/clips",
    comfyui_url: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    resume: bool = True,
    generation_params: dict | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    stop_flag: dict | None = None,
    custom_init_image_b64: str | None = None,
) -> list[Path]:
    """
    Génère tous les clips vidéo I2V de façon séquentielle.

    Architecture:
        Scene 1 → T2V (pas d'image init)
        Scene 2 → I2V (init = dernière frame de scène 1)
        Scene N → I2V (init = dernière frame de scène N-1)

    Args:
        scenes: Liste des scènes depuis module2_scenarios
        output_dir: Dossier de sortie pour les clips
        comfyui_url: URL de ComfyUI (optionnel)
        on_progress: Callback(scene_idx, total, status)
        resume: Si True, skip les scènes déjà générées (fichier existant)
        generation_params: Override des params WAN (steps, cfg, etc.)

    Returns:
        Liste des chemins vers les clips MP4 générés (dans l'ordre)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_kwargs = {"base_url": comfyui_url} if comfyui_url else {}
    client = ComfyUIClient(**client_kwargs)
    model = get_model(model_id)

    generated_clips: list[Path] = []
    last_frame_b64: str | None = None
    total = len(scenes)

    print(f"\n[Loop] Démarrage génération: {total} scènes → {output_dir}")
    stats = {"generated": 0, "skipped": 0, "failed": 0}

    # ── Image init pour la première scène ────────────────────────────────────
    p0 = {**(generation_params or {}), "num_frames": model.default_frames}
    w0 = p0.get("width", model.default_width)
    h0 = p0.get("height", model.default_height)

    if custom_init_image_b64:
        print(f"[Loop] 🖼️ Image de départ personnalisée fournie")
        init_frame_b64 = custom_init_image_b64
    else:
        first_scene_prompt = scenes[0].get("visual_prompt", "") if scenes else ""
        print(f"[Loop] Génération image init (Gemini Imagen) pour scène 1...")
        init_frame_b64 = _generate_gemini_image_b64(first_scene_prompt, w0, h0)
        if init_frame_b64:
            print(f"[Loop] ✅ Image init Imagen générée ({w0}×{h0})")
        else:
            print(f"[Loop] ⚠️  Imagen échoué → frame grise")
            init_frame_b64 = _make_gray_frame_b64(w0, h0)

    for idx, scene in enumerate(scenes):
        # ── Stop check ────────────────────────────────────────────────────────
        if stop_flag and stop_flag.get("stop_requested"):
            print(f"\n[Loop] ⏹ Arrêt demandé après {idx} scènes")
            break

        scene_id = scene.get("scene_id", idx + 1)
        clip_path = output_dir / f"scene_{scene_id:03d}.mp4"
        status_prefix = f"  [{idx+1}/{total}] Scène {scene_id}"

        # ── Calcul de la durée en premier (nécessaire partout) ────────────────
        t_start_s = float(scene.get("start_time") or 0)
        t_end_s   = float(scene.get("end_time") or scene.get("duration") or (t_start_s + 4))
        duration  = max(t_end_s - t_start_s, 1.0)

        # ── Resume: skip si fichier déjà là ──────────────────────────────────
        if resume and clip_path.exists() and clip_path.stat().st_size > 10_000:
            print(f"{status_prefix} [SKIP] Déjà générée ({clip_path.stat().st_size // 1024}KB)")
            generated_clips.append(clip_path)
            stats["skipped"] += 1
            last_frame_b64 = _extract_last_frame_local(clip_path)
            continue

        print(f"{status_prefix} Génération...")
        print(f"    Prompt: {scene.get('visual_prompt','')[:80]}...")
        print(f"    Motion: {scene.get('motion_prompt','')}")
        print(f"    Durée:  {duration:.1f}s")

        if on_progress:
            on_progress(idx, total, f"Generating scene {scene_id}")

        # ── Calcul du nombre de frames selon la durée ─────────────────────────
        fps = (generation_params or {}).get("fps", 16)
        num_frames = _duration_to_wan_frames(duration, fps)

        params = {**(generation_params or {}), "num_frames": num_frames}

        # ── init_image: last_frame si dispo, sinon image Imagen de la scène 1 ──────
        init_b64 = last_frame_b64 if last_frame_b64 else init_frame_b64
        if last_frame_b64:
            print(f"    Mode:   I2V (continuité depuis scène précédente)")
        else:
            print(f"    Mode:   I2V (image init Imagen scène 1)")

        # ── Build + envoyer workflow via generate_scene ───────────────────────
        try:
            t_start = time.time()
            clip_path, last_frame_b64 = client.generate_scene(
                visual_prompt=scene.get("visual_prompt", "cinematic shot"),
                motion_prompt=scene.get("motion_prompt", "slow camera movement"),
                output_path=clip_path,
                init_image_b64=init_b64,
                params=params,
                model=model,
            )
            elapsed = time.time() - t_start

            print(f"{status_prefix} ✓ Généré en {elapsed:.0f}s → {clip_path.name}")
            generated_clips.append(clip_path)
            stats["generated"] += 1

            if last_frame_b64:
                print(f"    [OK] Dernière frame extraite pour scène {scene_id + 1}")
            else:
                print(f"    [WARN] Pas de dernière frame, scène {scene_id + 1} sera T2V")

        except Exception as e:
            print(f"{status_prefix} [ERREUR] {e}")
            stats["failed"] += 1
            generated_clips.append(None)  # type: ignore
            # On garde last_frame_b64 tel quel: la scène suivante repart
            # depuis la dernière frame WAN connue (pas de retour à Imagen)

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n[Loop] Terminé: {stats['generated']} générées | {stats['skipped']} skippées | {stats['failed']} erreurs")

    # Filtre les None (erreurs)
    valid_clips = [c for c in generated_clips if c is not None]
    print(f"[Loop] {len(valid_clips)} clips valides sur {total}")

    return valid_clips


def _generate_gemini_image_b64(prompt: str, width: int = 832, height: int = 480) -> str | None:
    """
    Génère une image via Gemini Imagen 3 pour la première scène.
    Retourne la base64 PNG, ou None si échec.
    """
    try:
        from google import genai
        from google.genai import types
        import base64

        # Aspect ratio selon les dimensions
        if width / height >= 1.7:
            aspect = "16:9"
        elif height / width >= 1.7:
            aspect = "9:16"
        else:
            aspect = "1:1"

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=f"Cinematic film still, {prompt}",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect,
            ),
        )
        if response.generated_images:
            return base64.b64encode(response.generated_images[0].image.image_bytes).decode()
    except Exception as e:
        print(f"  [WARN] Gemini Imagen: {e}")
    return None


def _make_gray_frame_b64(width: int = 832, height: int = 480) -> str:
    """Génère une image grise unie (PNG) en base64 — init pour première scène T2V."""
    import base64, struct, zlib
    # PNG minimal en pur Python (stdlib uniquement), couleur grise 50%
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit RGB
    # IDAT: chaque ligne = filtre 0 + pixels RGB gris (128,128,128)
    raw = b""
    row = b"\x00" + b"\x80\x80\x80" * width
    for _ in range(height):
        raw += row
    compressed = zlib.compress(raw)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


def _duration_to_wan_frames(duration: float, fps: int = 16) -> int:
    """
    WAN I2V supporte des longueurs fixes (4k+1).
    Retourne la valeur la plus proche de la durée demandée,
    plafonnée à 97 frames max pour des scènes de ~6s.
    """
    target_frames = duration * fps
    # Valeurs supportées par WAN (4k+1)
    wan_supported = [17, 33, 49, 65, 81, 97]
    best = min(wan_supported, key=lambda x: abs(x - target_frames))
    return best


def _extract_last_frame_local(video_path: Path) -> str | None:
    """Extrait la dernière frame d'un fichier local MP4 (base64 PNG)."""
    import base64, subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            "ffmpeg", "-sseof", "-0.1", "-i", str(video_path),
            "-vframes", "1", "-y", tmp_path
        ], capture_output=True, check=True)
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        Path(tmp_path).unlink(missing_ok=True)
        return b64
    except Exception as e:
        print(f"  [WARN] Extraction frame locale échouée: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python module4_generate.py scenes.json [output_dir]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        scenes = json.load(f)

    out = sys.argv[2] if len(sys.argv) > 2 else "output/clips"

    clips = generate_all_scenes(scenes, output_dir=out, model_id=DEFAULT_MODEL_ID)
    print("\nClips générés:")
    for c in clips:
        print(f"  {c}")
