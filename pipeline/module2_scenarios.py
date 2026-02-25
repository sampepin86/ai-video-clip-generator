"""
Module 2 — Génération de scénarios visuels via Google Gemini.

Deux modes:
  1. analyze_audio_direct()  — envoie le MP3 directement à Gemini 2.5 Flash
                               (pas de Whisper nécessaire, analyse audio native)
  2. generate_scenarios()    — fallback si on a déjà des segments Whisper
"""
from __future__ import annotations
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import TypedDict

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Durée cible par scène (secondes) — WAN I2V génère jusqu'à ~6s par clip
SCENE_DURATION_MIN = 4.0
SCENE_DURATION_MAX = 6.0


class Scene(TypedDict):
    scene_id: int
    start_time: float
    end_time: float
    visual_prompt: str       # Prompt détaillé pour SDXL / WAN T2I
    motion_prompt: str       # Type de mouvement I2V
    consistency_tags: str    # Description courte du sujet (IP-Adapter)


SYSTEM_PROMPT = """Tu es un directeur artistique expert en clips musicaux et en IA générative.
Tu reçois les paroles d'une chanson avec leurs timestamps. Tu dois:
1. Analyser le ton, le genre, l'émotion, l'ambiance des paroles.
2. En déduire automatiquement un STYLE ARTISTIQUE cohérent pour le clip.
3. Générer un découpage en scènes visuelles basé sur ce style.

RÈGLES STRICTES:
1. Chaque scène dure entre 4 et 6 secondes (contrainte technique I2V).
2. Les scènes doivent couvrir TOUTE la durée de la chanson sans chevauchement ni trou.
3. Les visual_prompts doivent être en anglais, très détaillés, style cinématographique.
4. Les motion_prompts décrivent le mouvement de caméra ou du sujet.
5. Les consistency_tags décrivent le sujet principal (maintien de cohérence entre scènes).
6. Retourne UNIQUEMENT un JSON valide, aucun texte avant ou après.

FORMAT DE SORTIE (objet JSON):
{
  "style": "description courte du style artistique déduit (1-2 lignes, en anglais)",
  "style_analysis": "explication en français de pourquoi ce style correspond à la chanson",
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 5.5,
      "visual_prompt": "cinematic close-up of [description], [style elements], dramatic lighting, film grain, 4k",
      "motion_prompt": "slow dolly forward, gentle camera movement",
      "consistency_tags": "young woman with dark hair, urban environment"
    }
  ]
}"""

# Même prompt mais on précise explicitement qu'on fournit l'audio
AUDIO_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Tu reçois les paroles d'une chanson avec leurs timestamps.",
    "Tu reçois directement le fichier audio d'une chanson. "
    "Écoute-le, transcris mentalement les paroles, identifie les timestamps "
    "de chaque phrase, et génère le plan de scènes visuelles.",
)


def analyze_audio_direct(
    audio_path: str | Path,
    api_key: str = GEMINI_API_KEY,
) -> tuple[list["Scene"], str, str, list[dict]]:
    """
    Envoie le fichier audio directement à Gemini 2.5 Flash.
    Gemini fait la transcription ET génère les scènes en un seul appel.
    Pas besoin de Whisper.

    Args:
        audio_path: Chemin local vers le MP3/WAV/M4A.
        api_key: Clé API Gemini.

    Returns:
        (scenes, style_en, style_analysis, segments)
        - scenes: scènes prêtes pour I2V
        - style_en: style artistique en anglais
        - style_analysis: explication française du style
        - segments: transcription [{start, end, text}, ...] extraite par Gemini
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        raise ImportError("Installe: pip install google-generativeai") from e

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")

    genai.configure(api_key=api_key)

    # Détection MIME
    mime = mimetypes.guess_type(str(audio_path))[0] or "audio/mpeg"

    file_size_mb = audio_path.stat().st_size / 1024**2
    print(f"[Gemini-Audio] Upload '{audio_path.name}' ({file_size_mb:.1f}MB)...")

    # Upload via Files API (supporte jusqu'à ~2GB)
    uploaded = genai.upload_file(str(audio_path), mime_type=mime)
    print(f"[Gemini-Audio] Fichier uploadé: {uploaded.name}")

    user_prompt = """Analyse ce fichier audio musical.

Retourne un objet JSON avec:
1. La transcription complète avec timestamps précis (chaque phrase/vers)
2. Le style artistique déduit de la musique ET des paroles
3. Le plan de scènes visuelles couvrant toute la durée

FORMAT EXACT:
{
  "style": "style en anglais pour les prompts vidéo",
  "style_analysis": "explication en français du style choisi",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.5, "text": "paroles ici"}
  ],
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 4.0,
      "visual_prompt": "...",
      "motion_prompt": "...",
      "consistency_tags": "..."
    }
  ]
}

Règles scènes: 3-5 secondes chacune, couvrir TOUTE la durée, visual_prompts en anglais cinématographique."""

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        },
        system_instruction=AUDIO_SYSTEM_PROMPT,
    )

    print("[Gemini-Audio] Analyse en cours (transcription + scènes)...")
    response = model.generate_content([uploaded, user_prompt])

    # Nettoyage fichier distant
    try:
        uploaded.delete()
    except Exception:
        pass

    raw = response.text.strip()
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise ValueError(f"Gemini n'a pas retourné de JSON valide:\n{raw[:500]}")

    data = json.loads(json_match.group())

    style_en: str = data.get("style", "cinematic music video, moody atmosphere")
    style_analysis: str = data.get("style_analysis", "")
    scenes: list[Scene] = data.get("scenes", [])
    segments: list[dict] = data.get("segments", [])

    print(f"[Gemini-Audio] Style: {style_en}")
    print(f"[Gemini-Audio] {len(segments)} segments transcrits | {len(scenes)} scènes générées")

    for s in scenes:
        duration = float(s.get("end_time") or 0) - float(s.get("start_time") or 0)
        if duration > SCENE_DURATION_MAX + 0.5:
            print(f"  [WARN] Scène {s.get('scene_id','?')} trop longue: {duration:.1f}s")

    return scenes, style_en, style_analysis, segments


def generate_scenarios(
    segments: list[dict],
    total_duration: float | None = None,
    api_key: str = GEMINI_API_KEY,
) -> tuple[list[Scene], str, str]:
    """
    Analyse la chanson et génère les scènes visuelles depuis les segments Whisper.
    Le style artistique est déduit automatiquement par Gemini à partir des paroles.

    Args:
        segments: Liste de segments [{start, end, text}, ...]
        total_duration: Durée totale audio (optionnel, infère depuis segments)
        api_key: Clé API Gemini

    Returns:
        (scenes, style_en, style_analysis_fr)
        - scenes: liste de scènes prêtes pour I2V
        - style_en: style artistique en anglais (pour les prompts)
        - style_analysis_fr: explication en français du style choisi
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as e:
        raise ImportError("Installe: pip install google-generativeai") from e

    genai.configure(api_key=api_key)

    if total_duration is None and segments:
        total_duration = segments[-1]["end"]

    # Formate les paroles pour Gemini
    lyrics_text = "\n".join(
        f"[{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}"
        for seg in segments
    )

    user_prompt = f"""DURÉE TOTALE: {total_duration:.1f} secondes

PAROLES AVEC TIMESTAMPS:
{lyrics_text}

Analyse ces paroles, déduis le style artistique, puis génère le plan de scènes en JSON.
Chaque scène = 3-5 secondes. Le style doit être cohérent avec les émotions et le contenu des paroles."""

    print(f"[Gemini] Génération des scénarios pour {total_duration:.1f}s de contenu...")

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.8,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        },
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(user_prompt)
    raw = response.text.strip()

    # Parse JSON (robuste aux balisages markdown)
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise ValueError(f"Gemini n'a pas retourné de JSON valide:\n{raw[:500]}")

    data = json.loads(json_match.group())

    style_en: str = data.get("style", "cinematic music video, moody atmosphere")
    style_analysis: str = data.get("style_analysis", "")
    scenes: list[Scene] = data.get("scenes", [])

    print(f"[Gemini] Style déduit: {style_en}")
    if style_analysis:
        print(f"[Gemini] Analyse: {style_analysis}")
    print(f"[Gemini] {len(scenes)} scènes générées")

    for s in scenes:
        duration = float(s.get("end_time") or 0) - float(s.get("start_time") or 0)
        if duration > SCENE_DURATION_MAX + 0.5:
            print(f"  [WARN] Scène {s.get('scene_id','?')} trop longue: {duration:.1f}s")

    return scenes, style_en, style_analysis


def save_scenes(
    scenes: list[Scene],
    output_path: str,
    style_en: str = "",
    style_analysis: str = "",
) -> None:
    payload = {
        "style": style_en,
        "style_analysis": style_analysis,
        "scenes": scenes,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Gemini] Scènes sauvegardées → {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python module2_scenarios.py audio.mp3")
        print("       python module2_scenarios.py segments.json  # fallback Whisper")
        sys.exit(1)

    arg = sys.argv[1]
    if Path(arg).suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
        # Mode audio direct
        scenes, style_en, style_analysis, segments = analyze_audio_direct(arg)
        print(f"\nStyle: {style_en}")
        print(f"Analyse: {style_analysis}")
        print(f"Segments: {len(segments)} | Scènes: {len(scenes)}")
        print(json.dumps(scenes[:2], indent=2, ensure_ascii=False))
    else:
        # Fallback segments Whisper
        from module1_transcribe import load_segments
        segs = load_segments(arg)
        scenes, style_en, style_analysis = generate_scenarios(segs)
        print(f"Style: {style_en}")
        print(json.dumps(scenes[:2], indent=2, ensure_ascii=False))
