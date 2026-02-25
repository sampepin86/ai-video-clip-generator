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


SYSTEM_PROMPT = """Tu es un réalisateur de clips musicaux professionnel spécialisé en vidéo IA.
Ton rôle: créer un storyboard visuel qui COLLE AUX PAROLES de la chanson.

PRINCIPES FONDAMENTAUX:
- Chaque scène doit ILLUSTRER VISUELLEMENT ce que les paroles disent à ce moment précis.
- Si les paroles parlent de "la nuit", montre une scène de nuit. Si elles parlent de "courir", montre quelqu'un qui court.
- NE PAS inventer un univers sci-fi/fantastique SAUF si les paroles le suggèrent explicitement.
- Reste RÉALISTE et CINÉMATOGRAPHIQUE par défaut. Pense clip musical professionnel (Drake, The Weeknd, Stromae style).
- Un personnage principal cohérent doit apparaître dans la majorité des scènes.

CONTINUITÉ VISUELLE (CRUCIAL):
- La DERNIÈRE FRAME de chaque scène devient automatiquement l'IMAGE DE DÉPART de la scène suivante.
- Donc la FIN d'une scène doit visuellement TRANSITIONNER vers le DÉBUT de la scène suivante.
- Évite les ruptures brutales: si scène 3 finit dans une rue sombre, scène 4 doit commencer dans un lieu visuellement compatible (pas un saut vers une plage ensoleillée).
- Pense en termes de FLUX VISUEL continu: même palette de couleurs entre scènes adjacentes, transitions de lieux progressives.
- Si tu changes de lieu, fais-le graduellement (intérieur → porte → extérieur) plutôt que brutalement.

RÈGLES TECHNIQUES STRICTES:
1. Chaque scène dure EXACTEMENT entre 4 et 6 secondes. JAMAIS plus de 6s. JAMAIS moins de 4s.
2. Les scènes couvrent TOUTE la durée de la chanson sans trou ni chevauchement.
3. Les visual_prompts sont en anglais, ultra-détaillés, photographiques/cinématographiques.
4. Chaque visual_prompt DOIT contenir:
   - Le sujet principal (personnage, lieu, action)
   - L'éclairage et l'ambiance (golden hour, neon lights, moody shadows...)
   - Le cadrage (close-up, wide shot, medium shot, aerial...)
   - Le style photo (35mm film, shallow depth of field, high contrast...)
5. Les motion_prompts décrivent le mouvement de caméra (pas le sujet).
6. Les consistency_tags IDENTIQUES pour le personnage principal à travers tout le clip.
7. Le visual_prompt de chaque scène doit décrire un point de DÉPART visuellement compatible avec la FIN de la scène précédente.

MAUVAIS EXEMPLE (trop générique, déconnecté des paroles):
  "A holographic interface in a cyberpunk cityscape with neon lights"

BON EXEMPLE (continuité visuelle entre scènes adjacentes):
  Scène 3 (paroles: "walking through the rain"): "Cinematic medium shot of a young man walking alone on a rain-soaked city street at night, streetlights reflecting on wet asphalt, moody blue-orange color grading, 35mm film grain, shallow depth of field"
  Scène 4 (paroles: "I stop and look up"): "Close-up of same young man stopping on the rain-soaked sidewalk, tilting his head up toward the sky, rain drops on his face, neon signs blurred in background, same moody blue-orange tones, 35mm film grain"
  → La fin de scène 3 (homme marchant sous la pluie en ville) transitionne naturellement vers scène 4 (même homme, même lieu, action suivante)

FORMAT DE SORTIE (objet JSON strict):
{
  "style": "description courte du style visuel (1-2 lignes, en anglais)",
  "style_analysis": "explication en français de pourquoi ce style correspond à la chanson",
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 5.0,
      "visual_prompt": "...",
      "motion_prompt": "slow dolly forward",
      "consistency_tags": "young man, mid-20s, dark jacket, urban setting"
    }
  ]
}"""

# Variante pour le mode audio direct
AUDIO_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Tu es un réalisateur de clips musicaux professionnel spécialisé en vidéo IA.\nTon rôle: créer un storyboard visuel qui COLLE AUX PAROLES de la chanson.",
    "Tu es un réalisateur de clips musicaux professionnel spécialisé en vidéo IA.\n"
    "Tu reçois directement le fichier audio d'une chanson. "
    "Écoute-le attentivement, transcris les paroles avec leurs timestamps exacts, "
    "puis crée un storyboard visuel qui ILLUSTRE PRÉCISÉMENT les paroles."
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

    # Le titre de la chanson donne un contexte crucial pour le clip
    song_title = audio_path.stem  # ex: "Never Done again"

    user_prompt = f"""TITRE DE LA CHANSON: "{song_title}"

Analyse ce fichier audio musical. Le titre est un indice MAJEUR pour comprendre le thème.

Écoute attentivement les paroles et la musique, puis crée un clip vidéo qui:
- ILLUSTRE les paroles vers par vers (chaque scène = ce que les paroles disent à ce moment)
- Reflète le THÈME du titre "{song_title}" comme fil conducteur
- Reste RÉALISTE et cinématographique (pas de sci-fi sauf si les paroles le demandent)

Retourne un objet JSON avec:
1. La transcription complète avec timestamps précis (chaque phrase/vers)
2. Le style artistique déduit de la musique ET des paroles ET du titre
3. Le plan de scènes visuelles couvrant toute la durée

FORMAT EXACT:
{{
  "style": "style en anglais pour les prompts vidéo",
  "style_analysis": "explication en français du style choisi et du lien avec le titre",
  "segments": [
    {{"id": 0, "start": 0.0, "end": 3.5, "text": "paroles ici"}}
  ],
  "scenes": [
    {{
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 5.0,
      "visual_prompt": "description cinématographique détaillée illustrant les paroles de cette section",
      "motion_prompt": "mouvement de caméra",
      "consistency_tags": "personnage principal + décor récurrent"
    }}
  ]
}}

Règles scènes: entre 4 et 6 secondes CHACUNE (jamais plus !), couvrir TOUTE la durée, visual_prompts en anglais cinématographique liés aux paroles."""

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

    scenes = _enforce_scene_durations(scenes)
    return scenes, style_en, style_analysis, segments


def generate_scenarios(
    segments: list[dict],
    total_duration: float | None = None,
    api_key: str = GEMINI_API_KEY,
    song_title: str = "",
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
TITRE DE LA CHANSON: "{song_title}"

PAROLES AVEC TIMESTAMPS:
{lyrics_text}

Le titre "{song_title}" est le THÈME CENTRAL du clip — chaque scène doit s'y rattacher.
Chaque visual_prompt doit ILLUSTRER ce que les paroles disent à ce moment précis.
Chaque scène = entre 4 et 6 secondes (jamais plus de 6s !).
Reste RÉALISTE et cinématographique.

Génère le plan de scènes en JSON."""

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

    scenes = _enforce_scene_durations(scenes)
    return scenes, style_en, style_analysis


def _enforce_scene_durations(scenes: list[dict], max_dur: float = SCENE_DURATION_MAX) -> list[dict]:
    """
    Découpe les scènes trop longues (>max_dur) en sous-scènes de ~5s.
    Renumérate les scene_id séquentiellement.
    """
    fixed: list[dict] = []
    for s in scenes:
        start = float(s.get("start_time") or 0)
        end = float(s.get("end_time") or start + 5)
        dur = end - start

        if dur <= max_dur + 0.5:
            fixed.append(s)
        else:
            # Split en chunks de ~5s
            chunk_dur = 5.0
            n_chunks = max(1, round(dur / chunk_dur))
            actual_chunk = dur / n_chunks
            for i in range(n_chunks):
                chunk_start = start + i * actual_chunk
                chunk_end = start + (i + 1) * actual_chunk
                fixed.append({
                    **s,
                    "start_time": round(chunk_start, 2),
                    "end_time": round(chunk_end, 2),
                })
            print(f"  [FIX] Scène {s.get('scene_id','?')} ({dur:.1f}s) → {n_chunks} sous-scènes de {actual_chunk:.1f}s")

    # Renuméroter
    for i, sc in enumerate(fixed):
        sc["scene_id"] = i + 1

    if len(fixed) != len(scenes):
        print(f"  [FIX] {len(scenes)} scènes → {len(fixed)} après découpe")

    return fixed


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
