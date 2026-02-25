#!/usr/bin/env python3
"""
AI Video Clip Generator — Orchestrateur principal
Transforme un MP3 en clip vidéo synchronisé via WAN 2.1 I2V sur RunPod/ComfyUI.

Usage:
    python main.py audio.mp3 [--style "cinematic, moody"] [--output output/] [--resume]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
COMFYUI_URL     = os.environ.get("COMFYUI_BASE_URL", "https://{}-8188.proxy.runpod.net".format(os.environ.get("RUNPOD_POD_ID", "bn7d2bx6engu1k")))
WHISPER_MODEL   = "large-v3"
DEFAULT_STYLE   = "cinematic music video, moody atmosphere, professional photography"

# Paramètres WAN I2V (adaptés RTX 6000 Ada 48GB)
WAN_PARAMS = {
    "width": 832,
    "height": 480,
    "fps": 24,
    "steps": 20,
    "cfg": 6.0,
}


def run_pipeline(
    audio_path: str,
    style: str = DEFAULT_STYLE,
    output_dir: str = "output",
    resume: bool = True,
    skip_transcribe: bool = False,
    skip_scenarios: bool = False,
) -> Path:
    """
    Pipeline complet:
        MP3 → Whisper → Gemini → ComfyUI WAN I2V → MoviePy → MP4
    """
    t_global = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio introuvable: {audio_path}")

    print("=" * 60)
    print(" AI VIDEO CLIP GENERATOR")
    print(f" Audio:  {audio_path.name}")
    print(f" Style:  {style}")
    print(f" Output: {output_dir}")
    print(f" GPU:    RTX 6000 Ada (RunPod) | WAN 2.1 I2V 480P")
    print("=" * 60)

    # ── Module 1: Transcription Whisper ───────────────────────────────────────
    segments_file = output_dir / f"{audio_path.stem}.segments.json"

    if not skip_transcribe and not (resume and segments_file.exists()):
        print("\n[1/5] TRANSCRIPTION WHISPER")
        from pipeline.module1_transcribe import transcribe, save_segments
        segments = transcribe(audio_path, model_size=WHISPER_MODEL)
        save_segments(segments, segments_file)
    else:
        print(f"\n[1/5] TRANSCRIPTION [SKIP] Chargement depuis {segments_file.name}")
        from pipeline.module1_transcribe import load_segments
        segments = load_segments(segments_file)

    print(f"  → {len(segments)} segments | Durée: {segments[-1]['end']:.1f}s")

    # ── Module 2: Génération scénarios Gemini ─────────────────────────────────
    scenes_file = output_dir / f"{audio_path.stem}.scenes.json"

    if not skip_scenarios and not (resume and scenes_file.exists()):
        print("\n[2/5] GÉNÉRATION SCÉNARIOS (Gemini)")
        from pipeline.module2_scenarios import generate_scenarios, save_scenes
        scenes = generate_scenarios(segments, style=style, api_key=GEMINI_API_KEY)
        save_scenes(scenes, str(scenes_file))
    else:
        print(f"\n[2/5] SCÉNARIOS [SKIP] Chargement depuis {scenes_file.name}")
        with open(scenes_file) as f:
            scenes = json.load(f)

    print(f"  → {len(scenes)} scènes planifiées")

    # ── Module 3+4: Génération I2V WAN via ComfyUI ────────────────────────────
    clips_dir = output_dir / "clips"
    print(f"\n[3-4/5] GÉNÉRATION CLIPS WAN I2V ({len(scenes)} scènes)")
    print(f"  ComfyUI: {COMFYUI_URL}")

    from pipeline.module4_generate import generate_all_scenes
    clips = generate_all_scenes(
        scenes=scenes,
        output_dir=clips_dir,
        comfyui_url=COMFYUI_URL,
        resume=resume,
        generation_params=WAN_PARAMS,
    )

    if not clips:
        raise RuntimeError("Aucun clip généré!")

    print(f"  → {len(clips)} clips disponibles")

    # ── Module 5: Assemblage final ────────────────────────────────────────────
    final_output = output_dir / f"{audio_path.stem}_final.mp4"
    print(f"\n[5/5] ASSEMBLAGE FINAL")

    from pipeline.module5_assemble import assemble_video
    result = assemble_video(
        audio_path=audio_path,
        video_segments=clips,
        scenes=scenes,
        output_path=final_output,
        target_resolution=(WAN_PARAMS["width"], WAN_PARAMS["height"]),
        fps=WAN_PARAMS["fps"],
    )

    elapsed = time.time() - t_global
    print(f"\n{'=' * 60}")
    print(f" ✓ PIPELINE TERMINÉ en {elapsed/60:.1f} minutes")
    print(f" Fichier final: {result}")
    print(f"{'=' * 60}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Video Clip Generator — MP3 → Clip vidéo IA"
    )
    parser.add_argument("audio", help="Chemin vers le fichier MP3")
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        help="Style artistique global pour les prompts Gemini",
    )
    parser.add_argument(
        "--output", default="output", help="Dossier de sortie"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reprend depuis le dernier checkpoint (défaut: activé)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Repart de zéro",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip Whisper si segments.json existe déjà",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip Gemini si scenes.json existe déjà",
    )
    parser.add_argument(
        "--scenes-only",
        action="store_true",
        help="Génère uniquement transcription + scénarios (sans I2V ni assemblage)",
    )

    args = parser.parse_args()

    # Mode scènes uniquement (planification sans génération)
    if args.scenes_only:
        args.skip_transcribe = False
        args.skip_scenarios = False
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = Path(args.audio)

        from pipeline.module1_transcribe import transcribe, save_segments
        segs = transcribe(audio_path, WHISPER_MODEL)
        seg_file = output_dir / f"{audio_path.stem}.segments.json"
        save_segments(segs, seg_file)

        from pipeline.module2_scenarios import generate_scenarios, save_scenes
        scenes = generate_scenarios(segs, style=args.style, api_key=GEMINI_API_KEY)
        sc_file = output_dir / f"{audio_path.stem}.scenes.json"
        save_scenes(scenes, str(sc_file))

        print(json.dumps(scenes, indent=2, ensure_ascii=False))
        sys.exit(0)

    run_pipeline(
        audio_path=args.audio,
        style=args.style,
        output_dir=args.output,
        resume=args.resume,
        skip_transcribe=args.skip_transcribe,
        skip_scenarios=args.skip_scenarios,
    )


if __name__ == "__main__":
    main()
