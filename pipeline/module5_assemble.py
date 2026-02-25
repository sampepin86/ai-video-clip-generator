"""
Module 5 — Post-production: assemblage des clips + synchronisation audio.
Fusionne les fichiers MP4 générés et synchronise avec l'audio original.
"""
from __future__ import annotations
import json
from pathlib import Path


def assemble_video(
    audio_path: str | Path,
    video_segments: list[str | Path],
    scenes: list[dict] | None = None,
    output_path: str | Path = "output/music_video_final.mp4",
    target_resolution: tuple[int, int] = (832, 480),
    fps: int = 24,
    fade_duration: float = 0.0,
) -> Path:
    """
    Assemble les clips MP4 et synchronise avec l'audio original.

    Args:
        audio_path: Chemin vers le MP3 original
        video_segments: Liste ordonnée des chemins MP4 (scène 1, 2, ...)
        scenes: Métadonnées des scènes (pour ajuster les durées exactes)
        output_path: Chemin de sortie du fichier final
        target_resolution: Résolution cible (W, H) pour normaliser tous les clips
        fps: FPS de la sortie
        fade_duration: Durée du fondu enchaîné entre scènes (0 = aucun)

    Returns:
        Chemin vers le fichier vidéo final
    """
    try:
        from moviepy.editor import (  # type: ignore
            VideoFileClip,
            AudioFileClip,
            concatenate_videoclips,
        )
    except ImportError as e:
        raise ImportError("Installe: pip install moviepy") from e

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[PostProd] Assemblage de {len(video_segments)} clips...")
    print(f"  Audio:  {audio_path.name}")
    print(f"  Sortie: {output_path}")

    # ── Charge et normalise chaque clip ──────────────────────────────────────
    clips = []
    for i, seg_path in enumerate(video_segments):
        seg_path = Path(seg_path)
        if not seg_path.exists():
            print(f"  [WARN] Clip manquant: {seg_path}, skipped")
            continue

        clip = VideoFileClip(str(seg_path))

        # Redimensionne si nécessaire
        if clip.size != list(target_resolution):
            clip = clip.resize(target_resolution)

        # Ajuste la durée exacte si les métadonnées de scène sont disponibles
        if scenes and i < len(scenes):
            expected_duration = scenes[i]["end_time"] - scenes[i]["start_time"]
            if abs(clip.duration - expected_duration) > 0.1:
                clip = clip.subclip(0, min(clip.duration, expected_duration))

        clips.append(clip)
        print(f"  [{i+1}/{len(video_segments)}] {seg_path.name} | {clip.duration:.2f}s | {clip.size}")

    if not clips:
        raise ValueError("Aucun clip valide à assembler")

    # ── Concatène ─────────────────────────────────────────────────────────────
    method = "compose" if fade_duration > 0 else "chain"

    if fade_duration > 0:
        # Fondus enchaînés
        from moviepy.editor import concatenate_videoclips
        clips_with_fade = []
        for i, c in enumerate(clips):
            if i > 0:
                c = c.crossfadein(fade_duration)
            clips_with_fade.append(c)
        final_video = concatenate_videoclips(clips_with_fade, padding=-fade_duration, method="compose")
    else:
        final_video = concatenate_videoclips(clips, method="chain")

    print(f"\n  Durée totale vidéo: {final_video.duration:.1f}s")

    # ── Synchronise l'audio ───────────────────────────────────────────────────
    audio = AudioFileClip(str(audio_path))
    print(f"  Durée audio: {audio.duration:.1f}s")

    # Coupe la vidéo à la durée de l'audio (ou vice versa)
    audio_duration = audio.duration
    video_duration = final_video.duration

    if video_duration > audio_duration:
        print(f"  [INFO] Vidéo plus longue que l'audio, coupe à {audio_duration:.1f}s")
        final_video = final_video.subclip(0, audio_duration)
    elif video_duration < audio_duration:
        print(f"  [INFO] Vidéo plus courte ({video_duration:.1f}s) que l'audio ({audio_duration:.1f}s)")
        # Allonge la dernière frame pour remplir
        from moviepy.editor import ImageClip
        last_frame = clips[-1].get_frame(clips[-1].duration - 0.1)
        padding_clip = ImageClip(last_frame, duration=audio_duration - video_duration)
        padding_clip = padding_clip.set_fps(fps)
        from moviepy.editor import concatenate_videoclips
        final_video = concatenate_videoclips([final_video, padding_clip])

    final_video = final_video.set_audio(audio)

    # ── Encode ────────────────────────────────────────────────────────────────
    print(f"\n  Encodage → {output_path} ({fps}fps H264)...")
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
        audio_bitrate="320k",
        threads=4,
        logger="bar",
    )

    # ── Nettoyage ─────────────────────────────────────────────────────────────
    for c in clips:
        c.close()
    audio.close()
    final_video.close()

    print(f"\n[PostProd] ✓ Vidéo finale: {output_path}")
    print(f"  Taille: {output_path.stat().st_size / 1024**2:.1f} MB")
    return output_path


def generate_srt(segments: list[dict], output_path: str | Path) -> Path:
    """Génère un fichier de sous-titres SRT depuis les segments Whisper."""
    output_path = Path(output_path)
    lines = []
    for i, seg in enumerate(segments):
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        lines.append(f"{i+1}\n{start} --> {end}\n{seg['text']}\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PostProd] SRT généré → {output_path}")
    return output_path


def _seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    import sys, glob

    if len(sys.argv) < 3:
        print("Usage: python module5_assemble.py audio.mp3 clips_dir/ [output.mp4]")
        sys.exit(1)

    audio = sys.argv[1]
    clips_dir = Path(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "output/music_video_final.mp4"

    clips = sorted(clips_dir.glob("scene_*.mp4"))
    if not clips:
        print(f"Aucun clip trouvé dans {clips_dir}")
        sys.exit(1)

    print(f"Clips trouvés: {len(clips)}")
    assemble_video(audio, clips, output_path=out)
