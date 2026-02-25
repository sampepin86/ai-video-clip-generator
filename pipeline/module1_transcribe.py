"""
Module 1 — Audio transcription with OpenAI Whisper.
Prend un MP3 et retourne un dict de segments avec timestamps.
Supporte la transcription locale ET distante (RunPod via SSH).
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

# Paramètres SSH RunPod (importés ici pour éviter une dépendance circulaire)
_RUNPOD_HOST = "195.26.233.74"
_RUNPOD_PORT = 19123
_RUNPOD_USER = "root"
_RUNPOD_KEY  = "~/.ssh/id_ed25519"
_RUNPOD_VENV = "/workspace/runpod-slim/ComfyUI/.venv/bin/python3"
_RUNPOD_TMP  = "/workspace/tmp_transcribe"


class Segment(TypedDict):
    id: int
    start: float
    end: float
    text: str


# ── Transcription locale ──────────────────────────────────────────────────────

def transcribe(audio_path: str | Path, model_size: str = "large-v3") -> list[Segment]:
    """
    Transcrit un fichier audio localement avec Whisper.

    Args:
        audio_path: Chemin vers le fichier MP3/WAV.
        model_size: Modèle Whisper à utiliser (tiny|base|medium|large-v3).

    Returns:
        Liste de segments: [{id, start, end, text}, ...]
    """
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise ImportError("Installe openai-whisper: pip install openai-whisper") from e

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")

    print(f"[Whisper-local] Chargement modèle '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"[Whisper-local] Transcription de '{audio_path.name}'...")
    result = model.transcribe(str(audio_path), word_timestamps=False)

    segments: list[Segment] = [
        {
            "id": i,
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        }
        for i, seg in enumerate(result["segments"])
    ]

    print(f"[Whisper-local] {len(segments)} segments extraits. Durée totale: {segments[-1]['end']:.1f}s")
    return segments


# ── Transcription distante (RunPod GPU) ───────────────────────────────────────

def transcribe_remote(
    audio_path: str | Path,
    model_size: str = "large-v3",
    ssh_host: str = _RUNPOD_HOST,
    ssh_port: int = _RUNPOD_PORT,
    ssh_key: str = _RUNPOD_KEY,
) -> list[Segment]:
    """
    Transcrit via SSH sur RunPod (GPU RTX 6000 Ada).
    Upload le MP3, Whisper tourne sur le GPU distant, récupère le JSON.

    Args:
        audio_path: Chemin local vers le MP3.
        model_size: Modèle Whisper (large-v3 recommandé sur GPU 48GB).
        ssh_host / ssh_port / ssh_key: Paramètres SSH RunPod.

    Returns:
        Liste de segments: [{id, start, end, text}, ...]
    """
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")

    ssh_base = [
        "ssh", "-p", str(ssh_port),
        "-i", str(Path(ssh_key).expanduser()),
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{_RUNPOD_USER}@{ssh_host}",
    ]

    # 1. Créer le dossier tmp sur RunPod
    print(f"[Whisper-remote] Préparation RunPod ({ssh_host}:{ssh_port})...")
    subprocess.run(ssh_base + [f"mkdir -p {_RUNPOD_TMP}"], check=True, capture_output=True)

    remote_audio = f"{_RUNPOD_TMP}/{audio_path.name}"
    remote_json  = f"{_RUNPOD_TMP}/{audio_path.stem}.segments.json"

    # 2. Upload le MP3
    print(f"[Whisper-remote] Upload '{audio_path.name}'...")
    scp_cmd = [
        "scp", "-P", str(ssh_port),
        "-i", str(Path(ssh_key).expanduser()),
        "-o", "StrictHostKeyChecking=no",
        str(audio_path),
        f"{_RUNPOD_USER}@{ssh_host}:{remote_audio}",
    ]
    subprocess.run(scp_cmd, check=True, capture_output=True)

    # 3. Lancer Whisper sur RunPod
    print(f"[Whisper-remote] Transcription GPU (modèle: {model_size})...")
    whisper_script = f"""
import json, sys
sys.path.insert(0, '/workspace/runpod-slim/ComfyUI/.venv/lib/python3.11/site-packages')
import whisper
model = whisper.load_model('{model_size}')
result = model.transcribe('{remote_audio}', word_timestamps=False)
segments = [
    {{'id': i, 'start': round(s['start'], 3), 'end': round(s['end'], 3), 'text': s['text'].strip()}}
    for i, s in enumerate(result['segments'])
]
with open('{remote_json}', 'w') as f:
    json.dump(segments, f, indent=2, ensure_ascii=False)
print(f'OK: {{len(segments)}} segments, durée={{segments[-1]["end"]:.1f}}s')
"""
    run_cmd = ssh_base + [
        f"source /workspace/runpod-slim/ComfyUI/.venv/bin/activate && "
        f"{_RUNPOD_VENV} -c \"{whisper_script.replace(chr(10), ';')}\""
    ]
    # Utiliser un script uploadé c'est plus propre qu'une commande inline complexe
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(whisper_script)
        tmp_script = tf.name

    remote_script = f"{_RUNPOD_TMP}/transcribe.py"
    # Upload le script
    scp_script = [
        "scp", "-P", str(ssh_port),
        "-i", str(Path(ssh_key).expanduser()),
        "-o", "StrictHostKeyChecking=no",
        tmp_script,
        f"{_RUNPOD_USER}@{ssh_host}:{remote_script}",
    ]
    subprocess.run(scp_script, check=True, capture_output=True)
    Path(tmp_script).unlink(missing_ok=True)

    result = subprocess.run(
        ssh_base + [
            f"source /workspace/runpod-slim/ComfyUI/.venv/bin/activate && "
            f"{_RUNPOD_VENV} {remote_script}"
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Whisper RunPod error:\n{result.stderr[-1000:]}")
    print(f"[Whisper-remote] {result.stdout.strip()}")

    # 4. Télécharger le JSON résultat
    print(f"[Whisper-remote] Téléchargement résultats...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        local_json = tf.name

    scp_dl = [
        "scp", "-P", str(ssh_port),
        "-i", str(Path(ssh_key).expanduser()),
        "-o", "StrictHostKeyChecking=no",
        f"{_RUNPOD_USER}@{ssh_host}:{remote_json}",
        local_json,
    ]
    subprocess.run(scp_dl, check=True, capture_output=True)

    with open(local_json, encoding="utf-8") as f:
        segments: list[Segment] = json.load(f)
    Path(local_json).unlink(missing_ok=True)

    print(f"[Whisper-remote] ✅ {len(segments)} segments | durée: {segments[-1]['end']:.1f}s")
    return segments


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_segments(segments: list[Segment], output_path: str | Path) -> None:
    """Sauvegarde les segments en JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[Whisper] Segments sauvegardés → {output_path}")


def load_segments(json_path: str | Path) -> list[Segment]:
    """Charge des segments depuis un fichier JSON (cache)."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python module1_transcribe.py audio.mp3 [model_size] [--remote]")
        sys.exit(1)
    audio = sys.argv[1]
    size  = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
    remote = "--remote" in sys.argv
    segs = transcribe_remote(audio, size) if remote else transcribe(audio, size)
    out = Path(audio).with_suffix(".segments.json")
    save_segments(segs, out)
    print(json.dumps(segs[:3], indent=2, ensure_ascii=False))
