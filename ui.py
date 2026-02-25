"""
Interface Graphique — AI Video Clip Generator
Gradio UI pour piloter tout le pipeline (local ou RunPod).
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from pathlib import Path

# Load .env file if present (for RunPod deployment)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import gradio as gr

# Ajoute le workspace root + pipeline/ au path
# (workspace root pour `from pipeline.xxx`, pipeline/ pour les imports internes)
_root = Path(__file__).parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "pipeline"))

from pipeline.config import (
    MODELS, DEFAULT_MODEL_ID, GEMINI_API_KEY,
    COMFYUI_BASE_URL, get_model, list_models_for_ui,
)
from pipeline.module1_transcribe import transcribe, transcribe_remote, save_segments, load_segments
from pipeline.module2_scenarios import generate_scenarios, analyze_audio_direct, save_scenes
from pipeline.module3_comfyui_client import ComfyUIClient
from pipeline.module4_generate import generate_all_scenes, _duration_to_wan_frames
from pipeline.module5_assemble import assemble_video

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── État global de génération ─────────────────────────────────────────────────
_state = {
    "segments": None,
    "scenes": None,
    "style_en": "",
    "style_analysis": "",
    "clips": [],
    "running": False,
    "stop_requested": False,
    "log": [],
}


def _log(msg: str):
    _state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print(msg)


def stop_generation():
    """Demande l'arrêt de la génération en cours."""
    if _state["running"]:
        _state["stop_requested"] = True
        _log("⏹ Arrêt demandé… la scène en cours va se terminer.")
        return "⏹ Arrêt demandé — fin de la scène en cours…"
    return "ℹ️ Aucune génération en cours"


# ── Fonctions du pipeline ─────────────────────────────────────────────────────

def check_comfyui_status():
    import urllib.request
    _UA = "Mozilla/5.0 (compatible; AI-Video-Generator/1.0)"
    try:
        req = urllib.request.Request(
            f"{COMFYUI_BASE_URL}/system_stats",
            headers={"User-Agent": _UA},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            s = json.loads(r.read())
        devs = s.get("devices", [{}])
        gpu = devs[0].get("name", "?") if devs else "?"
        vram = devs[0].get("vram_total", 0) // 1024**2 if devs else 0
        return f"✅ ComfyUI connecté | GPU: {gpu} | VRAM: {vram}MB"
    except Exception as e:
        return f"❌ ComfyUI hors ligne: {e}"


def get_available_models_status():
    client = ComfyUIClient()
    avail = client.list_available_models()
    diff = avail.get("diffusion_models", [])
    ckpts = avail.get("checkpoints", [])
    rows = []
    for mid, m in MODELS.items():
        pool = ckpts if m.is_checkpoint else diff
        present = m.filename in pool
        status = "✅" if present else "⬛ non téléchargé"
        rows.append([m.label, m.version, f"{m.default_width}×{m.default_height}",
                     f"{m.default_frames}f", status])
    return rows


def step_analyze_audio(audio_file, force_reanalyze=False, progress=gr.Progress()):
    """Mode Gemini natif: MP3 → transcription + style + scènes en 1 appel.
    Si le fichier JSON existe déjà et force_reanalyze=False, charge depuis le cache.
    """
    if audio_file is None:
        return "❌ Aucun fichier audio fourni", None, "", "", gr.update()

    _state["log"] = []
    audio_stem = Path(audio_file).stem
    cache_path = OUTPUT_DIR / f"{audio_stem}.scenes.json"

    # ── Chargement depuis cache si disponible ──────────────────────────────────
    if not force_reanalyze and cache_path.exists():
        try:
            _log(f"📦 Cache trouvé: {cache_path.name} — chargement sans appeler Gemini...")
            progress(0.2, desc="Chargement cache...")
            with open(cache_path) as f:
                data = json.load(f)
            scenes = data.get("scenes", [])
            style_en = data.get("style", "")
            style_analysis = data.get("style_analysis", "")

            # Segments (fichier séparé)
            seg_path = OUTPUT_DIR / f"{audio_stem}.segments.json"
            segments = []
            if seg_path.exists():
                with open(seg_path) as f:
                    segments = json.load(f)

            _state["scenes"] = scenes
            _state["segments"] = segments
            _state["style_en"] = style_en
            _state["style_analysis"] = style_analysis

            total_dur = sum((s.get("end_time") or 0) - (s.get("start_time") or 0) for s in scenes)
            preview_rows = [
                [s.get("scene_id", i+1), f"{s.get('start_time',0):.1f}→{s.get('end_time',0):.1f}s",
                 str(s.get("visual_prompt", ""))[:60] + "...", s.get("motion_prompt", "")]
                for i, s in enumerate(scenes[:15])
            ]
            _log(f"✅ Chargé depuis cache | {len(scenes)} scènes | Style: {style_en}")
            progress(1.0, desc="Cache chargé")
            return (
                f"✅ {len(scenes)} scènes chargées depuis cache ({total_dur:.0f}s)",
                preview_rows,
                style_en,
                style_analysis,
                gr.update(interactive=True),
            )
        except Exception as e:
            _log(f"⚠️  Cache illisible ({e}) — relance Gemini...")

    # ── Appel Gemini ────────────────────────────────────────────────────────────
    _log(f"🎵 Analyse audio Gemini 2.5 Flash | {Path(audio_file).name}...")
    progress(0.1, desc="Upload fichier vers Gemini...")

    try:
        scenes, style_en, style_analysis, segments = analyze_audio_direct(
            audio_file, api_key=GEMINI_API_KEY
        )
        _state["scenes"] = scenes
        _state["segments"] = segments
        _state["style_en"] = style_en
        _state["style_analysis"] = style_analysis

        audio_stem = Path(audio_file).stem
        save_scenes(scenes, str(OUTPUT_DIR / f"{audio_stem}.scenes.json"), style_en, style_analysis)
        save_segments(segments, OUTPUT_DIR / f"{audio_stem}.segments.json")

        total_dur = sum((s.get("end_time") or 0) - (s.get("start_time") or 0) for s in scenes)
        preview_rows = [
            [s.get("scene_id",i+1), f"{s.get('start_time',0):.1f}→{s.get('end_time',0):.1f}s",
             str(s.get("visual_prompt",""))[:60] + "...", s.get("motion_prompt","")]
            for i, s in enumerate(scenes[:15])
        ]
        _log(f"✅ {len(segments)} segments | {len(scenes)} scènes | Style: {style_en}")
        progress(1.0, desc="Analyse terminée")
        return (
            f"✅ {len(scenes)} scènes planifiées ({total_dur:.0f}s couverts)",
            preview_rows,
            style_en,
            style_analysis,
            gr.update(interactive=True),
        )
    except Exception as e:
        _log(f"❌ Erreur Gemini Audio: {e}")
        return f"❌ Erreur: {e}", None, "", "", gr.update(interactive=False)


def step1_transcribe(audio_file, whisper_model, transcribe_mode, progress=gr.Progress()):
    if audio_file is None:
        return "❌ Aucun fichier audio fourni", None, gr.update()

    _state["log"] = []
    remote = transcribe_mode == "RunPod (GPU)"
    mode_label = "RunPod GPU" if remote else "local"
    _log(f"🎵 Transcription {mode_label} | {Path(audio_file).name} | Whisper {whisper_model}...")
    progress(0, desc=f"Chargement Whisper {mode_label}...")

    try:
        if remote:
            segments = transcribe_remote(audio_file, model_size=whisper_model)
        else:
            segments = transcribe(audio_file, model_size=whisper_model)

        _state["segments"] = segments

        seg_file = OUTPUT_DIR / f"{Path(audio_file).stem}.segments.json"
        save_segments(segments, seg_file)

        duration = segments[-1]["end"]
        preview_text = "\n".join(
            f"[{s['start']:.1f}s → {s['end']:.1f}s] {s['text']}"
            for s in segments[:8]
        )
        if len(segments) > 8:
            preview_text += f"\n... ({len(segments)} segments au total)"

        _log(f"✅ {len(segments)} segments | Durée: {duration:.1f}s")
        progress(1.0, desc="Transcription terminée")
        return (
            f"✅ {len(segments)} segments extraits | Durée totale: {duration:.1f}s",
            preview_text,
            gr.update(interactive=True),
        )
    except Exception as e:
        _log(f"❌ Erreur transcription: {e}")
        return f"❌ Erreur: {e}", None, gr.update(interactive=False)


def step2_scenarios(audio_file, progress=gr.Progress()):
    if _state["segments"] is None:
        return "❌ Lance d'abord la transcription", None, "", "", gr.update()

    _log("🎬 Analyse Gemini: paroles → style artistique + scènes...")
    progress(0, desc="Appel Gemini...")

    try:
        audio_stem = Path(audio_file).stem if audio_file else "output"
        scenes, style_en, style_analysis = generate_scenarios(
            _state["segments"],
            api_key=GEMINI_API_KEY,
            song_title=audio_stem,
        )
        _state["scenes"] = scenes
        _state["style_en"] = style_en
        _state["style_analysis"] = style_analysis

        sc_file = OUTPUT_DIR / f"{audio_stem}.scenes.json"
        save_scenes(scenes, str(sc_file), style_en, style_analysis)

        total_dur = sum((s.get("end_time") or 0) - (s.get("start_time") or 0) for s in scenes)
        preview_rows = [
            [s.get("scene_id", i+1), f"{s.get('start_time',0):.1f}→{s.get('end_time',0):.1f}s",
             str(s.get("visual_prompt",""))[:60] + "...", s.get("motion_prompt","")]
            for i, s in enumerate(scenes[:15])
        ]
        _log(f"✅ Style: {style_en}")
        _log(f"✅ {len(scenes)} scènes générées | Durée totale: {total_dur:.1f}s")
        progress(1.0, desc="Scénarios prêts")
        return (
            f"✅ {len(scenes)} scènes planifiées",
            preview_rows,
            style_en,
            style_analysis,
            gr.update(interactive=True),
        )
    except Exception as e:
        _log(f"❌ Erreur Gemini: {e}")
        return f"❌ Erreur: {e}", None, "", "", gr.update(interactive=False)


def step3_generate(
    audio_file, init_image, model_id, num_frames, width, height,
    steps, cfg, resume, progress=gr.Progress()
):
    if _state["scenes"] is None:
        return "❌ Génère d'abord les scénarios", None, gr.update()

    _state["stop_requested"] = False
    _state["running"] = True
    model = get_model(model_id)
    clips_dir = OUTPUT_DIR / "clips"
    _log(f"🎥 Début génération: {len(_state['scenes'])} scènes | Modèle: {model.label}")
    _log(f"   Résolution: {width}×{height} | Frames: {num_frames} | Steps: {steps} | CFG: {cfg}")

    gen_params = {
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "fps": 24,
        "steps": int(steps),
        "cfg": float(cfg),
    }

    # Charger l'image de départ si fournie
    custom_init_b64 = None
    if init_image:
        import base64
        with open(init_image, "rb") as f:
            custom_init_b64 = base64.b64encode(f.read()).decode()
        _log(f"🖼️ Image de départ personnalisée chargée")

    progress_steps = [0]

    def on_scene_progress(scene_idx, total, status):
        pct = scene_idx / max(total, 1)
        progress(pct, desc=f"Scène {scene_idx+1}/{total}: {status}")
        _log(f"  → Scène {scene_idx+1}/{total}")

    try:
        clips = generate_all_scenes(
            scenes=_state["scenes"],
            output_dir=clips_dir,
            model_id=model_id,
            resume=resume,
            generation_params=gen_params,
            on_progress=on_scene_progress,
            stop_flag=_state,
            custom_init_image_b64=custom_init_b64,
        )
        _state["clips"] = [c for c in clips if c is not None]
        _state["running"] = False

        if _state["stop_requested"]:
            _log(f"⏹ Génération arrêtée | {len(_state['clips'])} clips générés")
            _state["stop_requested"] = False
            return (
                f"⏹ Arrêté — {len(_state['clips'])} clips générés sur {len(_state['scenes'])} scènes",
                None,
                gr.update(interactive=bool(_state['clips'])),
            )

        _log(f"✅ {len(_state['clips'])} clips générés")
        progress(1.0, desc="Génération terminée")
        return (
            f"✅ {len(_state['clips'])} clips générés sur {len(_state['scenes'])} scènes",
            None,
            gr.update(interactive=True),
        )
    except Exception as e:
        _log(f"❌ Erreur génération: {e}")
        return f"❌ Erreur: {e}", None, gr.update(interactive=False)


def step4_assemble(audio_file, progress=gr.Progress()):
    if not _state["clips"] or audio_file is None:
        return "❌ Clips ou audio manquants", None

    _log("🎞️ Assemblage final...")
    progress(0, desc="Assemblage MoviePy...")

    try:
        audio_stem = Path(audio_file).stem
        out = OUTPUT_DIR / f"{audio_stem}_final.mp4"

        first_clip = Path(_state["clips"][0])
        from moviepy.editor import VideoFileClip
        vc = VideoFileClip(str(first_clip))
        res = (vc.size[0], vc.size[1])
        vc.close()

        result = assemble_video(
            audio_path=audio_file,
            video_segments=_state["clips"],
            scenes=_state["scenes"],
            output_path=out,
            target_resolution=res,
            fps=24,
        )
        _log(f"✅ Vidéo finale: {result} ({result.stat().st_size/1024**2:.1f}MB)")
        progress(1.0, desc="Terminé !")
        return f"✅ Vidéo finale prête: {result}", str(result)
    except Exception as e:
        _log(f"❌ Erreur assemblage: {e}")
        return f"❌ Erreur: {e}", None


def run_full_pipeline(
    audio_file, init_image, model_id, num_frames, width, height,
    steps, cfg, resume, progress=gr.Progress()
):
    """Lance le pipeline complet d'un coup."""
    if audio_file is None:
        return "❌ Aucun fichier audio", None, "", "", "", []

    logs = []

    # Step 1+2 combiné via Gemini Audio
    progress(0.05, desc="Analyse Gemini Audio...")
    r1, _, style_en, style_analysis, _ = step_analyze_audio(audio_file)
    logs.append(r1)
    if "❌" in r1:
        return r1, None, "", "", "\n".join(_state["log"]), []

    # Step 3
    progress(0.20, desc="Génération vidéo I2V...")
    r3, _, _ = step3_generate(
        audio_file, init_image, model_id, num_frames, width, height, steps, cfg, resume
    )
    logs.append(r3)
    if "❌" in r3:
        return r3, None, style_en, style_analysis, "\n".join(_state["log"]), []

    # Step 4
    progress(0.95, desc="Assemblage final...")
    r4, video_path = step4_assemble(audio_file)
    logs.append(r4)

    progress(1.0, desc="Terminé !")
    return "\n".join(logs), video_path, style_en, style_analysis, "\n".join(_state["log"]), []


def get_logs():
    return "\n".join(_state["log"][-60:]) if _state["log"] else "Aucun log"


def update_model_params(model_id):
    """Met à jour les paramètres par défaut quand le modèle change."""
    m = get_model(model_id)
    return (
        gr.update(value=m.default_frames, minimum=min(m.frame_options),
                  maximum=max(m.frame_options),
                  info=f"Options: {m.frame_options}"),
        gr.update(value=m.default_width),
        gr.update(value=m.default_height),
        gr.update(value=m.default_steps),
        gr.update(value=m.default_cfg),
        f"**{m.label}**\n\n{m.description}\n\n"
        f"- Version: WAN {m.version}\n"
        f"- Résolution native: {m.default_width}×{m.default_height}\n"
        f"- Frames: {m.default_frames} ({m.default_frames/24:.1f}s @ 24fps)\n"
        f"- Taille: ~{m.size_gb:.0f}GB\n"
        f"- Qualité: {'⭐' * m.quality_score}",
    )


# ── Construction de l'UI ──────────────────────────────────────────────────────

def build_ui():
    css = """
    .container { max-width: 1200px; margin: auto; }
    .model-card { background: #1a1a2e; padding: 12px; border-radius: 8px; }
    .status-ok { color: #00ff88; font-weight: bold; }
    .status-err { color: #ff4444; font-weight: bold; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="AI Video Clip Generator",
    ) as app:

        gr.Markdown("# 🎬 AI Video Clip Generator\n*MP3 → Clip vidéo synchronisé via WAN 2.2 I2V sur RunPod*")

        with gr.Tabs():

            # ── TAB 1: Pipeline complet ───────────────────────────────────────
            with gr.TabItem("🚀 Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="🎵 Fichier MP3",
                            type="filepath",
                            elem_id="audio_input",
                        )
                        init_image_input = gr.Image(
                            label="🖼️ Image de départ (optionnel)",
                            type="filepath",
                            elem_id="init_image",
                        )
                        style_display = gr.Textbox(
                            label="🎨 Style artistique (généré par Gemini — modifiable)",
                            placeholder="Sera généré automatiquement à l'étape 2 (Scénarios)…",
                            lines=2,
                            interactive=True,
                        )
                        style_analysis_display = gr.Textbox(
                            label="💡 Analyse Gemini",
                            placeholder="Explication du style choisi…",
                            lines=3,
                            interactive=False,
                        )
                with gr.Row():
                    btn_gemini   = gr.Button("🎵 Analyser avec Gemini", variant="primary", size="lg")
                    force_reanalyze_cb = gr.Checkbox(
                        label="🔄 Forcer re-analyse (ignorer cache)",
                        value=False,
                        scale=0,
                    )

                with gr.Accordion("🔧 Fallback Whisper (si Gemini échoue)", open=False):
                    with gr.Row():
                        whisper_model = gr.Dropdown(
                            label="🎤 Modèle Whisper",
                            choices=["tiny", "base", "small", "medium", "large-v3"],
                            value="large-v3",
                        )
                        transcribe_mode = gr.Radio(
                            label="⚡ Moteur",
                            choices=["RunPod (GPU)", "Local (CPU)"],
                            value="RunPod (GPU)",
                        )
                    with gr.Row():
                        btn_transcribe = gr.Button("1️⃣ Transcrire (Whisper)", variant="secondary")
                        btn_scenarios  = gr.Button("2️⃣ Scénarios (Gemini texte)", variant="secondary")

                        resume_cb = gr.Checkbox(
                            label="♻️ Reprendre depuis checkpoint", value=True
                        )

                    with gr.Column(scale=2):
                        # Sélection modèle
                        model_choices = [(m.label, mid) for mid, m in MODELS.items()]
                        model_select = gr.Dropdown(
                            label="🤖 Modèle WAN",
                            choices=model_choices,
                            value=DEFAULT_MODEL_ID,
                        )
                        model_info = gr.Markdown(elem_classes=["model-card"])

                        with gr.Row():
                            num_frames = gr.Slider(
                                label="🎞️ Frames par scène",
                                minimum=17, maximum=129, step=16, value=97,
                                info="WAN: multiples de 4+1 (49=~2s, 81=~3.4s, 97=~4s, 129=~5.4s)",
                            )
                        with gr.Row():
                            width  = gr.Number(label="Largeur",  value=1280, precision=0)
                            height = gr.Number(label="Hauteur",  value=720,  precision=0)
                            steps  = gr.Slider(label="Steps",    minimum=10, maximum=50, step=1, value=20)
                            cfg    = gr.Slider(label="CFG",      minimum=1, maximum=15, step=0.5, value=5.0)

                # Mise à jour params quand modèle change
                model_select.change(
                    update_model_params,
                    inputs=[model_select],
                    outputs=[num_frames, width, height, steps, cfg, model_info],
                )

                with gr.Row():
                    btn_generate   = gr.Button("🎥 Générer clips", variant="primary")
                    btn_stop       = gr.Button("⏹ Stop", variant="stop")
                    btn_assemble   = gr.Button("🎞️ Assembler", variant="secondary")
                    btn_full       = gr.Button("⚡ Pipeline complet", variant="primary", size="lg")

                with gr.Row():
                    status_box = gr.Textbox(label="Statut", lines=3, interactive=False)

                with gr.Row():
                    with gr.Column():
                        segments_preview = gr.Textbox(
                            label="📝 Transcription (aperçu)", lines=8, interactive=False
                        )
                    with gr.Column():
                        scenes_table = gr.Dataframe(
                            label="🎬 Scènes planifiées",
                            headers=["ID", "Temps", "Prompt visuel", "Mouvement"],
                            interactive=False,
                        )

                video_output = gr.Video(label="🎬 Vidéo finale", height=400)
                logs_box = gr.Textbox(label="📋 Logs", lines=12, interactive=False)

                # Bouton principal — Gemini Audio
                btn_gemini.click(
                    step_analyze_audio,
                    inputs=[audio_input, force_reanalyze_cb],
                    outputs=[status_box, scenes_table, style_display, style_analysis_display, btn_generate],
                )
                # Fallback Whisper
                btn_transcribe.click(
                    step1_transcribe,
                    inputs=[audio_input, whisper_model, transcribe_mode],
                    outputs=[status_box, segments_preview, btn_scenarios],
                )
                btn_scenarios.click(
                    step2_scenarios,
                    inputs=[audio_input],
                    outputs=[status_box, scenes_table, style_display, style_analysis_display, btn_generate],
                )
                btn_generate.click(
                    step3_generate,
                    inputs=[audio_input, init_image_input, model_select, num_frames, width, height,
                            steps, cfg, resume_cb],
                    outputs=[status_box, video_output, btn_assemble],
                )
                btn_stop.click(
                    stop_generation,
                    outputs=[status_box],
                )
                btn_assemble.click(
                    step4_assemble,
                    inputs=[audio_input],
                    outputs=[status_box, video_output],
                )
                btn_full.click(
                    run_full_pipeline,
                    inputs=[audio_input, init_image_input, model_select, num_frames,
                            width, height, steps, cfg, resume_cb],
                    outputs=[status_box, video_output, style_display, style_analysis_display, logs_box, scenes_table],
                )

            # ── TAB 2: Scènes ────────────────────────────────────────────────
            with gr.TabItem("🎬 Éditeur de scènes"):
                gr.Markdown("Édite les scènes manuellement avant la génération.")

                with gr.Row():
                    btn_load_scenes = gr.Button("📂 Charger scènes courantes")
                    btn_save_scenes = gr.Button("💾 Sauvegarder", variant="primary")

                scenes_editor = gr.Dataframe(
                    label="Scènes",
                    headers=["scene_id", "start_time", "end_time",
                             "visual_prompt", "motion_prompt", "consistency_tags"],
                    datatype=["number", "number", "number", "str", "str", "str"],
                    interactive=True,
                    wrap=True,
                )
                edit_status = gr.Textbox(label="Statut", interactive=False)

                def load_scenes_to_editor():
                    if _state["scenes"] is None:
                        return gr.update(), "❌ Aucune scène en mémoire"
                    rows = [
                        [s.get("scene_id"), s.get("start_time"), s.get("end_time"),
                         s.get("visual_prompt", ""), s.get("motion_prompt", ""),
                         s.get("consistency_tags", "")]
                        for s in _state["scenes"]
                    ]
                    return rows, f"✅ {len(rows)} scènes chargées"

                def save_edited_scenes(data):
                    if data is None:
                        return "❌ Aucune donnée"
                    scenes = []
                    for i, row in enumerate(data.values.tolist()):
                        if len(row) < 5:
                            continue
                        scenes.append({
                            "scene_id": int(row[0]) if row[0] else i + 1,
                            "start_time": float(row[1]) if row[1] else 0,
                            "end_time": float(row[2]) if row[2] else 5,
                            "visual_prompt": str(row[3]),
                            "motion_prompt": str(row[4]),
                            "consistency_tags": str(row[5]) if len(row) > 5 else "",
                        })
                    _state["scenes"] = scenes
                    return f"✅ {len(scenes)} scènes sauvegardées en mémoire"

                btn_load_scenes.click(load_scenes_to_editor,
                                      outputs=[scenes_editor, edit_status])
                btn_save_scenes.click(save_edited_scenes,
                                      inputs=[scenes_editor], outputs=[edit_status])

            # ── TAB 3: Modèles ───────────────────────────────────────────────
            with gr.TabItem("🤖 Modèles"):
                with gr.Row():
                    comfyui_status = gr.Textbox(
                        label="Statut ComfyUI", interactive=False,
                        value="Cliquez sur Vérifier"
                    )
                    btn_check = gr.Button("🔄 Vérifier connexion")

                models_table = gr.Dataframe(
                    label="Modèles disponibles sur RunPod",
                    headers=["Modèle", "Version", "Résolution", "Frames", "Statut"],
                    interactive=False,
                )
                btn_refresh_models = gr.Button("🔄 Actualiser liste")

                gr.Markdown("""
### Guide de sélection

| Modèle | Usage recommandé |
|--------|-----------------|
| **WAN 2.2 Low Noise** | Continuité inter-scènes, mouvements doux, clips narratifs |
| **WAN 2.2 High Noise** | Scènes dynamiques, transitions énergiques, effets visuels |
| **WAN 2.1 I2V 720p** | Alternative 720p, bonne cohérence |
| **WAN 2.1 I2V 480p** | Prototypage rapide, moins de VRAM |

### Frames recommandées (RTX 6000 Ada — 48GB VRAM)
| Frames | Durée @ 24fps | VRAM ~  |
|--------|---------------|---------|
| 49     | 2.0s          | ~12 GB  |
| 81     | 3.4s          | ~18 GB  |
| 97     | 4.0s          | ~22 GB  |
| 113    | 4.7s          | ~26 GB  |
| 129    | 5.4s          | ~30 GB  |
                """)

                btn_check.click(check_comfyui_status, outputs=[comfyui_status])
                btn_refresh_models.click(get_available_models_status, outputs=[models_table])

            # ── TAB 4: Logs ──────────────────────────────────────────────────
            with gr.TabItem("📋 Logs"):
                live_logs = gr.Textbox(
                    label="Logs en temps réel", lines=30, interactive=False,
                    value="En attente..."
                )
                btn_refresh_logs = gr.Button("🔄 Rafraîchir")
                btn_clear_logs   = gr.Button("🗑️ Effacer")

                btn_refresh_logs.click(get_logs, outputs=[live_logs])
                btn_clear_logs.click(
                    lambda: (_state["log"].clear() or "Logs effacés"),
                    outputs=[live_logs],
                )

        # Initialisation au démarrage
        app.load(check_comfyui_status, outputs=[comfyui_status])
        app.load(get_available_models_status, outputs=[models_table])
        app.load(lambda: update_model_params(DEFAULT_MODEL_ID),
                 outputs=[num_frames, width, height, steps, cfg, model_info])

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Clip Generator — UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", default="127.0.0.1", help="Bind address (0.0.0.0 for RunPod)")
    parser.add_argument("--share", action="store_true", help="Crée un lien public Gradio")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print(" AI Video Clip Generator — Interface graphique")
    print(f" http://{args.server_name}:{args.port}")
    print("=" * 55)

    app = build_ui()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    )
