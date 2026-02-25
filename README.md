# AI Video Clip Generator

Generate AI video clips from audio/music using WAN 2.1/2.2 models via ComfyUI.
Upload a song, get a transcription, let Gemini write visual scenarios for each
segment, then generate and assemble video clips through a Gradio web interface.

## Architecture

```
+-----------------------------------------------------+
|                   Gradio UI (ui.py)                 |
|                  localhost:7860                      |
+----------+------------------------------+-----------+
           |                              |
     +-----v-----+                 +------v------+
     |  Pipeline  |                |   ComfyUI   |
     |  Modules   |--------------->|  localhost:  |
     |  1 - 5     |  WebSocket +   |    8188      |
     +------------+  REST API      +-------------+
           |
     +-----v-------------------------------------+
     |  Module 1: Whisper (transcription)        |
     |  Module 2: Gemini (scenarios)             |
     |  Module 3: ComfyUI client (WS/REST)      |
     |  Module 4: Video generation               |
     |  Module 5: FFmpeg assembly                |
     +-------------------------------------------+
```

## Quick Start - RunPod (Recommended)

### 1. Create a RunPod Template

- **Docker Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Docker Start Command**: `bash /workspace/ai-video-clip-generator/runpod_start.sh`
- **Expose HTTP Port**: `7860`
- **Volume Mount**: `/workspace` (persistent)
- **GPU**: RTX 4090 / RTX 6000 Ada / A6000 (24+ GB VRAM recommended)

### 2. Environment Variables (set in RunPod template)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for scenario generation |
| `RUNPOD_POD_ID` | Auto | Automatically set by RunPod |

### 3. Upload project files

Upload the project to `/workspace/ai-video-clip-generator/` on the pod
(via SSH/SCP or mount from GitHub).

### 4. Access the UI

Once the pod starts, access the Gradio UI at:
```
https://<POD_ID>-7860.proxy.runpod.net
```

## Quick Start - Local (Remote ComfyUI)

If you have ComfyUI running on a remote RunPod pod and want to run the UI locally:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export RUNPOD_POD_ID="your-pod-id"
pip install -r requirements.txt
python ui.py
```

The config auto-detects whether you are on RunPod (uses localhost:8188)
or running locally (uses RunPod proxy URL).

## Models Supported

| Model | Type | Resolution | VRAM | Speed | Quality |
|---|---|---|---|---|---|
| WAN 2.1 T2V 1.3B | Text-to-Video | 832x480 | ~6 GB | Ultra-fast | 4/10 |
| WAN 2.1 I2V 480p fp8 | Image-to-Video | 832x480 | ~18 GB | Fast | 7/10 |
| WAN 2.1 I2V 720p fp8 | Image-to-Video | 1280x720 | ~22 GB | Medium | 9/10 |
| WAN 2.1 I2V 720p fp16 | Image-to-Video | 1280x720 | ~28 GB | Slow | 10/10 |
| WAN 2.2 I2V MoE fp8 | Image-to-Video | 832x480 | ~24 GB | Medium | 9/10 |

## Project Structure

```
ui.py                          # Gradio web interface
main.py                        # CLI orchestrator
runpod_start.sh                # RunPod startup script
setup_wan_comfyui.sh           # Model download script
start_comfyui.sh               # ComfyUI launcher (manual)
requirements.txt               # Python dependencies
pipeline/
  config.py                    # Centralized configuration
  module1_transcribe.py        # Audio -> text segments (Whisper)
  module2_scenarios.py         # Segments -> visual scenarios (Gemini)
  module3_comfyui_client.py    # ComfyUI WebSocket/REST client
  module4_generate.py          # Scenario -> video clips (ComfyUI)
  module5_assemble.py          # Clips -> final video (FFmpeg)
```

## License

MIT
