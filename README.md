# AI Video Clip Generator

Transform any MP3 into a synchronized AI-generated music video using WAN video models on ComfyUI/RunPod.

## How it works

1. **Audio Analysis** — Gemini 2.5 Flash analyzes the MP3 directly (lyrics, mood, tempo)
2. **Scene Planning** — AI splits the song into 4-6 second scenes with visual/motion prompts
3. **Video Generation** — WAN I2V models on ComfyUI generate each scene (image-to-video)
4. **Assembly** — MoviePy stitches clips together with the original audio

## Supported Models

| Model | Type | Size | Speed | Quality |
|-------|------|------|-------|---------|
| WAN 2.1 I2V 480p fp8 | Image-to-Video | 16 GB | Fast | ★★★★★★★ |
| WAN 2.1 T2V 1.3B | Text-to-Video | 2.8 GB | Ultra-fast | ★★★★ |
| WAN 2.2 MoE I2V | Image-to-Video | 32 GB | Medium | ★★★★★★★★★ |
| WAN 2.1 I2V 720p | Image-to-Video | 16-28 GB | Slow | ★★★★★★★★★★ |

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/sampepin86/ai-video-clip-generator.git
cd ai-video-clip-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and RUNPOD_POD_ID

# 4. Launch UI
python ui.py
```

## RunPod Deployment

### Prerequisites
- RunPod account with a GPU pod (RTX 4090 / RTX 6000 Ada recommended)
- RunPod template with **ComfyUI pre-installed**
- Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### One-Click Setup

Set these **environment variables** in your RunPod template:
- `GEMINI_API_KEY` — Your Google Gemini API key
- `RUNPOD_POD_ID` — Auto-populated by RunPod

Then run:
```bash
curl -sSL https://raw.githubusercontent.com/sampepin86/ai-video-clip-generator/main/runpod_template.sh | bash
```

Or manually:
```bash
# SSH into your pod
git clone https://github.com/sampepin86/ai-video-clip-generator.git /workspace/ai-video-clip-generator
cd /workspace/ai-video-clip-generator
bash runpod_template.sh
```

### Access
- **Gradio UI**: `https://<pod-id>-7860.proxy.runpod.net`
- **ComfyUI**: `https://<pod-id>-8188.proxy.runpod.net`

## Project Structure

```
├── ui.py                      # Gradio web interface
├── main.py                    # CLI orchestrator
├── pipeline/
│   ├── config.py              # Models catalog & configuration
│   ├── module1_transcribe.py  # Whisper transcription
│   ├── module2_scenarios.py   # Gemini scene generation
│   ├── module3_comfyui_client.py  # ComfyUI API + WAN workflows
│   ├── module4_generate.py    # Sequential I2V generation loop
│   └── module5_assemble.py    # MoviePy video assembly
├── setup_wan_comfyui.sh       # Download WAN models for ComfyUI
├── start_all.sh               # Start ComfyUI + Gradio on RunPod
├── runpod_template.sh         # Full RunPod one-click setup
├── requirements.txt
└── .env.example
```

## CLI Usage

```bash
# Full pipeline
python main.py song.mp3 --style "cinematic, dark atmosphere"

# Planning only (no video generation)
python main.py song.mp3 --scenes-only

# Resume interrupted generation
python main.py song.mp3 --resume
```

## Configuration

All configuration is via environment variables (or `.env` file):

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `RUNPOD_POD_ID` | Yes | RunPod pod ID |
| `COMFYUI_BASE_URL` | No | Auto-derived from pod ID |

## Tech Stack

- **Video Models**: WAN 2.1 / 2.2 (Alibaba) via ComfyUI
- **Audio Analysis**: Google Gemini 2.5 Flash (native audio)
- **Transcription**: OpenAI Whisper (fallback)
- **Acceleration**: TeaCache (KJNodes) ~2-3x speedup
- **UI**: Gradio
- **GPU**: RunPod (RTX 4090 / 6000 Ada / A100)

## License

MIT
