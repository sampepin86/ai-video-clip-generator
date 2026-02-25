"""
Module 3 — Client ComfyUI multi-modèles (WAN 2.1 + WAN 2.2).

Différences d'architecture:
  WAN 2.1: WanImageToVideo → produit (LATENT, COND+, COND-)
  WAN 2.2: Wan22ImageToVideoLatent → produit LATENT seul
            CLIPTextEncode fournit les conditionings directement au KSampler
"""
from __future__ import annotations
import base64, json, time, uuid
from pathlib import Path
from typing import Any
import urllib.request, urllib.parse

from config import (
    COMFYUI_BASE_URL, COMFYUI_WS_URL,
    TEXT_ENCODER_FP8, CLIP_VISION_H,
    WanModel, get_model, DEFAULT_MODEL_ID,
)

# RunPod proxy bloque Python-urllib — on injecte un vrai User-Agent
_UA = "Mozilla/5.0 (compatible; AI-Video-Generator/1.0)"


def _req(url: str, data=None, headers: dict | None = None, method: str | None = None):
    """urllib.request.Request avec User-Agent correct."""
    h = {"User-Agent": _UA}
    if headers:
        h.update(headers)
    kw = {"headers": h}
    if method:
        kw["method"] = method
    return urllib.request.Request(url, data=data, **kw)


def _http_get(url: str, timeout: int = 10) -> bytes:
    with urllib.request.urlopen(_req(url), timeout=timeout) as r:
        return r.read()


def _http_download(url: str, dest: Path, timeout: int = 300) -> None:
    with urllib.request.urlopen(_req(url), timeout=timeout) as r, open(dest, "wb") as f:
        while chunk := r.read(65536):
            f.write(chunk)


# ────────────────────────────────────────────────────────────────────────────
# Constructeurs de workflow
# ────────────────────────────────────────────────────────────────────────────

def _base_nodes(model: WanModel, pos_prompt: str, neg_prompt: str,
                seed: int, p: dict) -> dict[str, Any]:
    """Nodes communs à WAN 2.1 et 2.2."""
    return {
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": model.vae},
        },
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model.filename,
                "weight_dtype": "fp8_e4m3fn" if "fp8" in model.filename else "default",
            },
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER_FP8, "type": "wan"},
        },
        "clip_vision_loader": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION_H},
        },
        "clip_pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": pos_prompt, "clip": ["clip_loader", 0]},
        },
        "clip_neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg_prompt, "clip": ["clip_loader", 0]},
        },
        "vae_decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["ksampler", 0], "vae": ["vae_loader", 0]},
        },
        "create_video": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["vae_decode", 0], "fps": float(p["fps"])},
        },
        "save_video": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": "wan_clip",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def build_wan21_workflow(
    model: WanModel,
    visual_prompt: str,
    motion_prompt: str,
    init_image_filename: str | None,
    params: dict,
    seed: int,
) -> dict[str, Any]:
    """
    Workflow WAN 2.1 :
      WanImageToVideo → (LATENT, COND+, COND-) → KSampler
    """
    neg = "blurry, low quality, distorted, artifacts, watermark, text"
    wf = _base_nodes(model, f"{visual_prompt}, {motion_prompt}", neg, seed, params)

    # ── TeaCache acceleration (~2-3x speedup) ────────────────────────
    wf["teacache"] = {
        "class_type": "WanVideoTeaCacheKJ",
        "inputs": {
            "model":          ["unet_loader", 0],
            "rel_l1_thresh":  model.teacache_threshold,
            "start_percent":  0.1,
            "end_percent":    1.0,
            "cache_device":   "main_device",
            "coefficients":   model.teacache_coefficients,
        },
    }

    if init_image_filename:
        wf["load_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image_filename},
        }
        wf["clip_vision_enc"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["clip_vision_loader", 0],
                "image": ["load_image", 0],
                "crop": "center",
            },
        }
        wf["wan_latent"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive":           ["clip_pos", 0],
                "negative":           ["clip_neg", 0],
                "vae":                ["vae_loader", 0],
                "clip_vision_output": ["clip_vision_enc", 0],
                "start_image":        ["load_image", 0],
                "width":              params["width"],
                "height":             params["height"],
                "length":             params["num_frames"],
                "batch_size":         1,
            },
        }
        wf["ksampler"] = {
            "class_type": "KSampler",
            "inputs": {
                "model":         ["teacache", 0],
                "positive":      ["wan_latent", 0],
                "negative":      ["wan_latent", 1],
                "latent_image":  ["wan_latent", 2],
                "seed": seed, "steps": params["steps"],
                "cfg": params["cfg"],
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        }
    else:
        wf["wan_latent"] = {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive":  ["clip_pos", 0], "negative": ["clip_neg", 0],
                "vae":       ["vae_loader", 0],
                "width":     params["width"], "height": params["height"],
                "length":    params["num_frames"], "batch_size": 1,
            },
        }
        wf["ksampler"] = {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["teacache", 0],
                "positive":     ["wan_latent", 0],
                "negative":     ["wan_latent", 1],
                "latent_image": ["wan_latent", 2],
                "seed": seed, "steps": params["steps"], "cfg": params["cfg"],
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        }
    return wf


def build_wan22_workflow(
    model: WanModel,
    visual_prompt: str,
    motion_prompt: str,
    init_image_filename: str | None,
    params: dict,
    seed: int,
) -> dict[str, Any]:
    """
    Workflow WAN 2.2 officiel — architecture MoE split-denoising.
    Ref: https://docs.comfy.org/tutorials/video/wan/wan2_2

    Principe:
      - Deux modèles chargés : high_noise + low_noise
      - ModelSamplingSD3 (shift) appliqué à chacun
      - WanImageToVideo produit (COND+, COND-, LATENT) — même noeud que 2.1
      - KSamplerAdvanced #1 : high_noise model, steps 0→split_step
      - KSamplerAdvanced #2 : low_noise model, steps split_step→total
      - VAE = wan_2.1_vae (pas wan2.2_vae !)
    """
    neg = "blurry, low quality, distorted, artifacts, watermark, text"
    pos = f"{visual_prompt}, {motion_prompt}"
    split_step = params.get("split_step", model.split_step)
    shift = params.get("shift", model.shift)

    wf: dict[str, Any] = {
        # ── Loaders ──────────────────────────────────────────────────────
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": model.vae},
        },
        "unet_high_noise": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model.high_noise_filename,
                "weight_dtype": "fp8_e4m3fn" if "fp8" in (model.high_noise_filename or "") else "default",
            },
        },
        "unet_low_noise": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model.filename,
                "weight_dtype": "fp8_e4m3fn" if "fp8" in model.filename else "default",
            },
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER_FP8, "type": "wan"},
        },
        # ── ModelSamplingSD3 sur les deux modèles ────────────────────────
        "model_sampling_high": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["unet_high_noise", 0],
                "shift": shift,
            },
        },
        "model_sampling_low": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["unet_low_noise", 0],
                "shift": shift,
            },
        },
        # ── Conditioning ─────────────────────────────────────────────────
        "clip_pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": pos, "clip": ["clip_loader", 0]},
        },
        "clip_neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg, "clip": ["clip_loader", 0]},
        },
    }

    # ── WanImageToVideo (même noeud que 2.1 !) ──────────────────────────
    wan_inputs: dict = {
        "positive":  ["clip_pos", 0],
        "negative":  ["clip_neg", 0],
        "vae":       ["vae_loader", 0],
        "width":     params["width"],
        "height":    params["height"],
        "length":    params["num_frames"],
        "batch_size": 1,
    }
    if init_image_filename:
        wf["load_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image_filename},
        }
        wan_inputs["start_image"] = ["load_image", 0]

    wf["wan_latent"] = {
        "class_type": "WanImageToVideo",
        "inputs": wan_inputs,
    }

    # ── TeaCache acceleration (~2-3x speedup) ────────────────────────
    wf["teacache_high"] = {
        "class_type": "WanVideoTeaCacheKJ",
        "inputs": {
            "model":          ["model_sampling_high", 0],
            "rel_l1_thresh":  model.teacache_threshold,
            "start_percent":  0.1,
            "end_percent":    1.0,
            "cache_device":   "main_device",
            "coefficients":   model.teacache_coefficients,
        },
    }
    wf["teacache_low"] = {
        "class_type": "WanVideoTeaCacheKJ",
        "inputs": {
            "model":          ["model_sampling_low", 0],
            "rel_l1_thresh":  model.teacache_threshold,
            "start_percent":  0.1,
            "end_percent":    1.0,
            "cache_device":   "main_device",
            "coefficients":   model.teacache_coefficients,
        },
    }

    # ── Pass 1 : high_noise model (étapes 0 → split_step) ───────────────
    wf["ksampler_high"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model":        ["teacache_high", 0],
            "positive":     ["wan_latent", 0],
            "negative":     ["wan_latent", 1],
            "latent_image": ["wan_latent", 2],
            "add_noise":    "enable",
            "noise_seed":   seed,
            "steps":        params["steps"],
            "cfg":          params["cfg"],
            "sampler_name": "euler",
            "scheduler":    "simple",
            "start_at_step": 0,
            "end_at_step":   split_step,
            "return_with_leftover_noise": "enable",
        },
    }

    # ── Pass 2 : low_noise model (étapes split_step → fin) ──────────────
    wf["ksampler_low"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model":        ["teacache_low", 0],
            "positive":     ["wan_latent", 0],
            "negative":     ["wan_latent", 1],
            "latent_image": ["ksampler_high", 0],
            "add_noise":    "disable",
            "noise_seed":   seed,
            "steps":        params["steps"],
            "cfg":          params["cfg"],
            "sampler_name": "euler",
            "scheduler":    "simple",
            "start_at_step": split_step,
            "end_at_step":   params["steps"],
            "return_with_leftover_noise": "disable",
        },
    }

    # ── Decode + Video ───────────────────────────────────────────────────
    wf["vae_decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["ksampler_low", 0], "vae": ["vae_loader", 0]},
    }
    wf["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["vae_decode", 0], "fps": float(params["fps"])},
    }
    wf["save_video"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": "wan_clip",
            "format": "mp4",
            "codec": "h264",
        },
    }
    return wf


def build_rapid_aio_workflow(
    model: WanModel,
    visual_prompt: str,
    motion_prompt: str,
    init_image_filename: str | None,
    params: dict,
    seed: int,
) -> dict[str, Any]:
    """
    Workflow WAN 2.2 Rapid AllInOne (Phr00t).

    Architecture simplifiée:
      - CheckpointLoaderSimple (un seul fichier: UNet + CLIP + VAE intégrés)
      - CLIPTextEncode pour pos/neg
      - WanImageToVideo pour I2V (ou EmptyLatentVideo pour T2V)
      - KSampler avec dpmpp_sde / beta / 4 steps / cfg 1.0
      - Pas besoin de TeaCache (déjà rapide en 4 steps)
    """
    neg = "blurry, low quality, distorted, artifacts, watermark, text"
    pos = f"{visual_prompt}, {motion_prompt}"

    wf: dict[str, Any] = {
        # ── Loader unique (checkpoint = UNet + CLIP + VAE) ───────────────
        "ckpt_loader": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model.filename},
        },
        # ── Conditioning ─────────────────────────────────────────────────
        "clip_pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": pos, "clip": ["ckpt_loader", 1]},
        },
        "clip_neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg, "clip": ["ckpt_loader", 1]},
        },
    }

    # ── I2V ou T2V ───────────────────────────────────────────────────────
    wan_inputs: dict = {
        "positive":   ["clip_pos", 0],
        "negative":   ["clip_neg", 0],
        "vae":        ["ckpt_loader", 2],
        "width":      params["width"],
        "height":     params["height"],
        "length":     params["num_frames"],
        "batch_size": 1,
    }

    if init_image_filename:
        wf["load_image"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image_filename},
        }
        wf["clip_vision_loader"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": CLIP_VISION_H},
        }
        wf["clip_vision_enc"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["clip_vision_loader", 0],
                "image": ["load_image", 0],
                "crop": "center",
            },
        }
        wan_inputs["start_image"] = ["load_image", 0]
        wan_inputs["clip_vision_output"] = ["clip_vision_enc", 0]

    wf["wan_latent"] = {
        "class_type": "WanImageToVideo",
        "inputs": wan_inputs,
    }

    # ── KSampler: dpmpp_sde / beta / 4 steps / cfg 1.0 ──────────────────
    wf["ksampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model":        ["ckpt_loader", 0],
            "positive":     ["wan_latent", 0],
            "negative":     ["wan_latent", 1],
            "latent_image": ["wan_latent", 2],
            "seed":          seed,
            "steps":         params["steps"],
            "cfg":           params["cfg"],
            "sampler_name":  "dpmpp_sde",
            "scheduler":     "beta",
            "denoise":       1.0,
        },
    }

    # ── Decode + Video ───────────────────────────────────────────────────
    wf["vae_decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["ksampler", 0], "vae": ["ckpt_loader", 2]},
    }
    wf["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["vae_decode", 0], "fps": float(params["fps"])},
    }
    wf["save_video"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": "wan_rapid",
            "format": "mp4",
            "codec": "h264",
        },
    }
    return wf


def build_workflow(
    model: WanModel,
    visual_prompt: str,
    motion_prompt: str,
    init_image_filename: str | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Point d'entrée unique — dispatche selon la version WAN."""
    p = {
        "width":      model.default_width,
        "height":     model.default_height,
        "num_frames": model.default_frames,
        "fps":        24,
        "steps":      model.default_steps,
        "cfg":        model.default_cfg,
        "split_step": model.split_step,
        "shift":      model.shift,
        **(params or {}),
    }
    seed = p.get("seed", -1)
    if seed < 0:
        seed = int(time.time() * 1000) % 2**32

    if model.version == "rapid":
        return build_rapid_aio_workflow(model, visual_prompt, motion_prompt,
                                        init_image_filename, p, seed)
    elif model.version == "2.2":
        return build_wan22_workflow(model, visual_prompt, motion_prompt,
                                    init_image_filename, p, seed)
    else:
        return build_wan21_workflow(model, visual_prompt, motion_prompt,
                                    init_image_filename, p, seed)


# ────────────────────────────────────────────────────────────────────────────
# Client ComfyUI
# ────────────────────────────────────────────────────────────────────────────

class ComfyUIClient:
    def __init__(self, base_url: str = COMFYUI_BASE_URL, ws_url: str = COMFYUI_WS_URL):
        self.base_url = base_url.rstrip("/")
        self.ws_url   = ws_url
        self.client_id = str(uuid.uuid4())

    def upload_image(self, image_data: bytes, filename: str = "init.png") -> str:
        boundary = "CB" + uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="image"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'
        ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()
        req = _req(
            f"{self.base_url}/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()).get("name", filename)

    def upload_image_b64(self, b64: str, filename: str = "init.png") -> str:
        return self.upload_image(base64.b64decode(b64), filename)

    def queue_prompt(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = _req(
            f"{self.base_url}/prompt", data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(body)
                details = err_json.get("error", {})
                node_errors = err_json.get("node_errors", {})
                msg = f"ComfyUI 400: {details}"
                if node_errors:
                    for nid, nerr in node_errors.items():
                        msg += f"\n  node {nid}: {nerr}"
            except Exception:
                msg = f"ComfyUI HTTP {e.code}: {body[:500]}"
            # Dump le workflow pour debug
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            json.dump(workflow, tmp, indent=2)
            tmp.close()
            print(f"  [DEBUG] Workflow dumped → {tmp.name}")
            raise RuntimeError(msg) from e
        if "error" in data:
            raise RuntimeError(f"ComfyUI: {data['error']}")
        pid = data["prompt_id"]
        print(f"  [ComfyUI] queued → {pid}")
        return pid

    def wait_for_completion(
        self, prompt_id: str, timeout: int = 900,
        progress_cb=None,
    ) -> None:
        try:
            import websocket as wsl
        except ImportError:
            self._poll(prompt_id, timeout)
            return

        ws = wsl.WebSocket()
        try:
            ws.connect(f"{self.ws_url}?clientId={self.client_id}", timeout=30)
        except Exception as e:
            print(f"  [ComfyUI] WebSocket connect failed ({e}), falling back to polling")
            self._poll(prompt_id, timeout)
            return

        t0, last = time.time(), 0
        try:
            while True:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"Timeout {timeout}s")
                try:
                    ws.settimeout(30)
                    raw = ws.recv()
                except Exception as e:
                    # WebSocket lost — fall back to polling for remaining time
                    print(f"\n  [ComfyUI] WebSocket recv error ({e}), falling back to polling")
                    ws.close()
                    remaining = max(int(timeout - (time.time() - t0)), 60)
                    self._poll(prompt_id, remaining)
                    return
                if not raw:
                    continue
                msg = json.loads(raw) if isinstance(raw, str) else {}
                t = msg.get("type", "")
                if t == "progress":
                    d = msg["data"]
                    if d.get("prompt_id") and d["prompt_id"] != prompt_id:
                        continue
                    pct = int(d.get("value", 0) / max(d.get("max", 1), 1) * 100)
                    if pct != last:
                        print(f"  [ComfyUI] {pct}%", end="\r", flush=True)
                        if progress_cb:
                            progress_cb(pct)
                        last = pct
                elif t == "executing":
                    d = msg.get("data", {})
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        print(f"\n  [ComfyUI] ✓ {time.time()-t0:.0f}s")
                        return
                elif t == "execution_error":
                    d = msg.get("data", {})
                    if d.get("prompt_id") == prompt_id:
                        err_msg = d.get("exception_message", "unknown error")
                        node_type = d.get("node_type", "?")
                        raise RuntimeError(
                            f"ComfyUI execution error in {node_type}: {err_msg}"
                        )
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _poll(self, prompt_id: str, timeout: int) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(5)
            try:
                h = self.get_history(prompt_id)
            except Exception:
                continue
            entry = h.get(prompt_id, {})
            status = entry.get("status", {})
            status_str = status.get("status_str", "")
            if status_str == "error":
                msgs = status.get("messages", [])
                err_msg = "unknown"
                for m in msgs:
                    if isinstance(m, list) and m[0] == "execution_error":
                        err_data = m[1] if len(m) > 1 else {}
                        err_msg = err_data.get("exception_message", str(err_data))
                        break
                raise RuntimeError(f"ComfyUI execution error: {err_msg}")
            if entry.get("outputs") and any(bool(v) for v in entry["outputs"].values()):
                return
        raise TimeoutError(f"Polling timeout {timeout}s")

    def get_history(self, prompt_id: str) -> dict:
        return json.loads(_http_get(f"{self.base_url}/history/{prompt_id}", timeout=15))

    def download_output_video(self, prompt_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = self.get_history(prompt_id).get(prompt_id, {}).get("outputs", {})
        for node_out in outputs.values():
            for key in ("videos", "gifs", "images"):
                for item in node_out.get(key, []):
                    fname = item.get("filename", "")
                    if not fname:
                        continue
                    params = urllib.parse.urlencode({
                        "filename": fname,
                        "subfolder": item.get("subfolder", ""),
                        "type": "output",
                    })
                    local = output_dir / fname
                    _http_download(f"{self.base_url}/view?{params}", local)
                    print(f"  [ComfyUI] ↓ {local.name} ({local.stat().st_size//1024}KB)")
                    return local
        raise FileNotFoundError(f"Aucun output pour {prompt_id}")

    def get_last_frame_b64(self, prompt_id: str) -> str | None:
        import subprocess, tempfile
        outputs = self.get_history(prompt_id).get(prompt_id, {}).get("outputs", {})
        for node_out in outputs.values():
            for key in ("videos", "gifs"):
                items = node_out.get(key, [])
                if not items:
                    continue
                item = items[0]
                fname = item.get("filename", "")
                if not fname:
                    continue
                params = urllib.parse.urlencode({
                    "filename": fname, "subfolder": item.get("subfolder", ""),
                    "type": "output",
                })
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
                    tmp_v = t.name
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                    tmp_i = t.name
                try:
                    _http_download(f"{self.base_url}/view?{params}", Path(tmp_v))
                    subprocess.run(
                        ["ffmpeg", "-sseof", "-0.1", "-i", tmp_v, "-vframes", "1", "-y", tmp_i],
                        capture_output=True, check=True,
                    )
                    with open(tmp_i, "rb") as f:
                        return base64.b64encode(f.read()).decode()
                except Exception as e:
                    print(f"  [WARN] last frame: {e}")
                    return None
                finally:
                    Path(tmp_v).unlink(missing_ok=True)
                    Path(tmp_i).unlink(missing_ok=True)
        return None

    def list_available_models(self) -> dict[str, list[str]]:
        """Retourne les modèles disponibles dans ComfyUI par catégorie."""
        result = {}
        for category in ("diffusion_models", "checkpoints", "vae", "text_encoders", "clip_vision"):
            try:
                result[category] = json.loads(_http_get(f"{self.base_url}/models/{category}", timeout=10))
            except Exception:
                result[category] = []
        return result

    def generate_scene(
        self,
        visual_prompt: str,
        motion_prompt: str,
        output_path: Path,
        model: WanModel,
        init_image_b64: str | None = None,
        params: dict | None = None,
        progress_cb=None,
    ) -> tuple[Path, str | None]:
        """Génère un clip. Retourne (chemin_video, dernière_frame_b64)."""
        init_filename = None
        if init_image_b64:
            init_filename = self.upload_image_b64(
                init_image_b64, f"init_{output_path.stem}.png"
            )

        workflow = build_workflow(
            model=model,
            visual_prompt=visual_prompt,
            motion_prompt=motion_prompt,
            init_image_filename=init_filename,
            params=params,
        )

        prompt_id = self.queue_prompt(workflow)
        self.wait_for_completion(prompt_id, progress_cb=progress_cb)

        downloaded = self.download_output_video(prompt_id, output_path.parent)
        if downloaded != output_path:
            downloaded.rename(output_path)

        # Extraire la dernière frame depuis le fichier LOCAL (plus fiable que re-download)
        last_frame = self._extract_last_frame(output_path)
        return output_path, last_frame

    @staticmethod
    def _extract_last_frame(video_path: Path) -> str | None:
        """Extrait la dernière frame d'un fichier vidéo local → base64 PNG."""
        import subprocess, tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                tmp_img = t.name
            subprocess.run(
                ["ffmpeg", "-sseof", "-0.1", "-i", str(video_path),
                 "-vframes", "1", "-y", tmp_img],
                capture_output=True, check=True,
            )
            with open(tmp_img, "rb") as f:
                data = f.read()
            if len(data) < 100:
                print(f"  [WARN] Frame extraite trop petite ({len(data)} bytes)")
                return None
            return base64.b64encode(data).decode()
        except FileNotFoundError:
            print("  [WARN] ffmpeg non installé — impossible d'extraire la dernière frame")
            return None
        except Exception as e:
            print(f"  [WARN] Extraction last frame: {e}")
            return None
        finally:
            Path(tmp_img).unlink(missing_ok=True) if 'tmp_img' in dir() else None


if __name__ == "__main__":
    c = ComfyUIClient()
    s = json.loads(_http_get(f"{c.base_url}/system_stats", timeout=10))
    print(f"ComfyUI OK | RAM {s['system']['ram_total']//1024**3}GB")
    avail = c.list_available_models()
    print("Diffusion models:", avail.get("diffusion_models", []))
