# -*- coding: utf-8 -*-
"""
🌐 API Client Module - WaveSpeed Integration
✅ Hybrid Prompt:
   1) TXT prompt (if exists and has text)
   2) Else auto-prompt via OpenAI Vision -> saved to TXT
✅ No ImgBB: upload local images directly to WaveSpeed Media Upload -> get download_url
"""

import os
import json
import base64
import mimetypes
import hashlib
import requests

from config import (
    WAVESPEED_API_KEY,
    NANO_BANANA_API_URL,
    NANO_BANANA_RESOLUTION,      # "1k" | "2k" | "4k" for edit, or "4k" | "8k" for edit-ultra
    WAVESPEED_OUTPUT_FORMAT,     # "png" | "jpeg"
    WAVESPEED_SYNC_MODE,         # bool
    WAVESPEED_TIMEOUT,           # seconds
)

# Optional (for auto prompt)
try:
    from config import OPENAI_API_KEY, OPENAI_MODEL
except Exception:
    OPENAI_API_KEY, OPENAI_MODEL = None, None


# =========================
# Paths / Prompt storage
# =========================

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)


# =========================
# Helpers
# =========================

def _safe_basename(path: str) -> str:
    b = os.path.splitext(os.path.basename(path))[0]
    # simple sanitize
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in b)

def _pair_prompt_filename(target_path: str, face_path: str) -> str:
    """
    Unique prompt file per (target, face) pair.
    Uses sha1 of absolute paths to avoid filename explosions for long names.
    """
    key = (os.path.abspath(target_path) + "||" + os.path.abspath(face_path)).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:12]
    return f"prompt_{_safe_basename(target_path)}__{_safe_basename(face_path)}__{h}.txt"

def _read_prompt_if_exists(prompt_path: str) -> str | None:
    if not os.path.exists(prompt_path):
        return None
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        return txt if txt else None
    except Exception:
        return None

def _write_prompt(prompt_path: str, prompt: str) -> None:
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write((prompt or "").strip() + "\n")


# =========================
# WaveSpeed Media Upload (replaces ImgBB)
# =========================

WAVESPEED_MEDIA_UPLOAD_URL = "https://api.wavespeed.ai/api/v3/media/upload/binary"  # docs :contentReference[oaicite:1]{index=1}

def upload_to_wavespeed_media(image_path: str) -> str | None:
    """
    Upload local file to WaveSpeed Media Upload, get download_url.

    Returns:
        download_url (str) or None
    """
    try:
        if not os.path.exists(image_path):
            print(f"   ❌ File not found: {image_path}")
            return None

        headers = {"Authorization": f"Bearer {WAVESPEED_API_KEY}"}

        with open(image_path, "rb") as f:
            filename = os.path.basename(image_path)
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            files = {"file": (filename, f, mime)}
            resp = requests.post(
                WAVESPEED_MEDIA_UPLOAD_URL,
                headers=headers,
                files=files,
                timeout=min(120, WAVESPEED_TIMEOUT),
            )

        if resp.status_code != 200:
            print(f"   ❌ WaveSpeed Media Upload failed: {resp.status_code}")
            if resp.text:
                print(f"   📄 Response: {resp.text[:250]}")
            return None

        data = resp.json()
        url = (data.get("data") or {}).get("download_url")
        if not url:
            print("   ❌ Media Upload response missing download_url")
            return None

        return url

    except requests.exceptions.Timeout:
        print("   ❌ WaveSpeed Media Upload timeout")
        return None
    except Exception as e:
        print(f"   ❌ WaveSpeed Media Upload error: {str(e)}")
        return None


# =========================
# OpenAI auto-prompt (Vision)
# =========================

def _img_to_data_url(path: str) -> str:
    """
    Convert local image to data URL (base64) for OpenAI vision input.
    """
    with open(path, "rb") as f:
        b = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def generate_prompt_with_openai(target_image_path: str, face_image_path: str) -> str | None:
    """
    Uses OpenAI Responses API to analyze images and produce a best-fit prompt.
    Returns prompt text or None.
    """
    if not OPENAI_API_KEY or not OPENAI_MODEL:
        print("   ⚠️  OPENAI_API_KEY / OPENAI_MODEL not set. Skipping auto-prompt.")
        return None

    try:
        # System-ish instruction: output prompt only.
        instruction = (
    "You are generating a single prompt for a head-swap image-edit model.\n"
    "INPUTS: (1) TARGET scene image, (2) FACE_REFERENCE person image.\n\n"

    "GOAL: Put the FACE_REFERENCE person into the TARGET naturally by swapping ONLY the head/face of the main person in TARGET.\n"
    "The result must look like the person belongs to the scene (interactive and coherent), not pasted.\n\n"

    "CRITICAL REQUIREMENTS:\n"
    "- Keep TARGET body, pose, clothing, background, and camera framing unchanged.\n"
    "- Match the TARGET head pose precisely (yaw/pitch/roll), perspective, and face scale.\n"
    "- Match gaze direction so the person appears engaged with the scene (same direction as the TARGET subject).\n"
    "- Match facial expression to the scene mood (subtle adjustment allowed) while preserving identity.\n"
    "- Seamless neck and jawline integration: correct neck thickness, skin tone, and edge blending.\n"
    "- Match lighting direction, shadow softness, and sharpness to TARGET.\n"
    "- Photorealistic; no artifacts; do not change hairstyle except where needed for clean blending.\n"
    "- If there are multiple people, apply to the most prominent person (largest/closest face) only.\n\n"

    "OUTPUT FORMAT:\n"
    "Return ONLY ONE concise prompt (1–3 sentences). No bullet points, no explanations, no quotes."
)

        target_data_url = _img_to_data_url(target_image_path)
        face_data_url = _img_to_data_url(face_image_path)

        payload = {
            "model": OPENAI_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_text", "text": "TARGET image:"},
                        {"type": "input_image", "image_url": target_data_url},
                        {"type": "input_text", "text": "FACE_REFERENCE image:"},
                        {"type": "input_image", "image_url": face_data_url},
                    ],
                }
            ],
            "text": {"format": {"type": "text"}},
        }

        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=min(120, WAVESPEED_TIMEOUT),
        )

        if resp.status_code != 200:
            print(f"   ❌ OpenAI auto-prompt error: {resp.status_code}")
            if resp.text:
                print(f"   📄 Response: {resp.text[:300]}")
            return None

        data = resp.json()

        # Responses API often returns output_text in output[...].content[...]
        # We'll extract best-effort.
        prompt_text = None
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    prompt_text = c.get("text")
                    if prompt_text:
                        break
            if prompt_text:
                break

        if not prompt_text:
            # fallback: some variants may include a "text" top-level summary
            prompt_text = (data.get("text") or "").strip()

        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            print("   ❌ OpenAI auto-prompt returned empty text")
            return None

        return prompt_text

    except requests.exceptions.Timeout:
        print("   ❌ OpenAI auto-prompt timeout")
        return None
    except Exception as e:
        print(f"   ❌ OpenAI auto-prompt exception: {str(e)}")
        return None


def resolve_prompt_hybrid(target_image_path: str, face_image_path: str) -> tuple[str, str]:
    """
    Returns (prompt_text, prompt_txt_path).
    Flow:
      - if prompt txt exists & non-empty -> use it
      - else auto-generate via OpenAI -> save -> use it
      - else fallback default prompt -> save -> use it
    """
    prompt_txt = os.path.join(PROMPTS_DIR, _pair_prompt_filename(target_image_path, face_image_path))

    existing = _read_prompt_if_exists(prompt_txt)
    if existing:
        return existing, prompt_txt

    auto_prompt = generate_prompt_with_openai(target_image_path, face_image_path)
    if auto_prompt:
        _write_prompt(prompt_txt, auto_prompt)
        return auto_prompt, prompt_txt

    # fallback safe default (still saved so user can edit it next run)
    fallback = (
        "Replace the person's head in the first image with the head from the second reference image. "
        "Keep pose, lighting, shadows, camera angle, background, and clothing unchanged. "
        "Make the blend seamless with natural neck alignment and matching skin tone and sharpness."
    )
    _write_prompt(prompt_txt, fallback)
    return fallback, prompt_txt


# =========================
# Main: perform_head_swap
# =========================

def perform_head_swap(target_image_path: str,
                      face_image_path: str,
                      output_filename: str,
                      face_url_cached: str | None = None,
                      target_url_cached: str | None = None):
    """
    Execute Head Swap using WaveSpeed Nano Banana Pro Edit/Ultra API.

    Args:
        target_image_path: local path of the base scene
        face_image_path: local path of the face reference
        output_filename: where to save output
        face_url_cached: optional already-uploaded WaveSpeed media URL for face
        target_url_cached: optional already-uploaded WaveSpeed media URL for target

    Returns:
        output_filename (str) or None
    """
    try:
        # Step 0: aspect ratio from utils (your existing helpers)
        from utils import get_image_dimensions, calculate_closest_aspect_ratio

        dims = get_image_dimensions(target_image_path)
        if dims:
            width, height = dims
            aspect_ratio = calculate_closest_aspect_ratio(width, height)
            print(f"   📐 Original: {width}x{height} → Aspect Ratio: {aspect_ratio}")
        else:
            aspect_ratio = "16:9"
            print(f"   ⚠️  Could not get dimensions, using default: {aspect_ratio}")

        # Step 1: prompt (Hybrid)
        prompt, prompt_path = resolve_prompt_hybrid(target_image_path, face_image_path)
        print(f"   📝 Prompt source: {prompt_path}")

        # Step 2: Upload target & face to WaveSpeed (or reuse cached URLs)
        if target_url_cached:
            target_url = target_url_cached
        else:
            print("   ☁️  Uploading target image to WaveSpeed...")
            target_url = upload_to_wavespeed_media(target_image_path)
            if not target_url:
                print("   ❌ Failed to upload target image")
                return None

        if face_url_cached:
            face_url = face_url_cached
        else:
            print("   ☁️  Uploading face image to WaveSpeed...")
            face_url = upload_to_wavespeed_media(face_image_path)
            if not face_url:
                print("   ❌ Failed to upload face image")
                return None

        # Step 3: Call Nano Banana Pro Edit/Ultra
        print("   🔄 Processing with WaveSpeed API...")

        payload = {
            "images": [target_url, face_url],  # must be URLs :contentReference[oaicite:2]{index=2}
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": NANO_BANANA_RESOLUTION,
            "output_format": WAVESPEED_OUTPUT_FORMAT,
            "enable_sync_mode": WAVESPEED_SYNC_MODE,
            "enable_base64_output": False,
        }

        headers = {
            "Authorization": f"Bearer {WAVESPEED_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            NANO_BANANA_API_URL,
            headers=headers,
            json=payload,
            timeout=WAVESPEED_TIMEOUT
        )

        if response.status_code != 200:
            print(f"   ❌ WaveSpeed API Error: {response.status_code}")
            if response.text:
                print(f"   📄 Response: {response.text[:300]}")
            return None

        result = response.json()

        # Step 4: Get output URL
        outputs = (result.get("data") or {}).get("outputs") or []
        if not outputs:
            print("   ❌ No output in API response")
            # show possible error field
            err = (result.get("data") or {}).get("error") or result.get("error")
            if err:
                print(f"   📄 Error: {err}")
            return None

        result_url = outputs[0]

        # Step 5: Download
        print("   ⬇️  Downloading result...")
        img_response = requests.get(result_url, timeout=WAVESPEED_TIMEOUT)

        if img_response.status_code != 200:
            print(f"   ❌ Failed to download result: {img_response.status_code}")
            return None

        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        with open(output_filename, "wb") as f:
            f.write(img_response.content)

        print(f"   ✅ Saved: {os.path.basename(output_filename)}")
        return output_filename

    except requests.exceptions.Timeout:
        print("   ❌ Request timeout")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None
