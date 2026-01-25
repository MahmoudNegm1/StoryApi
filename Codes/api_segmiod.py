# -*- coding: utf-8 -*-
"""
🌐 Segmind - Hybrid Pipeline (FLUX PRO back ✅)

✅ Same signature:
   perform_head_swap(target_image_path, face_image_path, output_filename, face_url_cached=None)

Flow (8 attempts total):
  Attempt 1-3: flux-2-pro with prompt_1..prompt_3
  Attempt 4-5: faceswap-v5 (no prompt)
  Attempt 6-8: flux-2-klein-4b with prompt_1..prompt_3

Interactive controlled by ENV:
  SEGMIND_INTERACTIVE=1  -> asks user accept/retry
  SEGMIND_INTERACTIVE=0  -> no input; runs all attempts and saves last successful as final

Notes:
- Upload happens ONCE (target + face) to Segmind Storage and reused across all attempts.
- face_url_cached can be used to skip face upload (pass a Segmind URL).
- Safe input prevents EOFError (common in workers / redirected stdin).
"""

import os
import time
import base64
import shutil
import requests

from Codes.config import (
    SEGMIND_API_KEY,
    SEGMIND_TIMEOUT,   # seconds, e.g. 120
    SEGMIND_SEED,      # int or None
)

SEGMIND_FLUX2_PRO_URL       = "https://api.segmind.com/v1/flux-2-pro"
SEGMIND_FACESWAP_V5_URL     = "https://api.segmind.com/v1/faceswap-v5"
SEGMIND_FLUX2_KLEIN_4B_URL  = "https://api.segmind.com/v1/flux-2-klein-4b"


# ---------------------------
# Helpers
# ---------------------------
def _is_interactive() -> bool:
    return os.getenv("SEGMIND_INTERACTIVE", "1").strip() in ("1", "true", "True", "yes", "YES")


def _pick_size_from_target(target_w: int, target_h: int, max_mp: int = 4_000_000) -> tuple[int, int]:
    """flux-2-pro: keep ratio, cap to ~4MP."""
    if target_w <= 0 or target_h <= 0:
        return (1024, 1024)

    pixels = target_w * target_h
    if pixels <= max_mp:
        return (target_w, target_h)

    scale = (max_mp / pixels) ** 0.5
    w = int(target_w * scale)
    h = int(target_h * scale)

    # even dims
    w = max(256, (w // 2) * 2)
    h = max(256, (h // 2) * 2)
    return (w, h)


def _upload_to_segmind_storage(image_path: str, retries: int = 3, wait_sec: int = 3) -> str | None:
    if not os.path.exists(image_path):
        print(f"   ❌ File not found: {image_path}")
        return None

    os.environ["SEGMIND_API_KEY"] = SEGMIND_API_KEY

    try:
        import segmind  # pip install segmind
    except Exception as e:
        print("   ❌ segmind package not installed. Run: pip install segmind")
        print(f"   📄 Import error: {e}")
        return None

    for attempt in range(1, retries + 1):
        try:
            print(f"   ⬆️  Upload attempt {attempt}/{retries}: {os.path.basename(image_path)}")
            result = segmind.files.upload(image_path)
            urls = (result or {}).get("file_urls") or []
            if urls:
                return urls[0]
            print("   ❌ Segmind upload returned no file_urls")
        except Exception as e:
            print(f"   ❌ Segmind upload error: {e}")
            if attempt < retries:
                time.sleep(wait_sec)
            else:
                return None

    return None


def _save_response_to_file(resp: requests.Response, output_path: str, timeout: int) -> bool:
    """
    Supports:
      - direct image bytes response (Content-Type: image/*)
      - JSON with output url / image url
      - JSON with base64 (best-effort)
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if "image/" in ctype:
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True

    try:
        j = resp.json()
    except Exception:
        print("   ❌ Response is not image and not JSON. Cannot save output.")
        try:
            print(resp.text[:800])
        except Exception:
            pass
        return False

    out_url = (
        j.get("output_url")
        or j.get("image_url")
        or j.get("url")
        or (j.get("data") or {}).get("url")
        or (j.get("data") or {}).get("output_url")
        or (j.get("data") or {}).get("image_url")
    )

    if out_url:
        r2 = requests.get(out_url, timeout=timeout)
        if r2.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(r2.content)
            return True

    b64 = (
        j.get("image_base64")
        or (j.get("data") or {}).get("image_base64")
        or (j.get("data") or {}).get("base64")
        or j.get("base64")
    )
    if b64:
        try:
            raw = base64.b64decode(b64)
            with open(output_path, "wb") as f:
                f.write(raw)
            return True
        except Exception:
            pass

    print("   ❌ JSON response did not include a usable output.")
    print(str(j)[:800])
    return False


def _safe_input(prompt: str) -> str:
    """Prevents crashing with EOFError (common inside multiprocessing workers)."""
    try:
        return input(prompt)
    except EOFError:
        return ""
    except Exception:
        return ""


def _ask_user_choice(image_path: str, attempt_num: int, max_attempts: int, label: str) -> int:
    # Non-interactive mode: always "retry next attempt"
    if not _is_interactive():
        return 2

    print("\n" + "=" * 60)
    print(f"🖼️  Preview saved: {image_path}")
    print(f"{label}  |  Attempt {attempt_num}/{max_attempts}")
    print("Choose:")
    print("  1) ✅ Yes, I like it (accept)")
    print("  2) ❌ No, retry next attempt")
    print("=" * 60)

    while True:
        val = _safe_input("Enter 1 or 2: ").strip()

        # If stdin not available -> keep going
        if val == "":
            print("   ⚠️ No stdin available (EOF). Defaulting to: 2 (retry next attempt)")
            return 2

        if val in ("1", "2"):
            return int(val)

        print("Invalid input. Please enter 1 or 2.")


# ---------------------------
# API Calls
# ---------------------------
def _call_flux2_pro(target_url: str, face_url: str, prompt: str, seed: int, width: int, height: int, timeout: int) -> requests.Response:
    headers = {"x-api-key": SEGMIND_API_KEY, "Content-Type": "application/json"}
    data = {
        "image_urls": [target_url, face_url],
        "prompt": prompt,
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "output_format": "png",
    }
    return requests.post(SEGMIND_FLUX2_PRO_URL, headers=headers, json=data, timeout=timeout)


def _call_faceswap_v5(target_url: str, face_url: str, seed: int, timeout: int) -> requests.Response:
    headers = {"x-api-key": SEGMIND_API_KEY, "Content-Type": "application/json"}
    data = {
        "target_image": target_url,
        "source_image": face_url,
        "seed": int(seed),
        "image_format": "png",
        "quality": 95,
    }
    return requests.post(SEGMIND_FACESWAP_V5_URL, headers=headers, json=data, timeout=timeout)


def _call_flux2_klein_4b(target_url: str, face_url: str, prompt: str, seed: int, timeout: int) -> requests.Response:
    headers = {"x-api-key": SEGMIND_API_KEY, "Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "image_urls": [target_url, face_url],
        "negative_prompt": (
            "floating head, disconnected head, extra head, extra face, wrong identity, "
            "mismatched head size, visible neck seam, distorted features, low quality, blurry"
        ),
        "seed": int(seed),
        "cfg": 5,
        "sampler": "euler",
        "steps": 20,
        "aspect_ratio": "16:9",
        "go_fast": True,
        "image_format": "png",
        "quality": 90,
    }
    return requests.post(SEGMIND_FLUX2_KLEIN_4B_URL, headers=headers, json=data, timeout=timeout)


# ---------------------------
# Main
# ---------------------------
def perform_head_swap(
    target_image_path: str,
    face_image_path: str,
    output_filename: str,
    face_url_cached: str | None = None,
):
    """
    DO NOT change signature.
    In non-interactive mode:
      - runs all attempts
      - if any attempt succeeded -> uses LAST successful preview as final output_filename
    """
    try:
        if not os.path.exists(target_image_path):
            print(f"   ❌ Target not found: {target_image_path}")
            return None
        if not os.path.exists(face_image_path):
            print(f"   ❌ Face not found: {face_image_path}")
            return None

        # Read target size (for flux-2-pro width/height)
        try:
            from PIL import Image
            with Image.open(target_image_path) as im:
                tw, th = im.size
        except Exception:
            tw, th = (1024, 1024)

        req_w, req_h = _pick_size_from_target(tw, th)

        # Upload ONCE
        print("   ☁️  Uploading target to Segmind Storage...")
        target_url = _upload_to_segmind_storage(target_image_path)
        if not target_url:
            print("   ❌ Failed to upload target")
            return None

        if face_url_cached:
            face_url = face_url_cached
        else:
            print("   ☁️  Uploading face to Segmind Storage...")
            face_url = _upload_to_segmind_storage(face_image_path)
            if not face_url:
                print("   ❌ Failed to upload face")
                return None

        # Prompts
        fluxpro_prompts = [
            (
                "High-fidelity face swap edit prioritizing exact identity preservation within a specific scene.\n"
                "Use the provided TARGET image as the foundation for the body, clothing, pose, hair, background, lighting, and overall art style.\n"
                "Use the provided REFERENCE image solely as the source for facial identity.\n"
                "ACTION: Replace the head/face in the TARGET image with the facial identity of the REFERENCE.\n"
                "CRITICAL CONSTRAINT (IDENTITY): Strictly maintain the exact facial structures, eye shape, nose, mouth, and unique likeness of the REFERENCE.\n"
                "INTEGRATION & STYLE: Seamlessly blend the REFERENCE face onto the TARGET body. Match skin tone, lighting, shadows, texture, and the target rendering style. Keep original hair/clothing unchanged."
            ),
            (
                "Replace ONLY the head/face with the REFERENCE identity. Keep body, outfit, pose, background unchanged. "
                "Ensure perfect alignment, painterly consistency, correct head scale, and natural neck transition with no seam."
            ),
            (
                "Strict head swap: preserve the REFERENCE likeness exactly, match the target’s art style and warm lighting, "
                "and blend cleanly at the neck. Do not modify the dress, body, pose, or background."
            ),
        ]

        klein_prompts = [
            (
                "High-fidelity, STRICT HEAD SWAP edit with exact identity preservation and seamless integration.\n"
                "Use the provided TARGET image as the rigid foundation. Keep body, dress, pose, background, and lighting unchanged.\n"
                "Use the provided REFERENCE image as the sole source for the new head (face + hair).\n"
                "ACTION: Replace the head completely with the REFERENCE girl's head and hair, aligned to the original head position.\n"
                "CRITICAL: preserve exact identity; match the painted style; avoid generic cartoon.\n"
                "INTEGRATION: match warm lighting and shadows; seamless neck blend."
            ),
            (
                "Replace only the head, preserving all facial features and hair. "
                "Match the original image's colors, lighting and painted style. "
                "Make it seamless and natural with no neck seam."
            ),
            (
                "Perform a clean head swap: take the girl's head from the reference and integrate it into the target scene. "
                "Preserve her exact identity and hair, match the target lighting and painterly style, "
                "and ensure perfect alignment and blending with the neck."
            ),
        ]

        base_name, base_ext = os.path.splitext(output_filename)
        if not base_ext:
            base_ext = ".png"

        total_attempts = 8
        seed_base = SEGMIND_SEED if SEGMIND_SEED is not None else 42

        print(f"   📐 Target: {tw}x{th} | flux-2-pro request: {req_w}x{req_h}")
        last_success_preview = None

        # Attempts 1-3: FLUX PRO
        for k in range(1, 4):
            attempt_num = k
            preview_path = f"{base_name}_try{attempt_num}{base_ext}"
            prompt = fluxpro_prompts[k - 1].strip() or "High quality. Preserve identity. Match lighting."
            attempt_seed = seed_base + (attempt_num - 1)

            print(f"\n🚀 Attempt {attempt_num}/{total_attempts} (flux-2-pro) | prompt_{k}")
            resp = _call_flux2_pro(target_url, face_url, prompt, attempt_seed, req_w, req_h, SEGMIND_TIMEOUT)

            if resp.status_code != 200:
                print(f"   ❌ Segmind flux-2-pro error: {resp.status_code}")
                try:
                    print(resp.text[:800])
                except Exception:
                    pass
                continue

            if not _save_response_to_file(resp, preview_path, SEGMIND_TIMEOUT):
                print("   ❌ Could not save flux-2-pro preview.")
                continue

            last_success_preview = preview_path

            # interactive accept?
            choice = _ask_user_choice(preview_path, attempt_num, total_attempts, label=f"flux-2-pro (prompt_{k})")
            if choice == 1:
                os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                shutil.copyfile(preview_path, output_filename)
                print(f"\n🎉 Accepted ✅ Final saved as: {output_filename}")
                return output_filename

        # Attempts 4-5: FaceSwap
        for i in range(4, 6):
            attempt_num = i
            preview_path = f"{base_name}_try{attempt_num}{base_ext}"
            attempt_seed = seed_base + (attempt_num - 1)

            print(f"\n🚀 Attempt {attempt_num}/{total_attempts} (faceswap-v5)")
            resp = _call_faceswap_v5(target_url, face_url, attempt_seed, SEGMIND_TIMEOUT)

            if resp.status_code != 200:
                print(f"   ❌ Segmind faceswap-v5 error: {resp.status_code}")
                try:
                    print(resp.text[:800])
                except Exception:
                    pass
                continue

            if not _save_response_to_file(resp, preview_path, SEGMIND_TIMEOUT):
                print("   ❌ Could not save faceswap preview.")
                continue

            last_success_preview = preview_path

            choice = _ask_user_choice(preview_path, attempt_num, total_attempts, label="faceswap-v5")
            if choice == 1:
                os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                shutil.copyfile(preview_path, output_filename)
                print(f"\n🎉 Accepted ✅ Final saved as: {output_filename}")
                return output_filename

        # Attempts 6-8: Klein
        for k in range(1, 4):
            attempt_num = 5 + k
            preview_path = f"{base_name}_try{attempt_num}{base_ext}"
            prompt = klein_prompts[k - 1].strip() or "High quality. Preserve identity. Match lighting."
            attempt_seed = seed_base + (attempt_num - 1)

            print(f"\n🚀 Attempt {attempt_num}/{total_attempts} (flux-2-klein-4b) | prompt_{k}")
            resp = _call_flux2_klein_4b(target_url, face_url, prompt, attempt_seed, SEGMIND_TIMEOUT)

            if resp.status_code != 200:
                print(f"   ❌ Segmind flux-2-klein-4b error: {resp.status_code}")
                try:
                    print(resp.text[:800])
                except Exception:
                    pass
                continue

            if not _save_response_to_file(resp, preview_path, SEGMIND_TIMEOUT):
                print("   ❌ Could not save flux-2-klein-4b preview.")
                continue

            last_success_preview = preview_path

            choice = _ask_user_choice(preview_path, attempt_num, total_attempts, label=f"flux-2-klein-4b (prompt_{k})")
            if choice == 1:
                os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                shutil.copyfile(preview_path, output_filename)
                print(f"\n🎉 Accepted ✅ Final saved as: {output_filename}")
                return output_filename

        # Non-interactive fallback: use last successful
        if last_success_preview and os.path.exists(last_success_preview):
            os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
            shutil.copyfile(last_success_preview, output_filename)
            print(f"\n✅ Non-interactive: saved last successful preview as final: {output_filename}")
            return output_filename

        print("\n❌ Reached max attempts without any successful output.")
        return None

    except requests.exceptions.Timeout:
        print("   ❌ Request timeout")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None
