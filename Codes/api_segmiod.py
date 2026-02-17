# -*- coding: utf-8 -*-
"""
🌐 Segmind - FaceSwap Only (INFINITE)

✅ Same signature:
   perform_head_swap(target_image_path, face_image_path, output_filename, face_url_cached=None)

Behavior:
- Uses ONLY faceswap-v5 (no prompts, no flux/klein).
- Interactive mode (SEGMIND_INTERACTIVE=1):
    infinite attempts until user accepts.
- Non-interactive mode (SEGMIND_INTERACTIVE=0):
    runs a finite number of attempts (default: 8) and saves last successful.
    You can control this with:
      SEGMIND_MAX_ATTEMPTS=8  (or any int)

Single attempt mode:
- SEGMIND_SINGLE_ATTEMPT=1
- SEGMIND_ATTEMPT_INDEX=1..∞  (any positive integer)
- Generates only that attempt preview and returns its path.

Caching:
- Saves uploaded target/face URLs to json next to output file to avoid re-uploading.

Safe input prevents EOFError.
"""

import os
import time
import base64
import shutil
import json
import requests

from Codes.config import (
    SEGMIND_API_KEY,
    SEGMIND_TIMEOUT,   # seconds, e.g. 120
    SEGMIND_SEED,      # int or None
)

SEGMIND_FACESWAP_V5_URL = "https://api.segmind.com/v1/faceswap-v5"


# ---------------------------
# Helpers (env)
# ---------------------------
def _is_interactive() -> bool:
    return os.getenv("SEGMIND_INTERACTIVE", "1").strip().lower() in ("1", "true", "yes", "y")


def _is_single_attempt() -> bool:
    return os.getenv("SEGMIND_SINGLE_ATTEMPT", "0").strip().lower() in ("1", "true", "yes", "y")


def _get_attempt_index_from_env() -> int:
    v = (os.getenv("SEGMIND_ATTEMPT_INDEX", "") or "").strip()
    if v.isdigit():
        n = int(v)
        return max(1, n)
    return 1


def _no_try_files() -> bool:
    return os.getenv("SEGMIND_NO_TRY_FILES", "0").strip().lower() in ("1", "true", "yes", "y")


def _get_max_attempts_non_interactive(default: int = 8) -> int:
    v = (os.getenv("SEGMIND_MAX_ATTEMPTS", "") or "").strip()
    if v.isdigit():
        return max(1, int(v))
    return default


# ---------------------------
# Cache helpers (URLs)
# ---------------------------
def _cache_path_for(output_filename: str) -> str:
    base, _ = os.path.splitext(output_filename)
    return base + "_segmind_cache.json"


def _load_cache(output_filename: str) -> dict:
    p = _cache_path_for(output_filename)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}


def _save_cache(output_filename: str, cache: dict) -> None:
    p = _cache_path_for(output_filename)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------
# Segmind storage upload
# ---------------------------
def _upload_to_segmind_storage(image_path: str, retries: int = 3, wait_sec: int = 3) -> str | None:
    if not os.path.exists(image_path):
        print(f"   ❌ File not found: {image_path}")
        return None

    os.environ["SEGMIND_API_KEY"] = SEGMIND_API_KEY
    
    # DEBUG: Log API key status
    print(f"   🔍 DEBUG: Segmind API Key present: {bool(SEGMIND_API_KEY)}")
    print(f"   🔍 DEBUG: API Key length: {len(SEGMIND_API_KEY) if SEGMIND_API_KEY else 0}")

    try:
        import segmind  # pip install segmind
        print(f"   🔍 DEBUG: segmind package imported successfully")
    except Exception as e:
        print("   ❌ segmind package not installed. Run: pip install segmind")
        print(f"   📄 Import error: {e}")
        return None

    for attempt in range(1, retries + 1):
        try:
            print(f"   ⬆️  Upload attempt {attempt}/{retries}: {os.path.basename(image_path)}")
            result = segmind.files.upload(image_path)
            print(f"   🔍 DEBUG: Segmind upload result: {result}")
            urls = (result or {}).get("file_urls") or []
            if urls:
                print(f"   🔍 DEBUG: Uploaded URL: {urls[0][:50]}...")
                return urls[0]
            print("   ❌ Segmind upload returned no file_urls")
        except Exception as e:
            print(f"   ❌ Segmind upload error: {e}")
            if attempt < retries:
                time.sleep(wait_sec)
            else:
                return None

    return None


# ---------------------------
# Segmind response save
# ---------------------------
def _save_response_to_file(resp: requests.Response, output_path: str, timeout: int) -> bool:
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
    try:
        return input(prompt)
    except EOFError:
        return ""
    except Exception:
        return ""


def _ask_user_choice(image_path: str, attempt_num: int, label: str) -> int:
    if not _is_interactive():
        return 2

    print("\n" + "=" * 60)
    print(f"🖼️  Preview saved: {image_path}")
    print(f"{label}  |  Attempt {attempt_num}")
    print("Choose:")
    print("  1) ✅ Yes, I like it (accept)")
    print("  2) ❌ No, retry next attempt")
    print("=" * 60)

    while True:
        val = _safe_input("Enter 1 or 2: ").strip()
        if val == "":
            print("   ⚠️ No stdin available (EOF). Defaulting to: 2 (retry next attempt)")
            return 2
        if val in ("1", "2"):
            return int(val)
        print("Invalid input. Please enter 1 or 2.")


# ---------------------------
# API Call (faceswap-v5 only)
# ---------------------------
def _call_faceswap_v5(target_url: str, face_url: str, seed: int, timeout: int) -> requests.Response:
    headers = {"x-api-key": SEGMIND_API_KEY, "Content-Type": "application/json"}
    data = {
        "target_image": target_url,
        "source_image": face_url,
        "seed": int(seed),
        "image_format": "png",
        "quality": 95,
    }
    
    # DEBUG: Log API call details
    print(f"   🔍 DEBUG: Calling Segmind faceswap-v5 API")
    print(f"   🔍 DEBUG: Target URL: {target_url[:50] if target_url else 'None'}...")
    print(f"   🔍 DEBUG: Face URL: {face_url[:50] if face_url else 'None'}...")
    print(f"   🔍 DEBUG: Seed: {seed}, Timeout: {timeout}")
    
    try:
        resp = requests.post(SEGMIND_FACESWAP_V5_URL, headers=headers, json=data, timeout=timeout)
        print(f"   🔍 DEBUG: Response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"   🔍 DEBUG: Response text: {resp.text[:500]}")
        return resp
    except requests.exceptions.Timeout:
        print(f"   ❌ DEBUG: Request timeout after {timeout} seconds")
        raise
    except Exception as e:
        print(f"   ❌ DEBUG: Request exception: {e}")
        raise


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

    Returns:
      - In normal mode: final output_filename path if accepted/saved, else None (only if all failed)
      - In single attempt mode: preview path (NOT final), or None
    """
    print(f"\n=== DEBUG: Starting head swap ===")
    print(f"   DEBUG: target_image_path: {target_image_path}")
    print(f"   DEBUG: face_image_path: {face_image_path}")
    print(f"   DEBUG: output_filename: {output_filename}")
    print(f"   DEBUG: face_url_cached: {face_url_cached}")
    
    try:
        if not os.path.exists(target_image_path):
            print(f"   ❌ Target not found: {target_image_path}")
            return None
        if not os.path.exists(face_image_path):
            print(f"   ❌ Face not found: {face_image_path}")
            return None

        base_name, base_ext = os.path.splitext(output_filename)
        if not base_ext:
            base_ext = ".png"

        seed_base = SEGMIND_SEED if SEGMIND_SEED is not None else 42
        print(f"   DEBUG: seed_base: {seed_base}")

        # ---------------------------
        # Load cache (urls)
        # ---------------------------
        cache = _load_cache(output_filename)

        target_url = cache.get("target_url")
        face_url = cache.get("face_url")

        if not target_url:
            print("   ☁️  Uploading target to Segmind Storage...")
            target_url = _upload_to_segmind_storage(target_image_path)
            if not target_url:
                print("   ❌ Failed to upload target")
                return None
            cache["target_url"] = target_url
            _save_cache(output_filename, cache)

        if face_url_cached:
            face_url = face_url_cached
            cache["face_url"] = face_url
            _save_cache(output_filename, cache)

        if not face_url:
            print("   ☁️  Uploading face to Segmind Storage...")
            face_url = _upload_to_segmind_storage(face_image_path)
            if not face_url:
                print("   ❌ Failed to upload face")
                return None
            cache["face_url"] = face_url
            _save_cache(output_filename, cache)

        # ---------------------------
        # SINGLE ATTEMPT MODE (any N)
        # ---------------------------
        if _is_single_attempt():
            attempt_num = _get_attempt_index_from_env()
            preview_path = output_filename if _no_try_files() else f"{base_name}_try{attempt_num}{base_ext}"
            attempt_seed = seed_base + (attempt_num - 1)

            print(f"\n🚀 Single Attempt {attempt_num} (faceswap-v5)")
            resp = _call_faceswap_v5(target_url, face_url, attempt_seed, SEGMIND_TIMEOUT)

            if resp.status_code != 200:
                print(f"   ❌ Segmind faceswap-v5 error: {resp.status_code}")
                try:
                    print(resp.text[:800])
                except Exception:
                    pass
                return None

            if not _save_response_to_file(resp, preview_path, SEGMIND_TIMEOUT):
                print("   ❌ Could not save preview.")
                return None

            print(f"✅ Single attempt preview saved: {preview_path}")
            return preview_path

        # ---------------------------
        # NORMAL MODE
        # - Interactive: infinite until accept
        # - Non-interactive: finite, save last success
        # ---------------------------
        last_success_preview = None

        if _is_interactive():
            print("\n" + "=" * 70)
            print("♾️  FaceSwap Only mode is ACTIVE (faceswap-v5).")
            print("➡️  Infinite attempts until you accept.")
            print("=" * 70)

            attempt_num = 1
            while True:
                preview_path = f"{base_name}_try{attempt_num}{base_ext}"
                attempt_seed = seed_base + (attempt_num - 1)

                print(f"\n🚀 Attempt {attempt_num} (faceswap-v5) [INFINITE]")
                resp = _call_faceswap_v5(target_url, face_url, attempt_seed, SEGMIND_TIMEOUT)

                if resp.status_code != 200:
                    print(f"   ❌ Segmind faceswap-v5 error: {resp.status_code}")
                    try:
                        print(resp.text[:800])
                    except Exception:
                        pass
                    attempt_num += 1
                    continue

                if not _save_response_to_file(resp, preview_path, SEGMIND_TIMEOUT):
                    print("   ❌ Could not save faceswap preview.")
                    attempt_num += 1
                    continue

                last_success_preview = preview_path
                choice = _ask_user_choice(preview_path, attempt_num, label="faceswap-v5")

                if choice == 1:
                    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                    shutil.copyfile(preview_path, output_filename)
                    print(f"\n🎉 Accepted ✅ Final saved as: {output_filename}")
                    return output_filename

                attempt_num += 1

        # Non-interactive finite loop
        max_attempts = _get_max_attempts_non_interactive(default=8)
        for attempt_num in range(1, max_attempts + 1):
            preview_path = f"{base_name}_try{attempt_num}{base_ext}"
            attempt_seed = seed_base + (attempt_num - 1)

            print(f"\n🚀 Attempt {attempt_num}/{max_attempts} (faceswap-v5) [NON-INTERACTIVE]")
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

        if last_success_preview and os.path.exists(last_success_preview):
            os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
            shutil.copyfile(last_success_preview, output_filename)
            print(f"\n✅ Non-interactive: saved last successful preview as final: {output_filename}")
            return output_filename

        print("\n❌ No successful output produced.")
        return None

    except requests.exceptions.Timeout:
        print("   ❌ Request timeout")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None
