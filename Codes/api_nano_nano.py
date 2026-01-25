# -*- coding: utf-8 -*-
"""
🌐 API Client Module - WaveSpeed Integration
✅ No ImgBB: upload local images directly to WaveSpeed Media Upload -> get download_url
"""

import os
import mimetypes
import requests

from config import (
    WAVESPEED_API_KEY,
    NANO_BANANA_API_URL,
    NANO_BANANA_RESOLUTION,
    WAVESPEED_OUTPUT_FORMAT,
    WAVESPEED_SYNC_MODE,
    WAVESPEED_TIMEOUT,
)

# =========================
# WaveSpeed Media Upload
# =========================

WAVESPEED_MEDIA_UPLOAD_URL = "https://api.wavespeed.ai/api/v3/media/upload/binary"

def upload_to_wavespeed_media(image_path: str) -> str | None:
    """
    Upload local file to WaveSpeed Media Upload, returns download_url.
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
# Main: perform_head_swap
# =========================

def perform_head_swap(target_image_path: str,
                      face_image_path: str,
                      output_filename: str,
                      face_url_cached: str | None = None):
    """
    تنفيذ Head Swap باستخدام WaveSpeed API (Nano Banana / Edit Ultra).

    Args:
        target_image_path: مسار الصورة الأساسية (المشهد)
        face_image_path: مسار صورة الوجه (الشخصية)
        output_filename: مسار حفظ النتيجة
        face_url_cached: (اختياري) URL لصورة الوجه مرفوعة مسبقاً على WaveSpeed (download_url)

    Returns:
        str: مسار الملف المحفوظ أو None في حالة الفشل
    """
    try:
        # الخطوة 0: حساب aspect ratio من أبعاد الصورة الأصلية
        from utils import get_image_dimensions, calculate_closest_aspect_ratio

        dims = get_image_dimensions(target_image_path)
        if dims:
            width, height = dims
            aspect_ratio = calculate_closest_aspect_ratio(width, height)
            print(f"   📐 Original: {width}x{height} → Aspect Ratio: {aspect_ratio}")
        else:
            aspect_ratio = "16:9"
            print(f"   ⚠️  Could not get dimensions, using default: {aspect_ratio}")

        # الخطوة 1: رفع الصورة الأساسية على WaveSpeed
        print("   ☁️  Uploading target image to WaveSpeed...")
        target_url = upload_to_wavespeed_media(target_image_path)
        if not target_url:
            print("   ❌ Failed to upload target image")
            return None

        # الخطوة 2: استخدام face_url المحفوظ أو رفع الوجه على WaveSpeed
        if face_url_cached:
            face_url = face_url_cached
        else:
            print("   ☁️  Uploading face image to WaveSpeed...")
            face_url = upload_to_wavespeed_media(face_image_path)
            if not face_url:
                print("   ❌ Failed to upload face image")
                return None

        # الخطوة 3: إرسال الطلب لـ WaveSpeed API
       
        print("   🔄 Processing with WaveSpeed API...")
#       "prompt": "Replace ONLY the head/face in the target image with the reference head. Keep body, pose, clothing, background, and lighting unchanged. Render the reference head with high realism regarding skin texture, facial details, and hair structure. Adapt the shading and lighting to integrate seamlessly with the target image's general art style, creating a realistic yet stylized look that matches the body. Seamless neck blending.",

        payload = {
            "images": [target_url, face_url],  # [base_image_url, reference_face_url]
           "prompt": "Replace ONLY the facial features (skin, eyes, nose, mouth) in the target image with features from the reference image. Strictly keep the original hair, hairline, body, pose, clothing, background, and lighting unchanged. Render the new facial area with high realism regarding skin texture and details. Adapt the shading and lighting of the new face to integrate seamlessly with the surrounding original hair and the target image's general art style, creating a realistic yet stylized look that matches the body. Seamless blending along the jawline and hairline.",
            "aspect_ratio": aspect_ratio,
          "aspect_ratio": aspect_ratio,
            "resolution": NANO_BANANA_RESOLUTION,  # لازم تكون 1k/2k/4k حسب الـ endpoint
            "output_format": WAVESPEED_OUTPUT_FORMAT,
            "enable_sync_mode": WAVESPEED_SYNC_MODE
        }

        headers = {
            "Authorization": f"Bearer {WAVESPEED_API_KEY}",
            "Content-Type": "application/json"
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

        outputs = (result.get("data") or {}).get("outputs") or []
        if not outputs:
            print("   ❌ No output in API response")
            err = (result.get("data") or {}).get("error") or result.get("error")
            if err:
                print(f"   📄 Error: {err}")
            return None

        result_url = outputs[0]

        # الخطوة 4: تحميل الصورة النهائية
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
