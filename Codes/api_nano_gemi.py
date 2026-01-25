# -*- coding: utf-8 -*-
"""
🌐 Gemini Nano Banana Pro (gemini-3-pro-image-preview) - Direct
✅ No ImgBB
✅ Direct local images
✅ Keeps same signature: perform_head_swap(target_image_path, face_image_path, output_filename, face_url_cached=None)
"""

import os
from PIL import Image

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,        # "gemini-3-pro-image-preview"
    GEMINI_TIMEOUT,
    GEMINI_IMAGE_SIZE,   # "1K" | "2K" | "4K"  (uppercase K)
)


def _pick_aspect_ratio(width: int, height: int) -> str:
    """
    Maps to one of the common ratios used by Nano Banana models.
    (You can expand if you want.)
    """
    ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    # compare by float
    target = width / max(1, height)
    best = "1:1"
    best_diff = 10**9
    for r in ratios:
        a, b = r.split(":")
        val = float(a) / float(b)
        d = abs(val - target)
        if d < best_diff:
            best_diff = d
            best = r
    return best


def perform_head_swap(target_image_path: str,
                      face_image_path: str,
                      output_filename: str,
                      face_url_cached=None):
    """
    Head/Face swap using Gemini 3 Pro Image Preview (Nano Banana Pro).

    Args:
        target_image_path: path to scene image (target)
        face_image_path: path to face reference
        output_filename: path to save result
        face_url_cached: ignored (kept for compatibility)

    Returns:
        output_filename or None
    """
    try:
        if not os.path.exists(target_image_path):
            print(f"   ❌ Target not found: {target_image_path}")
            return None
        if not os.path.exists(face_image_path):
            print(f"   ❌ Face not found: {face_image_path}")
            return None

        # Read target dims -> choose aspect ratio
        with Image.open(target_image_path) as im:
            w, h = im.size
        aspect_ratio = _pick_aspect_ratio(w, h)
        print(f"   📐 Original: {w}x{h} → Aspect Ratio: {aspect_ratio}")

        # Prompt (نفس “النتايج” اللي عندك: تغيير وش بس + دمج نضيف)
        prompt = (
            "You are performing a realistic face swap edit.\n"
            "TARGET is the scene image. REFERENCE is the face image.\n"
            "Task:\n"
            "1) Replace ONLY the facial identity in TARGET with the person from REFERENCE.\n"
            "2) Keep TARGET hair, hairline, head shape, body, pose, clothing, background, camera angle, and composition unchanged.\n"
            "3) Match TARGET expression and gaze direction so the person looks naturally engaged with the scene.\n"
            "4) Seamless blending at jawline and hairline, correct neck alignment, matching skin tone, shadows, and sharpness.\n"
            "Output: ONE edited image only."
        )

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Load images as PIL
        target_img = Image.open(target_image_path)
        face_img = Image.open(face_image_path)

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],  # يرجع صورة بس
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=GEMINI_IMAGE_SIZE,  # "1K"/"2K"/"4K" (K كبيرة) :contentReference[oaicite:2]{index=2}
            ),
            # (اختياري) لو عايز تمنع أي شرح نصي:
            # response_modalities=["IMAGE"]
        )

        print("   🔄 Processing with Gemini Nano Banana Pro...")
        resp = client.models.generate_content(
            model=GEMINI_MODEL,                 # gemini-3-pro-image-preview :contentReference[oaicite:3]{index=3}
            contents=[prompt, target_img, face_img],
            config=config,
            
        )

        # Extract first image part
        saved = False
        for part in getattr(resp, "parts", []) or []:
            if getattr(part, "inline_data", None) is not None:
                img = part.as_image()
                os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                img.save(output_filename)
                saved = True
                break

        if not saved:
            # sometimes SDK uses resp.candidates[0].content.parts
            try:
                cand = resp.candidates[0]
                for part in cand.content.parts:
                    if getattr(part, "inline_data", None) is not None:
                        img = part.as_image()
                        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
                        img.save(output_filename)
                        saved = True
                        break
            except Exception:
                pass

        if not saved:
            print("   ❌ No image returned from Gemini.")
            # لو رجّع نص
            try:
                print("   📄 Text:", getattr(resp, "text", "")[:300])
            except Exception:
                pass
            return None

        print(f"   ✅ Saved: {os.path.basename(output_filename)}")
        return output_filename

    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None
