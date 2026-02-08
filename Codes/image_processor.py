# -*- coding: utf-8 -*-
"""
🖼️ Image Processor Module (CLEAN - Interactive CLI Attempts)

What this module does:
- Batch head-swap for all API images (attempt 1 only) + copy normal images
- Interactive CLI selection to regenerate a specific slide with multiple tries
- Text overlay (sequential or parallel)
- Resolution handling

IMPORTANT:
- Uses api_segmiod.py (note the name) and its fixed signature:
    perform_head_swap(target_image_path, face_image_path, output_filename, face_url_cached=None)

- This module controls attempts using env vars supported by api_segmiod.py:
    SEGMIND_SINGLE_ATTEMPT=1
    SEGMIND_ATTEMPT_INDEX=1..∞ (any positive integer)
    SEGMIND_INTERACTIVE=0  (we do the interactive flow here; api stays non-interactive per attempt)
"""

import os
import cv2
import time
import shutil

from Codes.config import HEAD_SWAP_DELAY
from Codes.api_segmiod import perform_head_swap  # IMPORTANT: api_segmiod.py
from Codes.text_handler import render_image
from Codes.utils import get_image_dimensions


# ---------------------------
# General helpers
# ---------------------------
def resize_image_to_resolution(image, target_width, target_height):
    """Resize image to target resolution."""
    current_h, current_w = image.shape[:2]
    if current_w == target_width and current_h == target_height:
        return image
    interpolation = cv2.INTER_AREA if (target_width < current_w or target_height < current_h) else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def apply_resolution_to_images(images_dict, resolution_slides, use_parallel=None):
    """Apply fixed resolutions to selected slides."""
    resized_images = []
    for slide_name, target_w, target_h in resolution_slides:
        if slide_name in images_dict:
            resized_images.append(resize_image_to_resolution(images_dict[slide_name], target_w, target_h))
    return resized_images


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except Exception:
        return ""


def _parse_slide_key(val: str) -> str | None:
    """
    Accepts:
      - "8" -> "slide_08"
      - "08" -> "slide_08"
      - "slide_8" -> "slide_08"
      - "slide_08" -> "slide_08"
    """
    s = (val or "").strip().lower()
    if not s:
        return None

    if s.startswith("slide_"):
        tail = s.replace("slide_", "")
        if tail.isdigit():
            return f"slide_{int(tail):02d}"
        return s

    if s.isdigit():
        return f"slide_{int(s):02d}"

    return None


def _ensure_same_dims_as_original(scene_path: str, out_path: str) -> bool:
    """Force output image back to original scene resolution."""
    original = cv2.imread(scene_path)
    if original is None:
        return False
    oh, ow = original.shape[:2]

    img = cv2.imread(out_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    if (w, h) != (ow, oh):
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_path, img)
    return True


def _scale_labels(labels, src_w, src_h, dst_w, dst_h):
    """Safety-net: scale label positions/sizes when image dims differ."""
    if not labels or src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return labels

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    sf = min(sx, sy)

    out = []
    for el in labels:
        e = dict(el)
        e["x"] = int(e.get("x", 0) * sx)
        e["y"] = int(e.get("y", 0) * sy)
        e["width"] = int(e.get("width", 0) * sx)
        e["height"] = int(e.get("height", 0) * sy)

        gf = e.get("global_font", 0)
        if gf and gf != 0:
            try:
                e["global_font"] = float(gf) * sf
            except Exception:
                pass

        out.append(e)
    return out


# ---------------------------
# Text overlay
# ---------------------------
def apply_text_to_images(
    images_dict,
    text_data,
    original_dims_dict,
    app,
    fonts_loaded,
    language,
    use_parallel=None,
    first_slide_font=None,
    rest_slides_font=None,
):
    from Codes.config import USE_PARALLEL_TEXT_PROCESSING

    if use_parallel is None:
        use_parallel = USE_PARALLEL_TEXT_PROCESSING

    if use_parallel and len(images_dict) > 1:
        return _apply_text_parallel(
            images_dict=images_dict,
            text_data=text_data,
            original_dims_dict=original_dims_dict,
            language=language,
            first_slide_font=first_slide_font,
            rest_slides_font=rest_slides_font,
        )

    return _apply_text_sequential(
        images_dict=images_dict,
        text_data=text_data,
        original_dims_dict=original_dims_dict,
        app=app,
        fonts_loaded=fonts_loaded,
    )


def _apply_text_sequential(images_dict, text_data, original_dims_dict, app, fonts_loaded):
    processed_images = {}

    for image_name, img in images_dict.items():
        current_h, current_w = img.shape[:2]

        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            if current_w != orig_w or current_h != orig_h:
                img = resize_image_to_resolution(img, orig_w, orig_h)
                current_h, current_w = img.shape[:2]

        if image_name not in text_data:
            processed_images[image_name] = img
            continue

        labels_list = text_data[image_name]

        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            if current_w != orig_w or current_h != orig_h:
                labels_list = _scale_labels(labels_list, orig_w, orig_h, current_w, current_h)

        first_key = list(text_data.keys())[0] if text_data else image_name
        is_first = (image_name == "slide_01" or image_name == first_key)

        img_with_text = render_image(
            image_name=image_name,
             text_data_list=labels_list,
            fonts_loaded=fonts_loaded,
            is_first_slide=is_first,
            image_data=img,
)

        processed_images[image_name] = img_with_text if img_with_text is not None else img

    return processed_images


def _restore_image_worker(args):
    image_name, img, orig_w, orig_h = args
    current_h, current_w = img.shape[:2]
    if current_w != orig_w or current_h != orig_h:
        img = resize_image_to_resolution(img, orig_w, orig_h)
    return (image_name, img)


def _apply_text_parallel(images_dict, text_data, original_dims_dict, language, first_slide_font=None, rest_slides_font=None):
    from multiprocessing import Pool
    from Codes.config import MAX_TEXT_WORKERS, BASE_DIR
    from Codes.parallel_text_processor import apply_text_parallel

    restored_images = {}
    restore_tasks = []

    for image_name, img in images_dict.items():
        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            restore_tasks.append((image_name, img, orig_w, orig_h))
        else:
            restored_images[image_name] = img

    if restore_tasks:
        workers = min(MAX_TEXT_WORKERS, len(restore_tasks))
        with Pool(processes=workers) as pool:
            results = pool.map(_restore_image_worker, restore_tasks)
        for image_name, img in results:
            restored_images[image_name] = img

    if first_slide_font and rest_slides_font:
        first_font_path = os.path.join(BASE_DIR, first_slide_font) if not os.path.isabs(first_slide_font) else first_slide_font
        rest_font_path = os.path.join(BASE_DIR, rest_slides_font) if not os.path.isabs(rest_slides_font) else rest_slides_font
    else:
        from Codes.config import EN_FIRST_SLIDE_FONT, EN_REST_SLIDES_FONT, AR_FIRST_SLIDE_FONT, AR_REST_SLIDES_FONT
        if language == "en":
            first_font_path = EN_FIRST_SLIDE_FONT
            rest_font_path = EN_REST_SLIDES_FONT
        else:
            first_font_path = AR_FIRST_SLIDE_FONT
            rest_font_path = AR_REST_SLIDES_FONT

    return apply_text_parallel(
        images_dict=restored_images,
        text_data=text_data,
        first_font_path=first_font_path,
        rest_font_path=rest_font_path,
        num_workers=MAX_TEXT_WORKERS,
    )


# ---------------------------
# Head swap: batch + interactive refine
# ---------------------------
def _set_single_attempt_env(attempt_idx: int) -> None:
    """
    Force api_segmiod.py to generate only ONE preview (tryN) and return its path.
    """
    os.environ["SEGMIND_INTERACTIVE"] = "0"
    os.environ["SEGMIND_SINGLE_ATTEMPT"] = "1"
    os.environ["SEGMIND_ATTEMPT_INDEX"] = str(int(attempt_idx))


def _clear_single_attempt_env() -> None:
    """
    Optional cleanup to avoid affecting other modules.
    """
    os.environ["SEGMIND_SINGLE_ATTEMPT"] = "0"
    os.environ.pop("SEGMIND_ATTEMPT_INDEX", None)


def _generate_single_attempt(scene_path: str, face_image_path: str, final_out_path: str, attempt_idx: int) -> str | None:
    """
    Runs ONE attempt using api_segmiod.perform_head_swap in SINGLE ATTEMPT MODE.
    Returns preview path (base_tryN.ext) or None.
    """
    _set_single_attempt_env(attempt_idx)

    preview_path = perform_head_swap(
        target_image_path=scene_path,
        face_image_path=face_image_path,
        output_filename=final_out_path,
        face_url_cached=None,
    )

    if preview_path and os.path.exists(preview_path):
        _ensure_same_dims_as_original(scene_path, preview_path)
        return preview_path

    return None


def _slide_label_from_key(slide_key: str) -> str:
    # slide_04 -> "slide 04"
    if slide_key.startswith("slide_") and slide_key.replace("slide_", "").isdigit():
        return f"slide {int(slide_key.replace('slide_', '')):02d}"
    return slide_key.replace("_", " ")


def _try_label(slide_key: str, attempt_idx: int) -> str:
    # slide_04 + 2 -> "slide 04_try2"
    base = _slide_label_from_key(slide_key)
    return f"{base}_try{attempt_idx}"


def _interactive_refine_before_pdf(api_map: dict, face_image_path: str):
    """
    api_map: {"slide_04": {"scene": path, "out": path}, ...}

    Exact terminal flow requested:
      choose any image you need to change?
      1-slide 01
      ...
      input: 4
      regnerating photo ....
      done
      do you like the result y/n?
      y
      okay choose any result you want to save :
      1-slide 04_try1
      2-slide 04_try2
      input:2
      done saved slide 04
      do you want to retry with another photo?

    NOTE:
      Attempts are infinite here (attempt 1,2,3,...). No wrapping.
    """
    if not api_map:
        return

    slides = sorted(api_map.keys())

    while True:
        print("\nchoose any image you need to change? (n or 0 for exit)")
        for i, s in enumerate(slides, 1):
            print(f"{i}-{_slide_label_from_key(s)}")

        raw = _safe_input("input: ").strip()
        if raw.lower() in ("0", "q", "quit", "exit","n","no"):
            _clear_single_attempt_env()
            return

        slide_key = None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(slides):
                slide_key = slides[idx - 1]
        if not slide_key:
            slide_key = _parse_slide_key(raw)

        if not slide_key or slide_key not in api_map:
            print("Invalid input.")
            continue

        scene_path = api_map[slide_key]["scene"]
        final_out = api_map[slide_key]["out"]

        tries: list[str] = []
        attempt = 1

        while True:
            print("regnerating photo ....")
            preview = _generate_single_attempt(scene_path, face_image_path, final_out, attempt)
            print("done")

            if preview:
                tries.append(preview)

            yn = _safe_input("do you like the result y/n?\n").strip().lower()
            if yn != "y":
                attempt += 1
                continue

            print("okay choose any result you want to save :")
            for i, _p in enumerate(tries, 1):
                # list tries in the same order they were generated
                print(f"{i}-{_try_label(slide_key, i)}")

            pick_raw = _safe_input("input:").strip()
            pick = int(pick_raw) if pick_raw.isdigit() else len(tries)
            if pick < 1 or pick > len(tries):
                pick = len(tries)

            chosen_path = tries[pick - 1]
            shutil.copyfile(chosen_path, final_out)
            _ensure_same_dims_as_original(scene_path, final_out)

            print(f"done saved {_slide_label_from_key(slide_key)}")
            break

        nxt = _safe_input("do you want to retry with another photo?\n").strip().lower()
        if nxt in ("0", "q", "quit", "exit", "n", "no"):
            _clear_single_attempt_env()
            return
        # otherwise loop continues and user can pick another slide from menu
from PIL import Image

def process_head_swap(clean_images_folder, character_image_path, character_name, story_folder, prompts_dict=None, use_parallel=None):
    from PIL import Image
    head_swap_folder = os.path.join(story_folder, "Head_swap")
    os.makedirs(head_swap_folder, exist_ok=True)

    # Use uploaded image name for folder
    uploaded_name = os.path.splitext(os.path.basename(character_image_path))[0]
    char_output_folder = os.path.join(head_swap_folder, uploaded_name)
    os.makedirs(char_output_folder, exist_ok=True)

    api_images_folder = os.path.join(story_folder, "api_images")
    normal_images_folder = os.path.join(story_folder, "normal_images")

    api_images = [f for f in os.listdir(api_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(api_images_folder) else []
    normal_images = [f for f in os.listdir(normal_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(normal_images_folder) else []

    all_images = sorted(api_images + normal_images)
    if not all_images:
        return []

    output_list = []

    for filename in all_images:
        name_no_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(char_output_folder, f"{name_no_ext}.jpg")

        is_api = filename in api_images
        src_path = os.path.join(api_images_folder if is_api else normal_images_folder, filename)

        # Skip if already exists
        if not os.path.exists(out_path):
            if is_api:
                from Codes.api_segmiod import perform_head_swap
                os.environ["SEGMIND_INTERACTIVE"] = "0"
                os.environ["SEGMIND_SINGLE_ATTEMPT"] = "1"
                os.environ["SEGMIND_ATTEMPT_INDEX"] = "1"
                os.environ["SEGMIND_NO_TRY_FILES"] = "1"

                preview = perform_head_swap(
                    target_image_path=src_path,
                    face_image_path=character_image_path,
                    output_filename=out_path,
                    face_url_cached=None,
                )
                if preview and os.path.exists(preview):
                    out_path = preview

                os.environ["SEGMIND_SINGLE_ATTEMPT"] = "0"
                os.environ["SEGMIND_NO_TRY_FILES"] = "0"
                os.environ.pop("SEGMIND_ATTEMPT_INDEX", None)
            else:
                shutil.copyfile(src_path, out_path)

        try:
            img = Image.open(out_path).convert("RGB")
        except Exception:
            continue

        output_list.append({
            "name": name_no_ext,
            "path": out_path,
            "image": img
        })

    return output_list

# def process_head_swap(clean_images_folder, character_image_path, character_name, story_folder, prompts_dict=None, use_parallel=None):
#     """
#     Generate head-swapped slides:
#       1) Batch generate all API slides (Attempt 1 only)
#       2) Copy normal slides
#       3) Interactive refine BEFORE PDF (infinite attempts)
#       4) Reload outputs and return images dict

#     Returns:
#       (processed_images_dict, original_dims_dict)
#     """
#     head_swap_folder = os.path.join(story_folder, "Head_swap")
#     os.makedirs(head_swap_folder, exist_ok=True)

#     char_output_folder = os.path.join(head_swap_folder, character_name)
#     os.makedirs(char_output_folder, exist_ok=True)

#     api_images_folder = os.path.join(story_folder, "api_images")
#     normal_images_folder = os.path.join(story_folder, "normal_images")

#     api_images = [f for f in os.listdir(api_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(api_images_folder) else []
#     normal_images = [f for f in os.listdir(normal_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(normal_images_folder) else []

#     all_images = sorted(api_images + normal_images)
#     if not all_images:
#         return None, None

#     processed_images_dict = {}
#     original_dims_dict = {}

#     api_map = {}

#     print(f"\n📊 Total images: {len(all_images)} | API: {len(api_images)} | Normal: {len(normal_images)}")

#     for filename in all_images:
#         name_no_ext = os.path.splitext(filename)[0]
#         out_path = os.path.join(char_output_folder, f"{name_no_ext}.jpg")

#         is_api = filename in api_images
#         src_path = os.path.join(api_images_folder if is_api else normal_images_folder, filename)

#         dims = get_image_dimensions(src_path)
#         if dims:
#             orig_w, orig_h = dims
#             if orig_w and orig_h:
#                 original_dims_dict[name_no_ext] = (orig_w, orig_h)

#         if is_api:
#             api_map[name_no_ext] = {"scene": src_path, "out": out_path}

#         # reuse if already exists
#         if os.path.exists(out_path):
#             img = cv2.imread(out_path)
#             if img is not None:
#                 processed_images_dict[name_no_ext] = img
#             continue

#         # Normal image: copy as-is
#         if not is_api:
#             img = cv2.imread(src_path)
#             if img is not None:
#                 cv2.imwrite(out_path, img)
#                 processed_images_dict[name_no_ext] = img
#             continue

#         # API image: batch attempt 1
#         print(f"\n🧩 Generating (batch attempt_1): {filename}")
#         _set_single_attempt_env(1)

#         cand = perform_head_swap(
#             target_image_path=src_path,
#             face_image_path=character_image_path,
#             output_filename=out_path,
#             face_url_cached=None,
#         )

#         if cand and os.path.exists(cand):
#             shutil.copyfile(cand, out_path)
#             _ensure_same_dims_as_original(src_path, out_path)

#         img = cv2.imread(out_path)
#         if img is not None:
#             processed_images_dict[name_no_ext] = img

#         if HEAD_SWAP_DELAY and HEAD_SWAP_DELAY > 0:
#             time.sleep(HEAD_SWAP_DELAY)

#     # Interactive refine BEFORE PDF (infinite attempts)
#     if api_map:
#         _interactive_refine_before_pdf(api_map=api_map, face_image_path=character_image_path)

#         # Reload updated outputs
#         for slide_key, meta in api_map.items():
#             outp = meta["out"]
#             if os.path.exists(outp):
#                 img = cv2.imread(outp)
#                 if img is not None:
#                     processed_images_dict[slide_key] = img

#     _clear_single_attempt_env()
#     return (processed_images_dict, original_dims_dict) if processed_images_dict else (None, None)



