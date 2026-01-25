# -*- coding: utf-8 -*-
"""
🖼️ Image Processor Module
- Head swap batch (non-interactive) for all API slides
- THEN review mode: user can pick any slide and generate multiple candidates,
  then choose the best one.
- Force output back to original resolution
- Text overlay (sequential or parallel) + safety scaling for labels
"""

import os
import cv2
import time
import shutil

from Codes.config import HEAD_SWAP_DELAY
from Codes.api_segmiod import perform_head_swap  # keep as-is to avoid breaking other files
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


def _parse_slide_input(val: str) -> str | None:
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
    from Codes.config import USE_PARALLEL_TEXT_PROCESSING, MAX_TEXT_WORKERS

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

        # Safety scaling if needed
        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            if current_w != orig_w or current_h != orig_h:
                labels_list = _scale_labels(labels_list, orig_w, orig_h, current_w, current_h)

        first_key = list(text_data.keys())[0] if text_data else image_name
        is_first = (image_name == "slide_01" or image_name == first_key)

        img_with_text = render_image(
            image_name=image_name,
            text_data_list=labels_list,
            app=app,
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
    from parallel_text_processor import apply_text_parallel

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

    # Resolve font paths
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
# Head swap: batch + review mode
# ---------------------------
def _generate_one_output(scene_path: str, face_image_path: str, output_path: str) -> bool:
    """
    Calls perform_head_swap then forces output to original dimensions.
    Returns True if output exists and is valid.
    """
    result = perform_head_swap(
        target_image_path=scene_path,
        face_image_path=face_image_path,
        output_filename=output_path,
    )

    if not result or not os.path.exists(output_path):
        return False

    _ensure_same_dims_as_original(scene_path, output_path)
    return True


def _review_mode(api_map: dict, character_image_path: str, char_output_folder: str):
    """
    api_map: dict { "slide_08": {"scene": scene_path, "out": out_path} , ... }
    Creates candidates and lets user choose.
    """
    print("\n" + "=" * 60)
    print("🧪 Review Mode: اختار أي سلايد تعيد توليدها")
    print("اكتب رقم السلايد (مثلاً 8) أو slide_08")
    print("اكتب 0 لو خلاص وتكمل للـ PDF")
    print("=" * 60)

    while True:
        val = _safe_input("\n🎯 Slide to refine (0 to continue): ").strip()
        if val in ("0", "q", "quit", "exit"):
            print("✅ خروج من Review Mode.")
            return

        slide_key = _parse_slide_input(val)
        if not slide_key or slide_key not in api_map:
            print("❌ سلايد غير موجود ضمن API slides. جرّب تاني.")
            available = ", ".join(sorted(api_map.keys()))
            print(f"   المتاح: {available}")
            continue

        scene_path = api_map[slide_key]["scene"]
        final_out = api_map[slide_key]["out"]

        # Candidates folder
        cand_dir = os.path.join(char_output_folder, "_review_candidates", slide_key)
        os.makedirs(cand_dir, exist_ok=True)

        candidates = []

        print("\n" + "-" * 60)
        print(f"🖼️ Refining: {slide_key}")
        print(f"Scene: {os.path.basename(scene_path)}")
        print(f"Final output: {os.path.basename(final_out)}")
        print("-" * 60)
        print("هولّد Candidates واحدة واحدة.")
        print("بعد كل واحدة:")
        print("  1) Generate another")
        print("  2) Choose best from generated")
        print("  3) Cancel refinement (keep old)")
        print("-" * 60)

        # IMPORTANT: make headswap interactive during review (so you can accept/retry inside)
        os.environ["SEGMIND_INTERACTIVE"] = "1"

        while True:
            cand_idx = len(candidates) + 1
            cand_path = os.path.join(cand_dir, f"{slide_key}_cand{cand_idx}.jpg")

            print(f"\n✨ Generating candidate #{cand_idx} ...")
            ok = _generate_one_output(scene_path, character_image_path, cand_path)

            if not ok:
                print("❌ فشل توليد Candidate. جرّب تاني.")
            else:
                candidates.append(cand_path)
                print(f"✅ Candidate saved: {cand_path}")

            action = _safe_input("Choose (1=more, 2=select, 3=cancel): ").strip()

            if action == "1":
                continue

            if action == "3":
                print("↩️ Cancel refinement. Keeping old output.")
                break

            if action == "2":
                if not candidates:
                    print("❌ مفيش Candidates لسه.")
                    continue

                print("\n📌 Candidates:")
                for i, p in enumerate(candidates, 1):
                    print(f"  {i}) {p}")

                pick = _safe_input("Select best candidate number: ").strip()
                if not pick.isdigit() or not (1 <= int(pick) <= len(candidates)):
                    print("❌ اختيار غلط.")
                    continue

                chosen = candidates[int(pick) - 1]

                # Replace final output with chosen
                os.makedirs(os.path.dirname(final_out) or ".", exist_ok=True)
                shutil.copyfile(chosen, final_out)

                # Force dims again for safety
                _ensure_same_dims_as_original(scene_path, final_out)

                print(f"🎉 Selected ✅ {chosen}")
                print(f"✅ Updated final: {final_out}")
                break

            print("❌ اختيار غلط. جرّب 1 أو 2 أو 3.")


def process_head_swap(clean_images_folder, character_image_path, character_name, story_folder, prompts_dict=None, use_parallel=None):
    """
    Generate head-swapped slides:
      - Batch generate all API slides (non-interactive)
      - Copy normal slides
      - Then Review Mode: user can refine selected slide(s) by generating candidates and choosing best.

    Returns:
      (processed_images_dict, original_dims_dict)
    """
    head_swap_folder = os.path.join(story_folder, "Head_swap")
    os.makedirs(head_swap_folder, exist_ok=True)

    char_output_folder = os.path.join(head_swap_folder, character_name)
    os.makedirs(char_output_folder, exist_ok=True)

    api_images_folder = os.path.join(story_folder, "api_images")
    normal_images_folder = os.path.join(story_folder, "normal_images")

    api_images = [f for f in os.listdir(api_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(api_images_folder) else []
    normal_images = [f for f in os.listdir(normal_images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.exists(normal_images_folder) else []

    all_images = sorted(api_images + normal_images)
    if not all_images:
        return None, None

    processed_images_dict = {}
    original_dims_dict = {}

    # Keep map for review mode (API only)
    api_map = {}

    # Batch mode: no interactive input while generating all slides
    os.environ["SEGMIND_INTERACTIVE"] = "0"

    print(f"\n📊 Total images: {len(all_images)} | API: {len(api_images)} | Normal: {len(normal_images)}")

    for filename in all_images:
        name_no_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(char_output_folder, f"{name_no_ext}.jpg")

        is_api = filename in api_images
        src_path = os.path.join(api_images_folder if is_api else normal_images_folder, filename)

        dims = get_image_dimensions(src_path)
        if dims:
            orig_w, orig_h = dims
            if orig_w and orig_h:
                original_dims_dict[name_no_ext] = (orig_w, orig_h)

        # If exists, reuse
        if os.path.exists(out_path):
            img = cv2.imread(out_path)
            if img is not None:
                processed_images_dict[name_no_ext] = img

            if is_api:
                api_map[name_no_ext] = {"scene": src_path, "out": out_path}
            continue

        # Normal image: copy
        if not is_api:
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(out_path, img)
                processed_images_dict[name_no_ext] = img
            continue

        # API image: generate once (non-interactive)
        print(f"\n🧩 Generating (batch): {filename}")
        ok = _generate_one_output(src_path, character_image_path, out_path)
        if ok:
            img = cv2.imread(out_path)
            if img is not None:
                processed_images_dict[name_no_ext] = img

        api_map[name_no_ext] = {"scene": src_path, "out": out_path}

        if HEAD_SWAP_DELAY and HEAD_SWAP_DELAY > 0:
            time.sleep(HEAD_SWAP_DELAY)

    # ---------------------------
    # Review mode (after all done)
    # ---------------------------
    if api_map:
        _review_mode(api_map=api_map, character_image_path=character_image_path, char_output_folder=char_output_folder)

        # Reload any updated outputs (after review selection)
        for slide_key, meta in api_map.items():
            outp = meta["out"]
            if os.path.exists(outp):
                img = cv2.imread(outp)
                if img is not None:
                    processed_images_dict[slide_key] = img

    return (processed_images_dict, original_dims_dict) if processed_images_dict else (None, None)
