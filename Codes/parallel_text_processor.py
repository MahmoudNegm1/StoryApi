# -*- coding: utf-8 -*-
"""
Parallel Text Processor - Standalone
- Parallel HTML-text rendering on images using PySide6
- Returns processed OpenCV images (BGR)
"""

import os
import re
import time
import shutil
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image

from config import (
    ENABLE_TEXT_SHADOW,
    SHADOW_BLUR_RADIUS,
    SHADOW_COLOR,
    SHADOW_OFFSET_X,
    SHADOW_OFFSET_Y,
)


# =========================
# HTML helpers
# =========================

def inject_font_family(html_text: str, font_family: str | None) -> str:
    if not font_family:
        return html_text

    html_text = re.sub(r"font-family:\s*[^;'\"]+[;\"]", "", html_text)
    html_text = re.sub(r"font-family:\s*'[^']+'[;\"]?", "", html_text)
    html_text = re.sub(r'font-family:\s*"[^"]+"[;\"]?', "", html_text)

    def add_font_to_style(match):
        style_content = match.group(1)
        new_style = f"font-family: '{font_family}' !important; " + style_content
        return f'style="{new_style}"'

    html_text = re.sub(r'style="([^"]*)"', add_font_to_style, html_text)

    base_style = f"font-family: '{font_family}' !important;"
    html_text = re.sub(r"<p(\s|>)", f'<p style="{base_style}"\\1', html_text)
    html_text = re.sub(r"<span(\s|>)", f'<span style="{base_style}"\\1', html_text)
    html_text = re.sub(r"<div(\s|>)", f'<div style="{base_style}"\\1', html_text)

    return html_text


def scale_font_sizes(html_text: str, global_font: float) -> str:
    if not global_font or global_font == 0:
        return html_text

    def replace_font_size(match):
        original_size = float(match.group(1))
        unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else "pt"
        new_size = int(original_size * global_font)
        new_size = max(1, new_size)
        return f"font-size:{new_size}{unit}"

    return re.sub(r"font-size:(\d+(?:\.\d+)?)(pt|px)?", replace_font_size, html_text)


def make_waw_transparent(html_text: str) -> str:
    html_text = re.sub(
        r"(<span[^>]*color:\s*#000000[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:#000000", "color:transparent") + "و" + m.group(2),
        html_text,
    )

    html_text = re.sub(
        r"(<span[^>]*color:\s*#000(?![0-9a-fA-F])[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:#000", "color:transparent") + "و" + m.group(2),
        html_text,
    )

    html_text = re.sub(
        r"(<span[^>]*color:\s*black[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:black", "color:transparent") + "و" + m.group(2),
        html_text,
    )

    return html_text


# =========================
# Worker
# =========================

def process_single_image_worker(args):
    """
    Worker: renders text on a single image file path and returns PNG bytes.
    args:
      (image_name, image_path, text_data_list, is_first_slide, first_font_path, rest_font_path)
    returns:
      (image_name, png_bytes_or_None, status_str)
    """
    (image_name, image_path, text_data_list, is_first_slide, first_font_path, rest_font_path) = args

    try:
        from PySide6.QtWidgets import QApplication, QLabel, QGraphicsDropShadowEffect
        from PySide6.QtGui import QPixmap, QPainter, QFontDatabase, QColor
        from PySide6.QtCore import Qt, QBuffer, QIODevice

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        fonts_loaded = {}

        if first_font_path and os.path.exists(first_font_path):
            font_id = QFontDatabase.addApplicationFont(first_font_path)
            if font_id != -1:
                fams = QFontDatabase.applicationFontFamilies(font_id)
                if fams:
                    fonts_loaded["first"] = fams[0]

        if rest_font_path and os.path.exists(rest_font_path):
            font_id = QFontDatabase.addApplicationFont(rest_font_path)
            if font_id != -1:
                fams = QFontDatabase.applicationFontFamilies(font_id)
                if fams:
                    fonts_loaded["rest"] = fams[0]

        font_family = None
        if is_first_slide and "first" in fonts_loaded:
            font_family = fonts_loaded["first"]
        elif (not is_first_slide) and "rest" in fonts_loaded:
            font_family = fonts_loaded["rest"]

        base_pixmap = QPixmap(str(image_path))
        if base_pixmap.isNull():
            return (image_name, None, "Failed to load image")

        result_pixmap = QPixmap(base_pixmap.size())
        result_pixmap.fill(Qt.transparent)

        painter = QPainter(result_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)

        for element in text_data_list:
            html = element.get("html", "")
            x = int(element.get("x", 0))
            y = int(element.get("y", 0))
            w = int(element.get("width", 400))
            h = int(element.get("height", 200))
            global_font = float(element.get("global_font", 0) or 0)

            if font_family:
                html = inject_font_family(html, font_family)
            if global_font != 0:
                html = scale_font_sizes(html, global_font)

            html = make_waw_transparent(html)

            label = QLabel()
            label.setText(html)
            label.setWordWrap(True)
            label.setStyleSheet("background: transparent;")
            label.setGeometry(x, y, w, h)

            if ENABLE_TEXT_SHADOW:
                shadow = QGraphicsDropShadowEffect()
                shadow.setBlurRadius(SHADOW_BLUR_RADIUS)
                shadow.setColor(QColor(*SHADOW_COLOR))
                shadow.setOffset(SHADOW_OFFSET_X, SHADOW_OFFSET_Y)
                label.setGraphicsEffect(shadow)

            pix = label.grab()
            painter.drawPixmap(x, y, pix)

        painter.end()

        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        result_pixmap.save(buffer, "PNG")
        buffer.close()

        return (image_name, bytes(buffer.data()), "OK")

    except Exception as e:
        return (image_name, None, f"Worker error: {e}")


# =========================
# Main Parallel API
# =========================

def apply_text_parallel(images_dict: dict,
                        text_data: dict,
                        first_font_path: str,
                        rest_font_path: str,
                        num_workers: int | None = None) -> dict:
    """
    Parallel add text to images.

    images_dict: {image_name: cv2_image(BGR)}
    text_data:  {image_name: [ {html,x,y,width,height,global_font}, ... ]}

    Returns:
      processed_images: {image_name: cv2_image(BGR)}
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"\n[Parallel] Starting with {num_workers} workers")

    temp_dir = tempfile.mkdtemp(prefix="parallel_text_")
    temp_path = Path(temp_dir)

    try:
        tasks = []
        for idx, (image_name, img) in enumerate(images_dict.items()):
            if image_name not in text_data:
                continue

            labels_list = text_data[image_name]
            is_first = (idx == 0)

            tmp_file = temp_path / f"{image_name}.png"
            cv2.imwrite(str(tmp_file), img)

            tasks.append((
                image_name,
                str(tmp_file),
                labels_list,
                is_first,
                first_font_path,
                rest_font_path
            ))

        if not tasks:
            print("[Parallel] No tasks found. Returning original images.")
            return images_dict

        print(f"[Parallel] Prepared {len(tasks)} tasks")
        print("[Parallel] Processing...")

        start = time.time()

        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_image_worker, tasks)

        processed_images = {}
        failed = 0

        for i, (image_name, image_bytes, status) in enumerate(results, 1):
            if image_bytes is not None:
                nparr = np.frombuffer(image_bytes, np.uint8)
                out = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if out is not None:
                    processed_images[image_name] = out
                    print(f"[{i}/{len(tasks)}] OK  {image_name}")
                else:
                    failed += 1
                    print(f"[{i}/{len(tasks)}] FAIL {image_name} - decode failed")
                    if image_name in images_dict:
                        processed_images[image_name] = images_dict[image_name]
            else:
                failed += 1
                print(f"[{i}/{len(tasks)}] FAIL {image_name} - {status}")
                if image_name in images_dict:
                    processed_images[image_name] = images_dict[image_name]

        # Add untouched images
        for image_name, img in images_dict.items():
            if image_name not in processed_images:
                processed_images[image_name] = img

        elapsed = time.time() - start
        ok_count = len(tasks) - failed

        print("\n" + "=" * 60)
        print("[Parallel] Done")
        print(f"[Parallel] Success: {ok_count}/{len(tasks)}")
        if failed:
            print(f"[Parallel] Failed:  {failed}/{len(tasks)}")
        print(f"[Parallel] Time:    {elapsed:.2f}s")
        if elapsed > 0:
            print(f"[Parallel] Speed:   {len(tasks)/elapsed:.2f} img/s")
        print("=" * 60 + "\n")

        return processed_images

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[Parallel] Warning: failed to remove temp dir: {e}")


# =========================
# PDF helper
# =========================

def create_pdf_from_images(images_list: list, output_path: str) -> bool:
    if not images_list:
        print("[PDF] No images provided.")
        return False

    print("[PDF] Creating PDF...")

    pil_images = []
    for idx, img in enumerate(images_list, 1):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        if pil_img.mode == "RGBA":
            rgb_img = Image.new("RGB", pil_img.size, (255, 255, 255))
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            pil_images.append(rgb_img)
        else:
            pil_images.append(pil_img.convert("RGB"))

        print(f"[PDF] Converting {idx}/{len(images_list)}")

    try:
        first = pil_images[0]
        rest = pil_images[1:] if len(pil_images) > 1 else []
        first.save(
            output_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=rest if rest else None,
        )
        print(f"[PDF] Done: {output_path}")
        return True

    except Exception as e:
        print(f"[PDF] Failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Parallel Text Processor - Standalone")
    print("=" * 60 + "\n")
    print("Import and use:")
    print("  from parallel_text_processor import apply_text_parallel")
