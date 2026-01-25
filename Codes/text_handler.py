# -*- coding: utf-8 -*-
"""
Text Handler Module
- Loads fonts
- Reads/cleans text JSON
- Renders HTML text onto images (Qt)
- Optional helper to integrate with Segmind head-swap pipeline (api_segmiod.py)
"""

import os
import json
import re
import cv2
import numpy as np

from PySide6.QtWidgets import QLabel, QGraphicsDropShadowEffect, QApplication
from PySide6.QtGui import QPixmap, QPainter, QFontDatabase, QColor, QImage
from PySide6.QtCore import Qt, QBuffer, QIODevice

from Codes.config import (
    EN_FIRST_SLIDE_FONT, EN_REST_SLIDES_FONT,
    AR_FIRST_SLIDE_FONT, AR_REST_SLIDES_FONT,
    ENABLE_TEXT_SHADOW,
    SHADOW_BLUR_RADIUS, SHADOW_COLOR, SHADOW_OFFSET_X, SHADOW_OFFSET_Y,
)


# =========================
# Fonts
# =========================

def load_custom_fonts(language: str,
                      first_slide_font_path: str | None = None,
                      rest_slides_font_path: str | None = None,
                      base_dir: str | None = None) -> dict:
    """
    Load custom fonts based on language.
    If info.txt provided relative paths, pass base_dir to resolve them.
    Returns: {"first": "FamilyName", "rest": "FamilyName"}
    """
    fonts_loaded: dict = {}

    if first_slide_font_path and base_dir:
        first_font = os.path.join(base_dir, first_slide_font_path)
    elif language == "en":
        first_font = EN_FIRST_SLIDE_FONT
    else:
        first_font = AR_FIRST_SLIDE_FONT

    if rest_slides_font_path and base_dir:
        rest_font = os.path.join(base_dir, rest_slides_font_path)
    elif language == "en":
        rest_font = EN_REST_SLIDES_FONT
    else:
        rest_font = AR_REST_SLIDES_FONT

    # First slide font
    if os.path.exists(first_font):
        font_id = QFontDatabase.addApplicationFont(first_font)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                fonts_loaded["first"] = families[0]
                print(f"[Fonts] Loaded FIRST font: {families[0]} ({os.path.basename(first_font)})")
        else:
            print(f"[Fonts] Failed to load FIRST font: {first_font}")
    else:
        print(f"[Fonts] FIRST font not found: {first_font}")

    # Rest slides font
    if os.path.exists(rest_font):
        font_id = QFontDatabase.addApplicationFont(rest_font)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                fonts_loaded["rest"] = families[0]
                print(f"[Fonts] Loaded REST font:  {families[0]} ({os.path.basename(rest_font)})")
        else:
            print(f"[Fonts] Failed to load REST font: {rest_font}")
    else:
        print(f"[Fonts] REST font not found: {rest_font}")

    return fonts_loaded


# =========================
# HTML helpers
# =========================

def inject_font_family(html_text: str, font_family: str | None) -> str:
    if not font_family:
        return html_text

    # Remove any existing font-family declarations
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

    def repl(match):
        original_size = float(match.group(1))
        unit = match.group(2)
        new_size = int(original_size * global_font)
        new_size = max(1, new_size)
        return f"font-size:{new_size}{unit}"

    return re.sub(r"font-size:(\d+(?:\.\d+)?)(pt|px)", repl, html_text)


def make_waw_transparent(html_text: str) -> str:
    """
    Make standalone Arabic letter 'و' transparent if it is explicitly styled as black.
    """
    html_text = re.sub(
        r"(<span[^>]*color:\s*#000000[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:#000000", "color:transparent") + "و" + m.group(2),
        html_text
    )
    html_text = re.sub(
        r"(<span[^>]*color:\s*#000(?![0-9a-fA-F])[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:#000", "color:transparent") + "و" + m.group(2),
        html_text
    )
    html_text = re.sub(
        r"(<span[^>]*color:\s*black[^>]*>)\s*و\s*(</span>)",
        lambda m: m.group(1).replace("color:black", "color:transparent") + "و" + m.group(2),
        html_text
    )
    return html_text


def replace_name_in_html(html_text: str, user_name: str, is_first_slide: bool = False, language: str = "en") -> str:
    if language == "en" and "[*NAME*]" in html_text:
        html_text = html_text.replace("[*NAME*]", user_name.upper() if is_first_slide else user_name)
    elif language == "ar" and "[*الاسم*]" in html_text:
        html_text = html_text.replace("[*الاسم*]", user_name.upper() if is_first_slide else user_name)
    return html_text


# =========================
# JSON text reader (with HTML cleaning)
# =========================

def read_text_data(file_path: str, user_name: str = "", language: str = "en") -> dict | None:
    """
    Reads JSON that contains: {"slide_01": [{"html": "...", "x":.., "y":.., ...}, ...], ...}
    Cleans broken quotes inside 'html' fields.
    Replaces name placeholders if user_name is provided.
    """
    if not os.path.exists(file_path):
        print(f"[Text] File not found: {file_path}")
        return None

    try:
        raw_content = open(file_path, "r", encoding="utf-8").read()
        if not raw_content.strip():
            return None

        result = []
        i = 0
        while i < len(raw_content):
            if raw_content[i:i+7] == '"html":':
                result.append(raw_content[i:i+7])
                i += 7

                while i < len(raw_content) and raw_content[i] in " \t":
                    result.append(raw_content[i])
                    i += 1

                if i < len(raw_content) and raw_content[i] == '"':
                    result.append('"')
                    i += 1

                    html_chars = []
                    while i < len(raw_content):
                        ch = raw_content[i]

                        if ch == '"':
                            peek = raw_content[i+1:i+20].lstrip()
                            if peek.startswith(",") or peek.startswith("}"):
                                cleaned_html = "".join(html_chars)

                                cleaned_html = cleaned_html.replace('\\"', "'").replace("\\'", "'")
                                cleaned_html = re.sub(r'(?<!=)"(?![>\s])', "'", cleaned_html)

                                cleaned_html = cleaned_html.replace("\\n", " ").replace("\\t", " ").replace("\\r", "")
                                cleaned_html = cleaned_html.replace("\\/", "/")
                                cleaned_html = cleaned_html.replace(',"', ",'").replace('",', "',")

                                result.append(cleaned_html)
                                result.append('"')
                                i += 1
                                break
                            else:
                                html_chars.append("'")
                                i += 1

                        elif ch == "\\" and i + 1 < len(raw_content):
                            nxt = raw_content[i + 1]
                            if nxt in ['"', "'"]:
                                html_chars.append("'")
                                i += 2
                            elif nxt == "\\":
                                html_chars.append("\\")
                                i += 2
                            elif nxt in "ntr":
                                html_chars.append(" ")
                                i += 2
                            else:
                                i += 1
                        else:
                            html_chars.append(ch)
                            i += 1
                    continue

            result.append(raw_content[i])
            i += 1

        content = "".join(result)
        data = json.loads(content)

        if user_name:
            slide_index = 0
            for image_name, labels_list in data.items():
                if isinstance(labels_list, list):
                    for label in labels_list:
                        if isinstance(label, dict) and "html" in label:
                            label["html"] = replace_name_in_html(
                                label["html"], user_name,
                                is_first_slide=(slide_index == 0),
                                language=language
                            )
                slide_index += 1

        return data

    except json.JSONDecodeError as e:
        print(f"[Text] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[Text] Error reading text data: {e}")
        return None


# =========================
# Scaling helpers (optional)
# =========================

def scale_text_positions(labels_list: list, ratio_x: float, ratio_y: float) -> list:
    import math
    font_ratio = math.sqrt(max(1e-9, ratio_x * ratio_y))

    scaled = []
    for item in labels_list:
        new_item = item.copy()
        new_item["x"] = int(item.get("x", 0) * ratio_x)
        new_item["y"] = int(item.get("y", 0) * ratio_y)
        new_item["width"] = int(item.get("width", 400) * ratio_x)
        new_item["height"] = int(item.get("height", 200) * ratio_y)

        gf = item.get("global_font", 0)
        if gf != 0:
            new_item["global_font"] = gf * font_ratio

        scaled.append(new_item)

    return scaled


# =========================
# Rendering (Qt)
# =========================

def _ensure_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def render_image(
    image_path: str | None = None,
    image_name: str = "",
    text_data_list: list | None = None,
    fonts_loaded: dict | None = None,
    is_first_slide: bool = False,
    image_data=None,
    silent: bool = False,
):
    """
    Render HTML labels onto image.
    Provide either image_path or image_data (OpenCV BGR numpy array).
    Returns OpenCV BGR numpy array or None.
    """
    if text_data_list is None:
        text_data_list = []
    if fonts_loaded is None:
        fonts_loaded = {}

    _ensure_qt_app()

    if not silent:
        print(f"[Render] Image: {image_name}")

    # Load base pixmap
    if image_data is not None:
        h, w, _ = image_data.shape
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        base_pixmap = QPixmap.fromImage(q_img)
    elif image_path:
        base_pixmap = QPixmap(image_path)
    else:
        if not silent:
            print("[Render] No image_path or image_data provided.")
        return None

    if base_pixmap.isNull():
        if not silent:
            print("[Render] Failed to load image.")
        return None

    # Pick font family
    font_family = None
    if is_first_slide and "first" in fonts_loaded:
        font_family = fonts_loaded["first"]
    elif (not is_first_slide) and "rest" in fonts_loaded:
        font_family = fonts_loaded["rest"]

    final_pixmap = QPixmap(base_pixmap.size())
    final_pixmap.fill(Qt.transparent)

    painter = QPainter(final_pixmap)
    painter.drawPixmap(0, 0, base_pixmap)

    for idx, item in enumerate(text_data_list, 1):
        html = item.get("html", "")
        x = int(item.get("x", 0))
        y = int(item.get("y", 0))
        w = int(item.get("width", 400))
        h = int(item.get("height", 200))
        global_font = float(item.get("global_font", 0) or 0)

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

        if not silent:
            print(f"[Render] Label {idx}: ({x},{y}) [{w}x{h}] FontScale={global_font:.2f}")

    painter.end()

    # Convert final_pixmap -> OpenCV BGR
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    final_pixmap.save(buffer, "PNG")
    buffer.close()

    arr = np.frombuffer(buffer.data(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# =========================
# Worker (parallel usage)
# =========================

def render_image_worker(args):
    """
    args:
      (image_name, image_bytes, text_data_list, is_first_slide,
       first_font_path, rest_font_path, language, base_dir)
    returns:
      (image_name, result_bytes_or_None, status_str)
    """
    (image_name, image_bytes, text_data_list, is_first_slide,
     first_font_path, rest_font_path, language, base_dir) = args

    try:
        _ensure_qt_app()

        fonts_loaded = load_custom_fonts(
            language=language,
            first_slide_font_path=first_font_path,
            rest_slides_font_path=rest_font_path,
            base_dir=base_dir,
        )

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return (image_name, None, "Failed to decode image bytes")

        out_cv = render_image(
            image_name=image_name,
            text_data_list=text_data_list,
            fonts_loaded=fonts_loaded,
            is_first_slide=is_first_slide,
            image_data=img_cv,
            silent=True,
        )

        if out_cv is None:
            return (image_name, None, "Render failed")

        ok, png = cv2.imencode(".png", out_cv)
        if not ok:
            return (image_name, None, "Failed to encode output PNG")

        return (image_name, png.tobytes(), "OK")

    except Exception as e:
        return (image_name, None, f"Worker error: {e}")


# =========================
# Integration with Segmind pipeline (api_segmiod.py)
# =========================

def segmind_swap_then_render_text(
    target_image_path: str,
    face_image_path: str,
    output_filename: str,
    text_data_list: list,
    fonts_loaded: dict,
    is_first_slide: bool = False,
):
    """
    1) Calls api_segmiod.perform_head_swap(...)
    2) Loads the swapped image
    3) Renders text on top
    4) Saves final output to output_filename

    Returns: output_filename or None
    """
    try:
        from api_segmiod import perform_head_swap  # your Segmind module name

        swapped_path = perform_head_swap(
            target_image_path=target_image_path,
            face_image_path=face_image_path,
            output_filename=output_filename,
            face_url_cached=None,
        )

        if not swapped_path or not os.path.exists(swapped_path):
            print("[Segmind+Text] Head swap failed.")
            return None

        base_img = cv2.imread(swapped_path)
        if base_img is None:
            print("[Segmind+Text] Failed to read swapped output image.")
            return None

        final_img = render_image(
            image_name=os.path.basename(swapped_path),
            text_data_list=text_data_list,
            fonts_loaded=fonts_loaded,
            is_first_slide=is_first_slide,
            image_data=base_img,
            silent=False,
        )

        if final_img is None:
            print("[Segmind+Text] Text render failed.")
            return None

        cv2.imwrite(output_filename, final_img)
        print(f"[Segmind+Text] Saved final: {os.path.basename(output_filename)}")
        return output_filename

    except Exception as e:
        print(f"[Segmind+Text] Error: {e}")
        return None
