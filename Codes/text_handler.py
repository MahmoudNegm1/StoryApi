# -*- coding: utf-8 -*-
"""
Text Handler Module (FIXED + DEBUG)
- Stable HTML render using QTextDocument + QGraphicsScene (captures shadow reliably)
- Auto-scales label positions if current image resolution != design resolution (from info.txt)
- Strong debug prints controlled by env vars:
    TEXT_DEBUG=1
    TEXT_DEBUG_HTML=1   (prints full html)
"""

import os
import json
import re
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QGraphicsScene,
    QGraphicsTextItem,
)
from PySide6.QtGui import (
    QFontDatabase,
    QColor,
    QImage,
    QPainter,
    QTextDocument,
)
from PySide6.QtCore import Qt, QRectF

from Codes.config import (
    EN_FIRST_SLIDE_FONT, EN_REST_SLIDES_FONT,
    AR_FIRST_SLIDE_FONT, AR_REST_SLIDES_FONT,
    ENABLE_TEXT_SHADOW,
    SHADOW_BLUR_RADIUS, SHADOW_COLOR, SHADOW_OFFSET_X, SHADOW_OFFSET_Y,
)

# =========================
# Debug helpers
# =========================

DEBUG = os.environ.get("TEXT_DEBUG", "0").strip() in ("1", "true", "True", "YES", "yes")
DEBUG_HTML = os.environ.get("TEXT_DEBUG_HTML", "0").strip() in ("1", "true", "True", "YES", "yes")


def _dprint(msg: str):
    if DEBUG:
        print(msg)


def _short(s: str, n: int = 180) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= n:
        return s
    return s[:n] + "..."


# =========================
# Auto-load design resolutions from info.txt
# =========================

_RES_MAP_CACHE = None


def _find_info_txt() -> str | None:
    # 1) ENV override
    envp = os.environ.get("TEXT_INFO_PATH")
    if envp and os.path.exists(envp):
        return envp

    # 2) Try next to this file
    here = Path(__file__).resolve().parent
    p1 = here / "info.txt"
    if p1.exists():
        return str(p1)

    # 3) Try project root (one/two levels up)
    for up in [here.parent, here.parent.parent, Path.cwd()]:
        p = up / "info.txt"
        if p.exists():
            return str(p)

    return None


def _load_resolution_map() -> dict[str, tuple[int, int]]:
    global _RES_MAP_CACHE
    if _RES_MAP_CACHE is not None:
        return _RES_MAP_CACHE

    res_map: dict[str, tuple[int, int]] = {}
    info_path = _find_info_txt()
    if not info_path:
        _RES_MAP_CACHE = res_map
        _dprint("[Info] info.txt not found -> no autoscale map")
        return _RES_MAP_CACHE

    try:
        info = json.loads(open(info_path, "r", encoding="utf-8").read())
        for name, w, h in info.get("resolution_slides", []):
            res_map[str(name)] = (int(w), int(h))
        _RES_MAP_CACHE = res_map
        _dprint(f"[Info] Loaded resolution map from: {info_path} ({len(res_map)} slides)")
        return _RES_MAP_CACHE
    except Exception as e:
        _RES_MAP_CACHE = {}
        _dprint(f"[Info] Failed to read info.txt: {e}")
        return _RES_MAP_CACHE


# =========================
# Fonts
# =========================

def load_custom_fonts(language: str,
                      first_slide_font_path: str | None = None,
                      rest_slides_font_path: str | None = None,
                      base_dir: str | None = None) -> dict:
    """
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
                _dprint(f"[Fonts] Loaded FIRST: {families[0]} ({os.path.basename(first_font)})")
        else:
            _dprint(f"[Fonts] Failed to load FIRST font: {first_font}")
    else:
        _dprint(f"[Fonts] FIRST font not found: {first_font}")

    # Rest slides font
    if os.path.exists(rest_font):
        font_id = QFontDatabase.addApplicationFont(rest_font)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                fonts_loaded["rest"] = families[0]
                _dprint(f"[Fonts] Loaded REST:  {families[0]} ({os.path.basename(rest_font)})")
        else:
            _dprint(f"[Fonts] Failed to load REST font: {rest_font}")
    else:
        _dprint(f"[Fonts] REST font not found: {rest_font}")

    return fonts_loaded


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


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def scale_font_sizes(html_text: str, global_font: float) -> str:
    if not global_font or global_font == 0:
        return html_text

    gf = _clamp(global_font, 0.1, 10.0)

    def repl(match):
        original_size = float(match.group(1))
        unit = match.group(2) if match.group(2) else "pt"
        new_size = int(original_size * gf)
        new_size = max(1, new_size)
        return f"font-size:{new_size}{unit}"

    return re.sub(r"font-size:(\d+(?:\.\d+)?)(pt|px)?", repl, html_text)


def make_waw_transparent(html_text: str) -> str:
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
    if not user_name:
        return html_text

    repl = ("  " + user_name.upper()) if is_first_slide else ("  " + user_name)

    if language == "en":
        html_text = html_text.replace("[*NAME*]", repl)
        html_text = html_text.replace("[*Name*]", repl)
    elif language == "ar":
        html_text = html_text.replace("[*الاسم*]", repl)
        html_text = html_text.replace("[*اسم*]", repl)

    return html_text


# =========================
# JSON text reader
# =========================

def read_text_data(file_path: str, user_name: str = "", language: str = "en") -> dict | None:
    if not os.path.exists(file_path):
        print(f"[Text] File not found: {file_path}")
        return None

    try:
        raw_content = open(file_path, "r", encoding="utf-8").read()
        if not raw_content.strip():
            return None

        # Keep your "clean broken quotes inside html" logic as-is
        result = []
        i = 0
        while i < len(raw_content):
            if raw_content[i:i + 7] == '"html":':
                result.append(raw_content[i:i + 7])
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
                            peek = raw_content[i + 1:i + 20].lstrip()
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
# Qt app
# =========================

def _ensure_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =========================
# Rendering core
# =========================

def _render_html_to_qimage(
    html: str,
    w: int,
    h: int,
    shadow: bool,
    blur_radius: int,
    shadow_color_rgba: tuple,
    shadow_offset: tuple[int, int],
) -> QImage:
    """
    Render html into transparent QImage using QTextDocument + QGraphicsScene.
    This captures QGraphicsDropShadowEffect correctly.
    """
    html = html or ""

    doc = QTextDocument()
    doc.setHtml(html)
    doc.setTextWidth(max(1, int(w)))

    item = QGraphicsTextItem()
    item.setDocument(doc)
    item.setDefaultTextColor(QColor(255, 255, 255, 255))  # html colors override anyway

    if shadow:
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(int(blur_radius))
        eff.setColor(QColor(*shadow_color_rgba))
        eff.setOffset(int(shadow_offset[0]), int(shadow_offset[1]))
        item.setGraphicsEffect(eff)

    scene = QGraphicsScene()
    scene.addItem(item)

    img = QImage(int(w), int(h), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)

    scene.render(p, QRectF(0, 0, w, h), QRectF(0, 0, w, h))
    p.end()
    return img


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    """
    PySide6-safe QImage -> numpy conversion (NO ptr.setsize usage)
    """
    img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    w = img.width()
    h = img.height()
    bpl = img.bytesPerLine()

    ptr = img.bits()  # memoryview in PySide6
    raw = ptr.tobytes()  # safe
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpl // 4, 4))
    arr = arr[:, :w, :]  # crop padding
    # ARGB32 premultiplied -> BGRA bytes order in memory is usually BGRA for Qt on little-endian,
    # but since we're using OpenCV decode later anyway, we keep it as BGRA then convert.
    bgra = arr.copy()
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return bgr


def _scale_rect(x, y, w, h, rx, ry):
    return int(x * rx), int(y * ry), int(w * rx), int(h * ry)


def render_image(
    image_path: str | None = None,
    image_name: str = "",
    text_data_list: list | None = None,
    fonts_loaded: dict | None = None,
    is_first_slide: bool = False,
    image_data=None,
    silent: bool = False,
    **kwargs,  # IMPORTANT: accepts unexpected args (like app=...) without crashing
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

    app = _ensure_qt_app()

    if not silent:
        _dprint("=" * 80)
        _dprint(f"[Render] Image: {image_name}")
        _dprint(f"[Render] labels_count={len(text_data_list)}")
        _dprint(f"[Render] fonts_loaded={fonts_loaded}")
        _dprint(f"[Render] shadow_enabled={ENABLE_TEXT_SHADOW} blur={SHADOW_BLUR_RADIUS} "
                f"off=({SHADOW_OFFSET_X},{SHADOW_OFFSET_Y}) color={tuple(SHADOW_COLOR)}")

    # Load cv image
    if image_data is not None:
        base_cv = image_data
    elif image_path:
        base_cv = cv2.imread(image_path)
    else:
        if not silent:
            print("[Render] No image_path or image_data provided.")
        return None

    if base_cv is None:
        if not silent:
            print("[Render] Failed to load base image.")
        return None

    base_h, base_w = base_cv.shape[:2]

    # Determine design resolution for this slide from info.txt
    res_map = _load_resolution_map()
    design_w, design_h = res_map.get(image_name, (base_w, base_h))
    rx = base_w / design_w if design_w else 1.0
    ry = base_h / design_h if design_h else 1.0

    if not silent:
        _dprint(f"[Render] CV image size:  {base_w}x{base_h}")
        _dprint(f"[Render] Design size:    {design_w}x{design_h}  scale=({rx:.4f},{ry:.4f})")

    # Choose font family
    font_family = None
    if is_first_slide and "first" in fonts_loaded:
        font_family = fonts_loaded["first"]
    elif (not is_first_slide) and "rest" in fonts_loaded:
        font_family = fonts_loaded["rest"]

    if not silent:
        _dprint(f"[Render] is_first_slide={is_first_slide} -> font_family={font_family}")

    # Convert base_cv -> QImage
    rgb = cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, base_w, base_h, 3 * base_w, QImage.Format_RGB888)

    # Paint on a new ARGB image
    out_img = QImage(base_w, base_h, QImage.Format_ARGB32_Premultiplied)
    out_img.fill(Qt.transparent)

    painter = QPainter(out_img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    # Draw base image
    painter.drawImage(0, 0, qimg)

    # Render each label
    for idx, item in enumerate(text_data_list, 1):
        html = item.get("html", "") or ""
        x = int(item.get("x", 0) or 0)
        y = int(item.get("y", 0) or 0)
        ww = int(item.get("width", 400) or 400)
        hh = int(item.get("height", 200) or 200)
        gf = float(item.get("global_font", 0) or 0)

        # Apply scaling if image size != design size
        sx, sy, sw, sh = _scale_rect(x, y, ww, hh, rx, ry)

        # Build final html
        html2 = html
        if font_family:
            html2 = inject_font_family(html2, font_family)
        if gf != 0:
            html2 = scale_font_sizes(html2, gf)
        html2 = make_waw_transparent(html2)

        if not silent:
            _dprint("-" * 80)
            _dprint(f"[Render] Label {idx}")
            _dprint(f"  rect_design=({x},{y},{ww},{hh}) gf={gf}")
            _dprint(f"  rect_scaled=({sx},{sy},{sw},{sh})")
            if y < 0:
                _dprint("  ⚠️ NOTE: y is negative in DESIGN coords")
            if x < 0:
                _dprint("  ⚠️ NOTE: x is negative in DESIGN coords")
            _dprint(f"  html_raw_preview:  {_short(html)}")
            if DEBUG_HTML:
                _dprint(f"  html_final_full:\n{html2}")
            else:
                _dprint(f"  html_final_preview:{_short(html2)}")

        # Render label to QImage (with shadow)
        label_img = _render_html_to_qimage(
            html=html2,
            w=max(1, sw),
            h=max(1, sh),
            shadow=bool(ENABLE_TEXT_SHADOW),
            blur_radius=int(SHADOW_BLUR_RADIUS),
            shadow_color_rgba=tuple(SHADOW_COLOR),
            shadow_offset=(int(SHADOW_OFFSET_X), int(SHADOW_OFFSET_Y)),
        )

        # Draw it (handles negative coords / clipping naturally)
        painter.drawImage(int(sx), int(sy), label_img)

        # If debugging, sample alpha to detect "fully transparent"
        if not silent:
            # sample a few pixels
            w2, h2 = label_img.width(), label_img.height()
            if w2 >= 20 and h2 >= 20:
                c1 = QColor(label_img.pixel(10, 10)).getRgb()
                c2 = QColor(label_img.pixel(w2 // 2, h2 // 2)).getRgb()
                c3 = QColor(label_img.pixel(w2 - 10, h2 - 10)).getRgb()
                _dprint(f"  label_sample RGBA: (10,10)={c1}  center={c2}  (w-10,h-10)={c3}")
                if c1[3] == 0 and c2[3] == 0 and c3[3] == 0:
                    _dprint("  ⚠️ WARNING: label render looks fully transparent")

    painter.end()

    # QImage -> OpenCV BGR
    out_bgr = _qimage_to_bgr(out_img)
    return out_bgr


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
    try:
        from api_segmiod import perform_head_swap

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
    