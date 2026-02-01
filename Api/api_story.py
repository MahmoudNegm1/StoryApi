# -*- coding: utf-8 -*-
"""
🎭 Stories Production API
Pipeline updated: head-swap selection + PDF/image generation
"""

import os
import sys
import logging
import shutil
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QGraphicsScene, QGraphicsTextItem
from PySide6.QtGui import QPainter, QFontDatabase, QColor, QImage, QTextDocument
from PySide6.QtCore import Qt, QRectF
from PIL import Image

from Codes.config import (
    RESULT_FOLDER,
    BASE_DIR,
    USE_PARALLEL_TEXT_PROCESSING,
    ENABLE_TEXT_SHADOW,
    SHADOW_BLUR_RADIUS,
    SHADOW_COLOR,
    SHADOW_OFFSET_X,
    SHADOW_OFFSET_Y,
)
from Codes.utils import read_info_file
from Codes.text_handler import load_custom_fonts, read_text_data
from Codes.image_processor import (
    process_head_swap,
    apply_text_to_images,
    apply_resolution_to_images,
)
from Codes.pdf_generator import create_pdf_from_images
from Codes.api_segmiod import perform_head_swap

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("StoriesAPI")

# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="Stories Production API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Temporary folders
# --------------------------------------------------
TEMP_UPLOADS = os.path.join(BASE_DIR, "TempUploads")
TEMP_HEAD_SWAP = os.path.join(BASE_DIR, "TempHeadSwaps")
os.makedirs(TEMP_UPLOADS, exist_ok=True)
os.makedirs(TEMP_HEAD_SWAP, exist_ok=True)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_gender_folder(gender: str) -> str:
    if gender == "male":
        return os.path.join(BASE_DIR, "Characters", "Boys")
    if gender == "female":
        return os.path.join(BASE_DIR, "Characters", "Girls")
    raise HTTPException(400, "Invalid gender")

def get_story_folder(gender: str, code: str) -> str:
    base = "Boys" if gender == "male" else "Girls"
    path = os.path.join(BASE_DIR, "Stories", base, code)
    if not os.path.exists(path):
        raise HTTPException(404, f"Story not found: {code}")
    return path

def get_default_character(folder: str):
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(folder, f), os.path.splitext(f)[0]
    return None, None

def build_pdf_filename(pdf_name: str, language: str, user_name: str) -> str:
    if language == "en":
        name = (
            pdf_name.replace("Name", user_name)
                    .replace("name", user_name)
                    .replace("NAME", user_name.upper())
        )
    else:
        name = (
            pdf_name.replace("الاسم", user_name)
                    .replace("اسم", user_name)
        )
    return f"{name}.pdf"

def save_uploaded_file(upload: UploadFile, base_folder: str) -> str:
    """
    Saves an uploaded file inside a subfolder named after the file (without extension).
    Returns the full path of the saved file.
    """
    ext = os.path.splitext(upload.filename)[1]
    file_name_no_ext = os.path.splitext(upload.filename)[0]

    # Create folder inside base_folder
    folder_path = os.path.join(base_folder, file_name_no_ext)
    os.makedirs(folder_path, exist_ok=True)

    # Save file inside that folder
    file_path = os.path.join(folder_path, f"{file_name_no_ext}{ext}")
    with open(file_path, "wb") as f:
        f.write(upload.file.read())

    return file_path

def _find_story_folder_from_slide(slide_path: str) -> Optional[Path]:
    p = Path(slide_path).resolve()
    for parent in p.parents:
        if (parent / "Translations").exists():
            return parent
    return None

def _find_source_scene_for_slide(slide_path: str) -> Optional[Path]:
    story_folder = _find_story_folder_from_slide(slide_path)
    if not story_folder:
        return None

    base_name = Path(slide_path).stem
    if "_try" in base_name:
        base_name = base_name.split("_try")[0]

    candidates = [
        story_folder / "api_images",
        story_folder / "normal_images",
        story_folder / "Images",
        story_folder / "Clean_Images",
    ]

    for folder in candidates:
        for ext in ALLOWED_EXT:
            cand = folder / f"{base_name}{ext}"
            if cand.exists():
                return cand

    return None

def _base_slide_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_try" in stem:
        stem = stem.split("_try")[0]
    return stem

def _set_single_attempt_env(attempt_idx: int) -> None:
    os.environ["SEGMIND_INTERACTIVE"] = "0"
    os.environ["SEGMIND_SINGLE_ATTEMPT"] = "1"
    os.environ["SEGMIND_ATTEMPT_INDEX"] = str(int(attempt_idx))

def _clear_single_attempt_env() -> None:
    os.environ["SEGMIND_SINGLE_ATTEMPT"] = "0"
    os.environ.pop("SEGMIND_ATTEMPT_INDEX", None)

# --------------------------------------------------
# Core pipeline
# --------------------------------------------------
def run_story_pipeline(
    language: str,
    gender: str,
    story_code: str,
    user_name: str,
    character_image_path: str,
    pre_swapped_image_path: str | None = None,
) -> Dict:
    """
    Full story pipeline:
    - Use pre-swapped head image if provided
    - Render text and resolution
    """
    qt_app = None
    try:
        language = language.lower().strip()
        gender = gender.lower().strip()
        user_name = user_name.strip()

        story_folder = get_story_folder(gender, story_code)
        gender_folder = get_gender_folder(gender)

        # ---------------- Story info ----------------
        (
            en_story_name,
            ar_story_name,
            resolution_slides,
            first_slide_font,
            rest_slides_font,
            ar_first_slide_font,
            ar_rest_slides_font,
        ) = read_info_file(story_folder)

        translations = os.path.join(story_folder, "Translations")
        if language == "en":
            text_file = os.path.join(translations, "en_text_data.txt")
            pdf_name = en_story_name or "Story_EN"
        else:
            ar_files = [f for f in os.listdir(translations) if f.startswith("ar_")]
            if not ar_files:
                raise HTTPException(400, "Arabic translation missing")
            text_file = os.path.join(translations, ar_files[0])
            pdf_name = ar_story_name or "Story_AR"

        text_data = read_text_data(text_file, user_name, language)
        if not text_data:
            raise HTTPException(400, "Empty text data")

        # ---------------- Qt & Fonts ----------------
        first_font = first_slide_font if language == "en" else ar_first_slide_font
        rest_font = rest_slides_font if language == "en" else ar_rest_slides_font

        if not USE_PARALLEL_TEXT_PROCESSING or len(text_data) <= 1:
            qt_app = QApplication.instance() or QApplication(sys.argv)
            fonts_loaded = load_custom_fonts(language, first_font, rest_font, BASE_DIR)
        else:
            fonts_loaded = None

        # ---------------- Head swap ----------------
        if pre_swapped_image_path:
            # Use user-selected head swap image
            processed_images = {i: Image.open(pre_swapped_image_path) for i in range(len(text_data))}
            original_dims = {i: processed_images[i].size for i in processed_images}
        else:
            character_name = os.path.splitext(os.path.basename(character_image_path))[0]
            processed_images, original_dims = process_head_swap(
                clean_images_folder=None,
                character_image_path=character_image_path,
                character_name=character_name,
                story_folder=story_folder,
            )

        # ---------------- Text rendering ----------------
        images_with_text = apply_text_to_images(
            processed_images,
            text_data,
            original_dims,
            qt_app,
            fonts_loaded,
            language,
            first_font,
            rest_font,
        )

        # ---------------- Resolution ----------------
        if resolution_slides:
            final_images = apply_resolution_to_images(images_with_text, resolution_slides)
        else:
            final_images = [images_with_text[k] for k in sorted(images_with_text)]

        return {
            "images": final_images,
            "pdf_name": build_pdf_filename(pdf_name, language, user_name),
        }

    finally:
        if qt_app:
            qt_app.quit()

# --------------------------------------------------
# ENDPOINT 1 -> PDF
# --------------------------------------------------
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}

DEBUG = os.environ.get("TEXT_DEBUG", "0").strip().lower() in ("1", "true", "yes")


# =========================================================
# Helpers: detect story folder + images folder
# =========================================================
def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def _pick_images_dir(root: Path) -> Optional[Path]:
    """
    root ممكن يكون:
    - story_folder
    - أو فولدر صور مباشرة
    """
    if root.exists() and root.is_dir() and any(_is_image_file(x) for x in root.iterdir()):
        return root

    candidates = [
        root / "api_images",
        root / "normal_images",
        root / "Images",
        root / "Clean_Images",
    ]
    for c in candidates:
        if c.exists() and c.is_dir() and any(_is_image_file(x) for x in c.iterdir()):
            return c

    # fallback: أول فولدر جواه صور
    if root.exists() and root.is_dir():
        for sub in root.rglob("*"):
            if sub.is_dir():
                try:
                    if any(_is_image_file(x) for x in sub.iterdir()):
                        return sub
                except Exception:
                    pass
    return None


def _find_story_folder(images_dir: Path) -> Optional[Path]:
    """
    استنتاج story_folder الحقيقي (اللي فيه Translations + info.txt غالبًا)
    """
    if images_dir.name.lower() in ("api_images", "normal_images", "images", "clean_images"):
        story = images_dir.parent
        if (story / "Translations").exists():
            return story

    cur = images_dir
    for _ in range(7):
        if (cur / "Translations").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


# =========================================================
# Helpers: HTML name replacement + RTL/BiDi for Arabic names
# =========================================================
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def wrap_name_for_html(name: str) -> str:
    name = name or ""
    # لو الاسم عربي داخل سياق إنجليزي: wrap RTL + unicode-bidi
    if _AR_RE.search(name):
        return f"<span dir='rtl' style='unicode-bidi:plaintext;'>{name}</span>"
    return name

def replace_name_tokens(html: str, user_name: str) -> str:
    u = wrap_name_for_html(user_name)
    # شائع في ملفاتك
    tokens = ["[*NAME*]", "[*name*]", "NAME", "Name", "name", "[*الاسم*]", "الاسم", "اسم"]
    out = html or ""
    for t in tokens:
        out = out.replace(t, u)
    return out


# =========================================================
# Helpers: slide key mapping (slide_01 -> slide_01.png)
# =========================================================
def normalize_text_keys_to_images(text_data: Dict, images_dict: Dict[str, np.ndarray]) -> Dict[str, List[Dict]]:
    """
    text_data keys: slide_01 / slide_02 ...
    images_dict keys: slide_01.png / slide_01.jpg ...
    """
    if not text_data or not images_dict:
        return text_data

    def _try_num(stem: str) -> int:
        if "_try" in stem:
            tail = stem.split("_try")[-1]
            return int(tail) if tail.isdigit() else 0
        return 0

    base_to_full = {}
    base_try_rank = {}

    for img_name in images_dict.keys():
        stem = Path(img_name).stem  # slide_01_try1
        base = stem.split("_try")[0] if "_try" in stem else stem  # slide_01
        rank = _try_num(stem)

        # prefer higher try number when multiple exist
        if base not in base_to_full or rank >= base_try_rank.get(base, -1):
            base_to_full[base] = img_name
            base_try_rank[base] = rank

    fixed = {}
    for k, blocks in text_data.items():
        base = Path(str(k)).stem  # slide_01
        if base in base_to_full:
            fixed[base_to_full[base]] = blocks
        else:
            fixed[str(k)] = blocks
    return fixed


# =========================================================
# Helpers: fonts from info.txt + absolute path resolve
# =========================================================
def _abs_path(p: Optional[str], story_folder: Path) -> str:
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp)

    # try BASE_DIR
    cand = Path(BASE_DIR) / p
    if cand.exists():
        return str(cand)

    # try story folder
    cand2 = story_folder / p
    if cand2.exists():
        return str(cand2)

    return str(p)  # آخر محاولة


def get_fonts_from_info(story_folder: Path) -> Tuple[str, str, str, str]:
    """
    returns: (first_en, rest_en, first_ar, rest_ar) as abs paths (may be "")
    """
    (
        en_story_name,
        ar_story_name,
        resolution_slides,
        first_slide_font,
        rest_slides_font,
        ar_first_slide_font,
        ar_rest_slides_font,
    ) = read_info_file(str(story_folder))

    first_en = _abs_path(first_slide_font, story_folder)
    rest_en  = _abs_path(rest_slides_font, story_folder)
    first_ar = _abs_path(ar_first_slide_font, story_folder)
    rest_ar  = _abs_path(ar_rest_slides_font, story_folder)
    return first_en, rest_en, first_ar, rest_ar


# =========================================================
# HTML helpers: inject font stack + scale sizes + waw transparent
# =========================================================
def inject_font_family(html_text: str, font_family_stack: str | None) -> str:
    if not font_family_stack:
        return html_text

    # remove existing font-family occurrences
    html_text = re.sub(r"font-family:\s*[^;'\"]+[;\"]", "", html_text)
    html_text = re.sub(r"font-family:\s*'[^']+'[;\"]?", "", html_text)
    html_text = re.sub(r'font-family:\s*"[^"]+"[;\"]?', "", html_text)

    def add_font_to_style(match):
        style_content = match.group(1)
        new_style = f"font-family: '{font_family_stack}' !important; " + style_content
        return f'style="{new_style}"'

    html_text = re.sub(r'style="([^"]*)"', add_font_to_style, html_text)

    base_style = f"font-family: '{font_family_stack}' !important;"
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

    def repl(m):
        original = float(m.group(1))
        unit = m.group(2) if m.group(2) else "pt"
        new_size = max(1, int(original * gf))
        return f"font-size:{new_size}{unit}"

    return re.sub(r"font-size:(\d+(?:\.\d+)?)(pt|px)?", repl, html_text)


def make_waw_transparent(html_text: str) -> str:
    # نفس منطقك
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


# =========================================================
# Renderer: HTML -> QImage (Stable, with shadow)
# =========================================================
def render_html_label(html: str, w: int, h: int) -> QImage:
    doc = QTextDocument()
    doc.setHtml(html)
    doc.setTextWidth(max(1, int(w)))

    item = QGraphicsTextItem()
    item.setDocument(doc)

    if ENABLE_TEXT_SHADOW:
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(int(SHADOW_BLUR_RADIUS))
        eff.setColor(QColor(*SHADOW_COLOR))
        eff.setOffset(int(SHADOW_OFFSET_X), int(SHADOW_OFFSET_Y))
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


def overlay_text_on_images_html(
    images_dict: Dict[str, np.ndarray],
    text_data: Dict[str, List[Dict]],
    font_stack_first: Optional[str],
    font_stack_rest: Optional[str],
) -> Dict[str, np.ndarray]:

    # Qt app (single process stable)
    app_qt = QApplication.instance()
    if app_qt is None:
        app_qt = QApplication([])

    out = {}
    total = len(images_dict)
    for idx, (img_name, base_cv) in enumerate(images_dict.items()):
        if DEBUG:
            logger.info("Render: %s (%d/%d)", img_name, idx + 1, total)
        if img_name not in text_data:
            out[img_name] = base_cv
            continue

        base_h, base_w = base_cv.shape[:2]
        rgb = cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB)
        base_q = QImage(rgb.data, base_w, base_h, 3 * base_w, QImage.Format_RGB888)

        out_img = QImage(base_w, base_h, QImage.Format_ARGB32_Premultiplied)
        out_img.fill(Qt.transparent)

        painter = QPainter(out_img)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.drawImage(0, 0, base_q)

        # choose font stack
        font_stack = font_stack_first if idx == 0 else font_stack_rest

        for element in text_data[img_name]:
            html = element.get("html", "") or ""
            x = int(element.get("x", 0) or 0)
            y = int(element.get("y", 0) or 0)
            w = int(element.get("width", 400) or 400)
            h = int(element.get("height", 200) or 200)
            gf = float(element.get("global_font", 0) or 0)

            # font + scale + waw
            if font_stack:
                html = inject_font_family(html, font_stack)
            if gf != 0:
                html = scale_font_sizes(html, gf)
            html = make_waw_transparent(html)

            if DEBUG:
                print(f"[Render] {img_name} rect=({x},{y},{w},{h}) gf={gf}")

            label_img = render_html_label(html, max(1, w), max(1, h))
            painter.drawImage(int(x), int(y), label_img)

        painter.end()

        # QImage -> cv2 BGR
        out_img = out_img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        w2, h2 = out_img.width(), out_img.height()
        bpl = out_img.bytesPerLine()
        raw = out_img.bits().tobytes()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h2, bpl // 4, 4))
        arr = arr[:, :w2, :]
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        out[img_name] = bgr

    return out


# =========================================================
# Endpoint
# =========================================================
@app.post("/generate-story/pdf")
async def generate_story_pdf(
    language: str = Form(...),
    user_name: str = Form(...),
    images_folder: str = Form(...),
):
    try:
        t0 = time.perf_counter()
        root = Path(images_folder)

        images_dir = _pick_images_dir(root)
        if not images_dir:
            return JSONResponse(
                {"status": "error", "message": f"No images found under: {images_folder}"},
                status_code=400,
            )
        logger.info("PDF: images_dir=%s (%.2fs)", images_dir, time.perf_counter() - t0)

        story_folder = _find_story_folder(images_dir)
        if not story_folder:
            return JSONResponse(
                {"status": "error", "message": f"Story folder not detected for images_dir: {images_dir}"},
                status_code=400,
            )
        logger.info("PDF: story_folder=%s (%.2fs)", story_folder, time.perf_counter() - t0)

        translations_dir = story_folder / "Translations"
        if not translations_dir.exists():
            return JSONResponse(
                {"status": "error", "message": f"Translations not found: {translations_dir}"},
                status_code=400,
            )

        # choose text file + placeholder
        lang = (language or "").lower().strip()
        if lang == "en":
            text_file = translations_dir / "en_text_data.txt"
            placeholder = "[*NAME*]"
        elif lang == "ar":
            # أول ملف ar_*.txt
            ar_files = sorted(translations_dir.glob("ar_*.txt"))
            text_file = ar_files[0] if ar_files else (translations_dir / "ar_text_data.txt")
            placeholder = "[*الاسم*]"
        else:
            return JSONResponse({"status": "error", "message": "language must be en or ar"}, status_code=400)

        if not text_file.exists():
            return JSONResponse(
                {"status": "error", "message": f"Text file not found: {text_file}"},
                status_code=400,
            )
        logger.info("PDF: text_file=%s (%.2fs)", text_file, time.perf_counter() - t0)

        # load slides json
        try:
            slides_json = json.loads(text_file.read_text(encoding="utf-8"))
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Invalid JSON in text file: {e}"}, status_code=400)
        logger.info("PDF: text_json_loaded (%.2fs)", time.perf_counter() - t0)

        # build text_data dict + replace NAME tokens
        text_data = {}
        for slide_key, blocks in slides_json.items():
            if not isinstance(blocks, list):
                continue
            new_blocks = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bb = dict(b)
                bb["html"] = replace_name_tokens(bb.get("html", ""), user_name)
                new_blocks.append(bb)
            text_data[str(slide_key)] = new_blocks

        # load images as dict keyed by filename
        image_files = sorted([p for p in images_dir.iterdir() if _is_image_file(p)])
        if not image_files:
            return JSONResponse({"status": "error", "message": f"No images in: {images_dir}"}, status_code=400)
        logger.info("PDF: image_files=%d (%.2fs)", len(image_files), time.perf_counter() - t0)

        images_dict = {}
        for p in image_files:
            img = cv2.imread(str(p))
            if img is not None:
                images_dict[p.name] = img

        if not images_dict:
            return JSONResponse({"status": "error", "message": "No readable images"}, status_code=400)
        logger.info("PDF: images_loaded=%d (%.2fs)", len(images_dict), time.perf_counter() - t0)

        # keep output sizes aligned with original story scenes (avoid unwanted resize)
        resized_count = 0
        for img_name in list(images_dict.keys()):
            base_name = _base_slide_name_from_filename(img_name)
            src_scene = None
            for ext in ALLOWED_EXT:
                cand = story_folder / "api_images" / f"{base_name}{ext}"
                if cand.exists():
                    src_scene = cand
                    break
                cand = story_folder / "normal_images" / f"{base_name}{ext}"
                if cand.exists():
                    src_scene = cand
                    break
            if not src_scene:
                continue
            src_img = cv2.imread(str(src_scene))
            if src_img is None:
                continue
            src_h, src_w = src_img.shape[:2]
            cur_h, cur_w = images_dict[img_name].shape[:2]
            if (cur_w, cur_h) != (src_w, src_h):
                images_dict[img_name] = cv2.resize(
                    images_dict[img_name],
                    (src_w, src_h),
                    interpolation=cv2.INTER_CUBIC,
                )
                resized_count += 1
        if resized_count:
            logger.info("PDF: resized_to_original=%d (%.2fs)", resized_count, time.perf_counter() - t0)

        # ✅ KEY FIX: map slide_01 -> slide_01.png (or .jpg)
        text_data = normalize_text_keys_to_images(text_data, images_dict)
        logger.info("PDF: text_keys_normalized (%.2fs)", time.perf_counter() - t0)

        # fonts from info.txt + build font stacks
        first_en, rest_en, first_ar, rest_ar = get_fonts_from_info(story_folder)

        # register fonts into Qt and get actual family names
        app_qt = QApplication.instance()
        if app_qt is None:
            app_qt = QApplication([])

        def reg_font(path: str) -> str:
            if not path or not Path(path).exists():
                return ""
            fid = QFontDatabase.addApplicationFont(path)
            if fid == -1:
                return ""
            fams = QFontDatabase.applicationFontFamilies(fid)
            return fams[0] if fams else ""

        logger.info("PDF: fonts_resolve_start (%.2fs)", time.perf_counter() - t0)
        fam_first_en = reg_font(first_en)
        fam_rest_en  = reg_font(rest_en)
        fam_first_ar = reg_font(first_ar)
        fam_rest_ar  = reg_font(rest_ar)
        logger.info("PDF: fonts_registered (%.2fs)", time.perf_counter() - t0)

        # ✅ fallback stack: primary, fallback (supports Arabic even in EN story)
        # first slide
        if fam_first_en and fam_first_ar:
            stack_first = f"{fam_first_en}', '{fam_first_ar}"
        else:
            stack_first = fam_first_en or fam_first_ar or None

        # rest slides
        if fam_rest_en and fam_rest_ar:
            stack_rest = f"{fam_rest_en}', '{fam_rest_ar}"
        else:
            stack_rest = fam_rest_en or fam_rest_ar or None

        if DEBUG:
            print("[Fonts]")
            print(" first_en:", first_en, "->", fam_first_en)
            print(" rest_en :", rest_en,  "->", fam_rest_en)
            print(" first_ar:", first_ar, "->", fam_first_ar)
            print(" rest_ar :", rest_ar,  "->", fam_rest_ar)
            print(" stack_first:", stack_first)
            print(" stack_rest :", stack_rest)

        # ✅ Render HTML (single process stable)
        logger.info("PDF: text_overlay_start (%.2fs)", time.perf_counter() - t0)
        processed = overlay_text_on_images_html(
            images_dict=images_dict,
            text_data=text_data,
            font_stack_first=stack_first,
            font_stack_rest=stack_rest,
        )
        logger.info("PDF: text_overlay_done (%.2fs)", time.perf_counter() - t0)

        # normalize all pages to the same width (preserve aspect)
        max_w = 0
        for img in processed.values():
            h, w = img.shape[:2]
            max_w = max(max_w, w)

        if max_w:
            normalized = {}
            resized_pages = 0
            for img_name, img in processed.items():
                h, w = img.shape[:2]
                if w != max_w:
                    new_h = max(1, int(h * (max_w / float(w))))
                    normalized[img_name] = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_CUBIC)
                    resized_pages += 1
                else:
                    normalized[img_name] = img
            processed = normalized
            if resized_pages:
                logger.info("PDF: pages_width_normalized=%d to width=%d (%.2fs)", resized_pages, max_w, time.perf_counter() - t0)

        # export PDF
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        pdf_name = f"story_{user_name}_{language}.pdf"
        pdf_path = str(Path(RESULT_FOLDER) / pdf_name)

        ordered_names = sorted(processed.keys())
        ok = create_pdf_from_images([processed[n] for n in ordered_names], pdf_path)
        if not ok:
            return JSONResponse({"status": "error", "message": "PDF generation failed"}, status_code=500)
        logger.info("PDF: pdf_written=%s (%.2fs)", pdf_path, time.perf_counter() - t0)

        return JSONResponse(
            {
                "status": "success",
                "pdf_path": pdf_path,
                "pdf_name": pdf_name,
                "story_folder": str(story_folder),
                "images_dir_used": str(images_dir),
                "text_file_used": str(text_file),
                "slides": len(ordered_names),
            }
        )

    except Exception as e:
        logger.exception("Endpoint failed")
        return JSONResponse({"status": "error", "message": f"Unexpected error: {e}"}, status_code=500)

# --------------------------------------------------
# ENDPOINT 2 → Images only
# --------------------------------------------------
@app.post("/head-swap")
async def head_swap_only(
    gender: str = Form(...),
    story_code: str = Form(...),
    character_image: UploadFile | None = File(None),
):
    gender_folder = get_gender_folder(gender)
    story_folder = get_story_folder(gender, story_code)

    if character_image:
        character_path = save_uploaded_file(character_image, TEMP_UPLOADS)
    else:
        character_path, _ = get_default_character(gender_folder)
        if not character_path:
            raise HTTPException(400, "No character image found")

    # 🔹 new automatic batch version, returns list
    processed_slides = process_head_swap(
        clean_images_folder=None,
        character_image_path=character_path,
        character_name=os.path.splitext(os.path.basename(character_path))[0],
        story_folder=story_folder,
    )

    image_info_list = []
    for slide in processed_slides:  # loop over list
        img = slide["image"]  # PIL Image
        img_filename = f"{story_code}_{slide['name']}.png"
        temp_path = os.path.join(TEMP_HEAD_SWAP, img_filename)
        # img.save(temp_path)

        image_info_list.append({
            "name": slide["name"],
            "path": slide["path"]  # <-- use this directly
        })

    images_folder = None
    if image_info_list:
        images_folder = os.path.dirname(image_info_list[0]["path"])

    return JSONResponse({
        "status": "success",
        "slides_count": len(image_info_list),
        "images_folder": images_folder,
        "images": image_info_list
    })

# --------------------------------------------------
# ENDPOINT 3 → one slide regeneration
# --------------------------------------------------
@app.post("/regenerate-slide")
async def regenerate_slide(
    slide_path: str = Form(...),  # path to the existing slide to regenerate
    source_scene_path: Optional[str] = Form(None),  # original scene image (optional)
    face_image: UploadFile | None = File(None),  # optional new face for head-swap
):
    if not os.path.exists(slide_path):
        raise HTTPException(400, f"Slide not found: {slide_path}")

    # Get folder and filename info
    folder, filename = os.path.split(slide_path)
    name_no_ext, ext = os.path.splitext(filename)

    # Detect base slide name, e.g., slide_02 from slide_02_try1
    if "_try" in name_no_ext:
        base_name = name_no_ext.split("_try")[0]
    else:
        base_name = name_no_ext

    # Find next available try number
    existing_files = [f for f in os.listdir(folder) if f.startswith(base_name)]
    tries = [0]
    for f in existing_files:
        if "_try" in f:
            try_num = int(f.split("_try")[-1].split(".")[0])
            tries.append(try_num)
    next_try = max(tries) + 1

    # New path for regenerated slide
    new_try_path = os.path.join(folder, f"{base_name}_try{next_try}.jpg")

    if face_image:
        face_path = save_uploaded_file(face_image, TEMP_UPLOADS)
        if source_scene_path:
            scene_path = Path(source_scene_path)
        else:
            scene_path = _find_source_scene_for_slide(slide_path)

        if not scene_path or not scene_path.exists():
            raise HTTPException(400, "Source scene image not found for regeneration")

        _set_single_attempt_env(next_try)
        try:
            preview_path = perform_head_swap(
                target_image_path=str(scene_path),
                face_image_path=face_path,
                output_filename=new_try_path,
                face_url_cached=None,
            )
        finally:
            _clear_single_attempt_env()

        if preview_path and os.path.exists(preview_path):
            new_try_path = preview_path
        elif not os.path.exists(new_try_path):
            raise HTTPException(500, "Head-swap regeneration failed to produce an image")
    else:
        # Copy original slide as a base (fallback)
        shutil.copyfile(slide_path, new_try_path)

    return JSONResponse({
        "status": "success",
        "original_slide": slide_path,
        "new_try_path": new_try_path,
        "used_source_scene": str(source_scene_path) if source_scene_path else None,
        "used_face_image": bool(face_image),
    })
