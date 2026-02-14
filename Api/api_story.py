# -*- coding: utf-8 -*-
"""
🎭 Stories Production API (Windows-friendly + Stable HTML→PDF)
✅ /file serves local files by absolute path (SAFE: under BASE_DIR only)
✅ /delete-file deletes a file only under Head_swap (safe)
✅ /head-swap saves images to Stories/<Boys|Girls>/<code>/Head_swap/<session>/ and returns urls
✅ /head-swap/list lists session images with encoded urls
✅ /regenerate-slide regenerates ONE slide and returns OLD + NEW (urls + paths)
✅ /generate-story/pdf writes HTML text onto OpenCV images (Qt) then exports PDF

✅ Fixes / Guarantees:
- Qt QApplication SINGLETON (one instance for whole process)
- PDF pipeline uses OpenCV numpy arrays end-to-end (no .shape errors)
- Text rendering uses HTML Qt renderer (shadow supported)
- Safe path checks (must be under BASE_DIR)
"""

import os
import sys
import json
import time
import re
import shutil
import logging
import mimetypes
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QGraphicsScene, QGraphicsTextItem
from PySide6.QtGui import QPainter, QFontDatabase, QColor, QImage, QTextDocument
from PySide6.QtCore import Qt, QRectF

from Codes.config import (
    RESULT_FOLDER,
    BASE_DIR,
    ENABLE_TEXT_SHADOW,
    SHADOW_BLUR_RADIUS,
    SHADOW_COLOR,
    SHADOW_OFFSET_X,
    SHADOW_OFFSET_Y,
)
from Codes.utils import read_info_file
from Codes.image_processor import process_head_swap  # head-swap batch (your existing)
from Codes.api_segmiod import perform_head_swap      # single slide regenerate (your existing)
from Codes.pdf_generator import create_pdf_from_images  # expects list[cv2 BGR] + output path


# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("StoriesAPI")


# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="Stories Production API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # keep as you had
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR_PATH = Path(BASE_DIR).resolve()
RESULT_FOLDER_PATH = Path(RESULT_FOLDER).resolve()
RESULT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

TEMP_UPLOADS = (BASE_DIR_PATH / "TempUploads").resolve()
TEMP_HEAD_SWAP = (BASE_DIR_PATH / "TempHeadSwaps").resolve()
TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)
TEMP_HEAD_SWAP.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
FORCE_HTTPS_URLS = os.environ.get("FORCE_HTTPS_URLS", "1").strip().lower() in ("1", "true", "yes")


# --------------------------------------------------
# Qt (PySide6) - MUST be single instance
# --------------------------------------------------
QT_APP = QApplication.instance()
if QT_APP is None:
    QT_APP = QApplication([])


# --------------------------------------------------
# Helpers: safety + io
# --------------------------------------------------
def _safe_quote_path(p: str) -> str:
    # keep : / \ unescaped (Windows paths)
    return quote(p, safe=":/\\")  # NOTE: still encodes spaces etc.

def _resolve_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (BASE_DIR_PATH / p).resolve()
    else:
        p = p.resolve()
    return p

def _public_base_url(request: Request) -> str:
    env_base = (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or "").strip()
    if env_base:
        if FORCE_HTTPS_URLS and env_base.lower().startswith("http://"):
            env_base = "https://" + env_base[7:]
        return env_base.rstrip("/")

    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc

    if FORCE_HTTPS_URLS:
        proto = "https"

    return f"{proto}://{host}"

def _file_url(request: Request, file_path: str) -> str:
    return f"{_public_base_url(request)}/file?path={_safe_quote_path(file_path)}"


def _assert_under_base_dir(p: Path) -> None:
    p = p.resolve()
    if BASE_DIR_PATH != p and BASE_DIR_PATH not in p.parents:
        raise HTTPException(status_code=403, detail="Path not allowed")


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXT


def _is_try_file(p: Path) -> bool:
    return "_try" in p.stem.lower()


def _json_error(status_code: int, msg: str):
    return JSONResponse({"status": "error", "message": msg}, status_code=status_code)


def build_pdf_filename(pdf_name: str, language: str, user_name: str) -> str:
    language = (language or "").lower().strip()
    user_name = (user_name or "").strip()

    if language == "en":
        name = (
            (pdf_name or "Story_EN")
            .replace("Name", user_name)
            .replace("name", user_name)
            .replace("NAME", user_name.upper())
        )
    else:
        name = (
            (pdf_name or "Story_AR")
            .replace("الاسم", user_name)
            .replace("اسم", user_name)
        )

    # Windows invalid filename chars
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")

    return f"{name}.pdf"


def save_uploaded_file(upload: UploadFile, base_folder: Path) -> str:
    ext = Path(upload.filename).suffix
    file_name_no_ext = Path(upload.filename).stem

    folder_path = (base_folder / file_name_no_ext).resolve()
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = (folder_path / f"{file_name_no_ext}{ext}").resolve()
    with open(file_path, "wb") as f:
        f.write(upload.file.read())

    return str(file_path)


def get_gender_folder(gender: str) -> str:
    gender = (gender or "").lower().strip()
    if gender == "male":
        return str((BASE_DIR_PATH / "Characters" / "Boys").resolve())
    if gender == "female":
        return str((BASE_DIR_PATH / "Characters" / "Girls").resolve())
    raise HTTPException(400, "Invalid gender")


def get_story_folder(gender: str, code: str) -> str:
    gender = (gender or "").lower().strip()
    base = "Boys" if gender == "male" else "Girls"
    path = (BASE_DIR_PATH / "Stories" / base / code).resolve()
    if not path.exists():
        raise HTTPException(404, f"Story not found: {code}")
    return str(path)


def get_default_character(folder: str):
    folderp = Path(folder)
    if not folderp.exists():
        return None, None
    for f in folderp.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
            return str(f.resolve()), f.stem
    return None, None


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
        if not folder.exists():
            continue
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
# /file (safe)
# --------------------------------------------------
@app.get("/file")
def get_file(path: str = Query(..., description="Absolute or relative file path under server base dir")):
    p = _resolve_input_path(path)
    _assert_under_base_dir(p)

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=media_type or "application/octet-stream")


# --------------------------------------------------
# /delete-file (safe: only under Head_swap)
# --------------------------------------------------
@app.post("/delete-file")
async def delete_file(path: str = Form(...)):
    p = _resolve_input_path(path)
    _assert_under_base_dir(p)

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if "Head_swap" not in str(p):
        raise HTTPException(status_code=403, detail="Delete not allowed here")

    p.unlink()
    return JSONResponse({"status": "success", "deleted": str(p)})


# --------------------------------------------------
# /head-swap
# --------------------------------------------------

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image
import hashlib, json, shutil

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_meta(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_meta(meta_path: Path, data: dict) -> None:
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _extract_story_context_from_path(p: Path) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
    parts = p.parts
    for i, part in enumerate(parts):
        if part.lower() == "stories" and i + 2 < len(parts):
            gender_folder = parts[i + 1]
            story_code = parts[i + 2]
            story_folder = Path(*parts[: i + 3])
            return gender_folder, story_code, story_folder
    return None, None, None

def _find_session_folder(slide_path: Path) -> Optional[Path]:
    parts = slide_path.parts
    for i, part in enumerate(parts):
        if part.lower() == "head_swap" and i + 1 < len(parts) - 1:
            return Path(*parts[: i + 2])
    return None

def _resolve_face_image_for_slide(slide_path: Path) -> Optional[str]:
    session_folder = _find_session_folder(slide_path)
    meta: Dict[str, Any] = {}

    if session_folder:
        meta_path = session_folder / "meta.json"
        if meta_path.exists():
            meta = _read_meta(meta_path) or {}

        meta_face = meta.get("character_path") or meta.get("face_path") or meta.get("character_image")
        if meta_face:
            try:
                face_p = _resolve_input_path(str(meta_face))
                _assert_under_base_dir(face_p)
                if face_p.exists():
                    return str(face_p)
            except Exception:
                pass

    session_name = session_folder.name if session_folder else ""
    gender_folder, _, _ = _extract_story_context_from_path(slide_path)

    if session_name.startswith("default_") and gender_folder:
        target_stem = session_name.replace("default_", "", 1)
        char_dir = (BASE_DIR_PATH / "Characters" / gender_folder).resolve()
        for ext in ALLOWED_EXT:
            cand = char_dir / f"{target_stem}{ext}"
            if cand.exists():
                return str(cand.resolve())
        # fallback: first available character
        default_path, _ = get_default_character(str(char_dir))
        if default_path:
            return default_path

    if session_name:
        temp_dir = (TEMP_UPLOADS / session_name).resolve()
        if temp_dir.exists() and temp_dir.is_dir():
            for p in temp_dir.iterdir():
                if _is_image_file(p):
                    return str(p.resolve())

    gender = (meta.get("gender") or "").lower().strip()
    if gender in ("male", "female"):
        gender_dir = "Boys" if gender == "male" else "Girls"
        char_dir = (BASE_DIR_PATH / "Characters" / gender_dir).resolve()
        default_path, _ = get_default_character(str(char_dir))
        if default_path:
            return default_path

    return None

@app.post("/head-swap")
async def head_swap_only(
    request: Request,
    gender: str = Form(...),
    story_code: str = Form(...),
    character_image: UploadFile | None = File(None),
):
    gender = (gender or "").lower().strip()
    story_code = (story_code or "").strip()

    gender_folder = get_gender_folder(gender)
    story_folder = get_story_folder(gender, story_code)

    # ---------- choose character ----------
    if character_image:
        character_path = save_uploaded_file(character_image, TEMP_UPLOADS)
        face_hash = _file_sha1(character_path)
        character_name = Path(character_path).stem
        # use original filename (no hash) for session folder
        session = character_name
    else:
        character_path, character_name = get_default_character(gender_folder)
        if not character_path:
            raise HTTPException(400, "No character image found")
        # default character: stable session by filename
        face_hash = _file_sha1(character_path)
        session = f"default_{Path(character_path).stem}"

    base_folder = "Boys" if gender == "male" else "Girls"
    out_dir = (BASE_DIR_PATH / "Stories" / base_folder / story_code / "Head_swap" / session).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    expected_meta = {
        "gender": gender,
        "story_code": story_code,
        "character_name": character_name,
        "face_hash": face_hash,
        "character_path": character_path,
    }

    # ---------- cache check ----------
    existing_imgs = [p for p in sorted(out_dir.iterdir()) if _is_image_file(p) and not _is_try_file(p)]
    existing_meta = _read_meta(meta_path)

    if existing_imgs and existing_meta.get("face_hash") == face_hash:
        images = [{
            "name": p.stem,
            "path": str(p.resolve()),
            "url": _file_url(request, str(p.resolve())),
        } for p in existing_imgs]

        return JSONResponse({
            "status": "success",
            "session": session,
            "ImageFolder": str(out_dir),
            "image_folder": str(out_dir),
            "count": len(images),
            "images": images,
            "cached": True,
        })

    # ---------- regenerate ----------
    # (اختياري) لو فيه صور قديمة من session غلط، امسحها
    for p in existing_imgs:
        try: p.unlink()
        except: pass

    processed_slides = process_head_swap(
        clean_images_folder=None,
        character_image_path=character_path,
        character_name=character_name,
        story_folder=story_folder,
    )

    if not isinstance(processed_slides, list):
        raise HTTPException(status_code=500, detail="process_head_swap did not return a list.")

    images = []
    for slide in processed_slides:
        if not isinstance(slide, dict):
            continue

        slide_name = slide.get("name") or "slide_unknown"
        pil_img = slide.get("image")
        existing_path = slide.get("path")
        saved_path = None

        if isinstance(pil_img, Image.Image):
            saved_path = str((out_dir / f"{slide_name}.jpg").resolve())
            pil_img.save(saved_path, format="JPEG", quality=95)

        elif existing_path and Path(existing_path).exists():
            src = Path(existing_path).resolve()
            dst = (out_dir / src.name).resolve()
            if src != dst:
                shutil.copyfile(str(src), str(dst))
            saved_path = str(dst)

        if saved_path and not _is_try_file(Path(saved_path)):
            images.append({
                "name": Path(saved_path).stem,
                "path": saved_path,
                "url": _file_url(request, saved_path),
            })

    # write meta (after success)
    _write_meta(meta_path, expected_meta)

    return JSONResponse({
        "status": "success",
        "session": session,
        "ImageFolder": str(out_dir),
        "image_folder": str(out_dir),
        "count": len(images),
        "images": images,
        "cached": False
    })

# --------------------------------------------------
# /head-swap/list
# --------------------------------------------------
@app.get("/head-swap/list")
def list_headswap_images(
    request: Request,
    gender: str = Query(...),
    story_code: str = Query(...),
    session: str = Query(...),
):
    gender = (gender or "").lower().strip()
    base = "Boys" if gender == "male" else "Girls"

    folder = (BASE_DIR_PATH / "Stories" / base / story_code / "Head_swap" / session).resolve()
    _assert_under_base_dir(folder)

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(404, f"Folder not found: {folder}")

    images = []
    for p in sorted(folder.iterdir()):
        if _is_image_file(p) and not _is_try_file(p):
            pp = p.resolve()
            images.append(
                {
                    "name": pp.stem,
                    "path": str(pp),
                    "url": _file_url(request, str(pp)),
                }
            )

    return {"status": "success", "count": len(images), "images": images}


# --------------------------------------------------
# /regenerate-slide
# --------------------------------------------------
@app.post("/regenerate-slide")
async def regenerate_slide(
    request: Request,
    path: Optional[str] = Form(None),
    slide_path: Optional[str] = Form(None),
):
    slide_path = (path or slide_path or "").strip()
    if not slide_path:
        raise HTTPException(400, "path is required")

    client_host = request.client.host if request.client else "unknown"
    logger.info(
        "Regenerate request: %s %s client=%s slide_path=%s",
        request.method,
        request.url.path,
        client_host,
        slide_path,
    )

    slide_path_p = _resolve_input_path(slide_path)
    _assert_under_base_dir(slide_path_p)

    if not slide_path_p.exists():
        raise HTTPException(400, f"Slide not found: {slide_path}")

    folder = slide_path_p.parent
    name_no_ext = slide_path_p.stem
    base_name = name_no_ext.split("_try")[0] if "_try" in name_no_ext else name_no_ext

    # choose target output folder (default: same folder)
    target_folder = folder

    # next try index (based on target folder)
    existing_files = [f.name for f in target_folder.iterdir() if f.is_file() and f.name.startswith(base_name)]
    tries = [0]
    for f in existing_files:
        if "_try" in f:
            try:
                tries.append(int(f.split("_try")[-1].split(".")[0]))
            except Exception:
                pass
    next_try = max(tries) + 1

    new_try_path = (target_folder / f"{base_name}_try{next_try}.jpg").resolve()

    face_path = _resolve_face_image_for_slide(slide_path_p)
    if not face_path:
        raise HTTPException(400, "Face image not found for this slide/session. Run /head-swap first.")

    scene_path = _find_source_scene_for_slide(str(slide_path_p))
    if not scene_path or not scene_path.exists():
        raise HTTPException(400, "Source scene image not found for regeneration")

    _assert_under_base_dir(scene_path)

    _set_single_attempt_env(next_try)
    try:
        preview_path = perform_head_swap(
            target_image_path=str(scene_path),
            face_image_path=face_path,
            output_filename=str(new_try_path),
            face_url_cached=None,
        )
    finally:
        _clear_single_attempt_env()

    if preview_path and Path(preview_path).exists():
        new_try_path = Path(preview_path).resolve()
    elif not new_try_path.exists():
        raise HTTPException(500, "Head-swap regeneration failed")

    old_path = str(slide_path_p)
    new_path = str(new_try_path)

    # keep only the chosen try (new_path); remove other _try files for same base
    try:
        for f in target_folder.iterdir():
            if f.is_file() and _is_image_file(f):
                stem = f.stem
                if stem.startswith(base_name) and "_try" in stem and f.resolve() != Path(new_path).resolve():
                    try:
                        f.unlink()
                    except Exception:
                        pass
    except Exception:
        pass

    logger.info("Regenerate success: old=%s new=%s try=%s", old_path, new_path, next_try)

    return JSONResponse(
        {
            "status": "success",
            "base_name": base_name,
            "try_index": next_try,
            "path": new_path,
            "url": _file_url(request, new_path),
            "old_path": old_path,
            "new_path": new_path,
            "old_url": _file_url(request, old_path),
            "new_url": _file_url(request, new_path),
        }
    )


# --------------------------------------------------
# Text helpers (HTML name replacement + RTL/BiDi)
# --------------------------------------------------
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def wrap_name_for_html(name: str) -> str:
    name = name or ""
    # Arabic name inside English context: wrap RTL + unicode-bidi
    if _AR_RE.search(name):
        return f"<span dir='rtl' style='unicode-bidi:plaintext;'>{name}</span>"
    return name


def replace_name_tokens(html: str, user_name: str) -> str:
    u = wrap_name_for_html(user_name)
    tokens = ["[*NAME*]", "[*name*]", "NAME", "Name", "name", "[*الاسم*]", "الاسم", "اسم"]
    out = html or ""
    for t in tokens:
        out = out.replace(t, u)
    return out


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


# --------------------------------------------------
# Fonts helpers (info.txt -> abs path -> Qt family)
# --------------------------------------------------
def _abs_path(p: Optional[str], story_folder: Path) -> str:
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp)

    # try BASE_DIR
    cand = BASE_DIR_PATH / p
    if cand.exists():
        return str(cand)

    # try story folder
    cand2 = story_folder / p
    if cand2.exists():
        return str(cand2)

    return str(p)


def get_fonts_from_info(story_folder: Path) -> Tuple[str, str, str, str, str, str]:
    """
    returns: (en_name, ar_name, first_en, rest_en, first_ar, rest_ar) as abs paths (may be "")
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
    rest_en = _abs_path(rest_slides_font, story_folder)
    first_ar = _abs_path(ar_first_slide_font, story_folder)
    rest_ar = _abs_path(ar_rest_slides_font, story_folder)

    return (en_story_name or "Story_EN", ar_story_name or "Story_AR", first_en, rest_en, first_ar, rest_ar)


def reg_font_family(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    fid = QFontDatabase.addApplicationFont(str(p))
    if fid == -1:
        return ""
    fams = QFontDatabase.applicationFontFamilies(fid)
    return fams[0] if fams else ""


# --------------------------------------------------
# HTML helpers: inject font stack + scale sizes + waw transparent
# --------------------------------------------------
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
    # same logic you had
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


# --------------------------------------------------
# Renderer: HTML -> QImage (Stable, with shadow)
# --------------------------------------------------
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
    debug: bool = False,
) -> Dict[str, np.ndarray]:
    # Qt app (global singleton)
    _ = QT_APP

    out: Dict[str, np.ndarray] = {}
    total = len(images_dict)

    for idx, (img_name, base_cv) in enumerate(images_dict.items()):
        if debug:
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

            if font_stack:
                html = inject_font_family(html, font_stack)
            if gf != 0:
                html = scale_font_sizes(html, gf)
            html = make_waw_transparent(html)

            if debug:
                logger.info("[Render] %s rect=(%d,%d,%d,%d) gf=%s", img_name, x, y, w, h, gf)

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


# --------------------------------------------------
# Endpoint: /generate-story/pdf
# --------------------------------------------------
DEBUG = os.environ.get("TEXT_DEBUG", "0").strip().lower() in ("1", "true", "yes")


def _pick_images_dir(root: Path) -> Optional[Path]:
    """
    root may be:
    - story_folder
    - or images folder directly
    """
    if root.exists() and root.is_dir() and any(_is_image_file(x) for x in root.iterdir()):
        return root

    candidates = [root / "api_images", root / "normal_images", root / "Images", root / "Clean_Images"]
    for c in candidates:
        if c.exists() and c.is_dir() and any(_is_image_file(x) for x in c.iterdir()):
            return c

    # fallback: first subfolder containing images
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
    Find real story_folder (has Translations)
    """
    if images_dir.name.lower() in ("api_images", "normal_images", "images", "clean_images"):
        story = images_dir.parent
        if (story / "Translations").exists():
            return story

    cur = images_dir
    for _ in range(10):
        if (cur / "Translations").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


@app.post("/generate-story/pdf")
async def generate_story_pdf(
    request: Request,
    language: str = Form(...),
    user_name: str = Form(...),
    images_folder: Optional[str] = Form(None),
    ImageFolder: Optional[str] = Form(None),
    image_folder: Optional[str] = Form(None),
):
    try:
        t0 = time.perf_counter()
        lang = (language or "").lower().strip()
        user_name = (user_name or "").strip()

        images_folder = (images_folder or ImageFolder or image_folder or "").strip()
        if not images_folder:
            return _json_error(400, "images_folder is required")

        root = _resolve_input_path(images_folder)
        _assert_under_base_dir(root)

        images_dir = _pick_images_dir(root)
        if not images_dir:
            return _json_error(400, f"No images found under: {images_folder}")
        logger.info("PDF: images_dir=%s (%.2fs)", images_dir, time.perf_counter() - t0)

        story_folder = _find_story_folder(images_dir)
        if not story_folder:
            return _json_error(400, f"Story folder not detected for images_dir: {images_dir}")
        logger.info("PDF: story_folder=%s (%.2fs)", story_folder, time.perf_counter() - t0)

        translations_dir = story_folder / "Translations"
        if not translations_dir.exists():
            return _json_error(400, f"Translations not found: {translations_dir}")

        # choose text file
        if lang == "en":
            text_file = translations_dir / "en_text_data.txt"
        elif lang == "ar":
            ar_files = sorted(translations_dir.glob("ar_*.txt"))
            text_file = ar_files[0] if ar_files else (translations_dir / "ar_text_data.txt")
        else:
            return _json_error(400, "language must be en or ar")

        if not text_file.exists():
            return _json_error(400, f"Text file not found: {text_file}")
        logger.info("PDF: text_file=%s (%.2fs)", text_file, time.perf_counter() - t0)

        # load slides json
        try:
            slides_json = json.loads(text_file.read_text(encoding="utf-8"))
        except Exception as e:
            return _json_error(400, f"Invalid JSON in text file: {e}")

        # build text_data dict + replace NAME tokens
        text_data: Dict[str, List[Dict]] = {}
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
        image_files = sorted([p for p in images_dir.iterdir() if _is_image_file(p) and not _is_try_file(p)])
        if not image_files:
            return _json_error(400, f"No images in: {images_dir}")
        logger.info("PDF: image_files=%d (%.2fs)", len(image_files), time.perf_counter() - t0)

        images_dict: Dict[str, np.ndarray] = {}
        for p in image_files:
            img = cv2.imread(str(p))
            if img is not None:
                images_dict[p.name] = img

        if not images_dict:
            return _json_error(400, "No readable images (cv2.imread returned None).")
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
                images_dict[img_name] = cv2.resize(images_dict[img_name], (src_w, src_h), interpolation=cv2.INTER_CUBIC)
                resized_count += 1

        if resized_count:
            logger.info("PDF: resized_to_original=%d (%.2fs)", resized_count, time.perf_counter() - t0)

        # map slide_01 -> slide_01.png/.jpg
        text_data = normalize_text_keys_to_images(text_data, images_dict)
        logger.info("PDF: text_keys_normalized (%.2fs)", time.perf_counter() - t0)

        # fonts from info.txt -> register to Qt -> build stacks
        en_story_name, ar_story_name, first_en, rest_en, first_ar, rest_ar = get_fonts_from_info(story_folder)

        fam_first_en = reg_font_family(first_en)
        fam_rest_en = reg_font_family(rest_en)
        fam_first_ar = reg_font_family(first_ar)
        fam_rest_ar = reg_font_family(rest_ar)

        # build font stacks (primary + fallback)
        if fam_first_en and fam_first_ar:
            stack_first = f"{fam_first_en}', '{fam_first_ar}"
        else:
            stack_first = fam_first_en or fam_first_ar or None

        if fam_rest_en and fam_rest_ar:
            stack_rest = f"{fam_rest_en}', '{fam_rest_ar}"
        else:
            stack_rest = fam_rest_en or fam_rest_ar or None

        if DEBUG:
            logger.info("[Fonts] first_en=%s -> %s", first_en, fam_first_en)
            logger.info("[Fonts] rest_en =%s -> %s", rest_en, fam_rest_en)
            logger.info("[Fonts] first_ar=%s -> %s", first_ar, fam_first_ar)
            logger.info("[Fonts] rest_ar =%s -> %s", rest_ar, fam_rest_ar)
            logger.info("[Fonts] stack_first=%s", stack_first)
            logger.info("[Fonts] stack_rest =%s", stack_rest)

        # Render HTML -> processed images (cv2 BGR)
        logger.info("PDF: text_overlay_start (%.2fs)", time.perf_counter() - t0)
        processed = overlay_text_on_images_html(
            images_dict=images_dict,
            text_data=text_data,
            font_stack_first=stack_first,
            font_stack_rest=stack_rest,
            debug=DEBUG,
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
                logger.info(
                    "PDF: pages_width_normalized=%d to width=%d (%.2fs)",
                    resized_pages, max_w, time.perf_counter() - t0
                )

        # export PDF
        pdf_base_name = en_story_name if lang == "en" else ar_story_name
        safe_pdf_name = build_pdf_filename(pdf_base_name, lang, user_name)
        pdf_path = (RESULT_FOLDER_PATH / safe_pdf_name).resolve()
        _assert_under_base_dir(pdf_path)

        ordered_names = sorted(processed.keys())
        pages = [processed[n] for n in ordered_names]

        def _write_pdf():
            ok = create_pdf_from_images(pages, str(pdf_path))
            if not ok:
                raise RuntimeError("PDF generation failed")

        await run_in_threadpool(_write_pdf)

        logger.info("PDF: pdf_written=%s (%.2fs)", pdf_path, time.perf_counter() - t0)

        return JSONResponse(
            {
                "status": "success",
                "pdf_path": str(pdf_path),
                "pdf_name": safe_pdf_name,
                "pdf_url": _file_url(request, str(pdf_path)),
                "story_folder": str(story_folder),
                "images_dir_used": str(images_dir),
                "text_file_used": str(text_file),
                "slides": len(ordered_names),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF endpoint failed\n%s", traceback.format_exc())
        return JSONResponse({"status": "error", "message": f"Unexpected error: {e}"}, status_code=500)
