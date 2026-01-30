# -*- coding: utf-8 -*-
"""
🎭 Stories Production API
Pipeline updated: head-swap selection + PDF/image generation
"""

import os
import sys
import uuid
import logging
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PySide6.QtWidgets import QApplication
from PIL import Image

from Codes.config import RESULT_FOLDER, BASE_DIR, USE_PARALLEL_TEXT_PROCESSING
from Codes.utils import read_info_file
from Codes.text_handler import load_custom_fonts, read_text_data
from Codes.image_processor import (
    process_head_swap,
    apply_text_to_images,
    apply_resolution_to_images,
)
from Codes.pdf_generator import create_pdf_from_images

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

def save_uploaded_file(upload: UploadFile, folder: str) -> str:
    ext = os.path.splitext(upload.filename)[1]
    file_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(folder, file_name)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path

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
# ENDPOINT 1 → PDF
# --------------------------------------------------
@app.post("/generate-story/pdf")
async def generate_story_pdf(
    language: str = Form(...),
    gender: str = Form(...),
    story_code: str = Form(...),
    user_name: str = Form(...),
    character_image: UploadFile | None = File(None),
):
    character_path = None
    if character_image:
        character_path = save_uploaded_file(character_image, TEMP_UPLOADS)
    else:
        character_path, _ = get_default_character(get_gender_folder(gender))
        if not character_path:
            raise HTTPException(400, "No character image found")

    result = run_story_pipeline(language, gender, story_code, user_name, character_path)

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    pdf_path = os.path.join(RESULT_FOLDER, result["pdf_name"])
    if not create_pdf_from_images(result["images"], pdf_path):
        raise HTTPException(500, "PDF generation failed")

    return FileResponse(pdf_path, media_type="application/pdf", filename=result["pdf_name"])

# --------------------------------------------------
# ENDPOINT 2 → Images only
# --------------------------------------------------
@app.post("/generate-story/images")
async def generate_story_images(
    language: str = Form(...),
    gender: str = Form(...),
    story_code: str = Form(...),
    user_name: str = Form(...),
    character_image: UploadFile | None = File(None),
):
    character_path = None
    if character_image:
        character_path = save_uploaded_file(character_image, TEMP_UPLOADS)
    else:
        character_path, _ = get_default_character(get_gender_folder(gender))
        if not character_path:
            raise HTTPException(400, "No character image found")

    result = run_story_pipeline(language, gender, story_code, user_name, character_path)
    return JSONResponse({
        "status": "success",
        "slides_count": len(result["images"]),
        "images": result["images"],
    })

# --------------------------------------------------
# ENDPOINT 3 → Head swap only
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

    processed_images, _ = process_head_swap(
        clean_images_folder=None,
        character_image_path=character_path,
        character_name=os.path.splitext(os.path.basename(character_path))[0],
        story_folder=story_folder,
    )

    result_paths = []
    for idx, img in processed_images.items():
        temp_path = os.path.join(TEMP_HEAD_SWAP, f"{uuid.uuid4().hex}.png")
        img.save(temp_path)
        result_paths.append(temp_path)

    return JSONResponse({
        "status": "success",
        "slides_count": len(result_paths),
        "images": result_paths
    })

# --------------------------------------------------
# ENDPOINT 4 → Regenerate story from chosen head-swap
# --------------------------------------------------
@app.post("/regenerate-story")
async def regenerate_story(
    language: str = Form(...),
    gender: str = Form(...),
    story_code: str = Form(...),
    user_name: str = Form(...),
    selected_image_path: str = Form(...),
):
    if not os.path.exists(selected_image_path):
        raise HTTPException(404, "Selected image not found")

    result = run_story_pipeline(
        language,
        gender,
        story_code,
        user_name,
        character_image_path="",
        
        pre_swapped_image_path=selected_image_path
    )

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    pdf_path = os.path.join(RESULT_FOLDER, result["pdf_name"])
    if not create_pdf_from_images(result["images"], pdf_path):
        raise HTTPException(500, "PDF generation failed")

    return FileResponse(pdf_path, media_type="application/pdf", filename=result["pdf_name"])
