# -*- coding: utf-8 -*-
"""
🎭 Stories Production API - With Logging
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from PySide6.QtWidgets import QApplication

from Codes.config import RESULT_FOLDER, BASE_DIR
from Codes.utils import read_info_file
from Codes.text_handler import load_custom_fonts, read_text_data
from Codes.image_processor import (
    process_head_swap,
    apply_text_to_images,
    apply_resolution_to_images,
)
from Codes.pdf_generator import create_pdf_from_images


# --------------------------------
# Logging Configuration
# --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("StoriesAPI")


# --------------------------------
# FastAPI App
# --------------------------------
app = FastAPI(
    title="Stories Production API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
# --------------------------------
# Helpers
# --------------------------------
def get_gender_folder(gender: str) -> str:
    logger.info(f"[Gender] Resolving gender folder for: {gender}")
    gender = gender.lower()

    if gender == "male":
        return os.path.join(BASE_DIR, "Characters", "Boys")
    if gender == "female":
        return os.path.join(BASE_DIR, "Characters", "Girls")

    raise HTTPException(400, "Invalid gender")


def get_story_folder(gender: str, code: str) -> str:
    logger.info(f"[Story] Resolving story folder | gender={gender}, code={code}")
    gender = gender.lower()

    if gender == "male":
        path = os.path.join(BASE_DIR, "Stories", "Boys", code)
    elif gender == "female":
        path = os.path.join(BASE_DIR, "Stories", "Girls", code)
    else:
        raise HTTPException(400, "Invalid gender")

    if not os.path.exists(path):
        logger.error(f"[Story] Story folder not found: {path}")
        raise HTTPException(404, f"Story not found for {gender}: {code}")

    return path


def get_default_character(gender_folder: str):
    logger.info(f"[Character] Searching default character in: {gender_folder}")
    for f in os.listdir(gender_folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.info(f"[Character] Default character found: {f}")
            return os.path.join(gender_folder, f), os.path.splitext(f)[0]

    logger.warning("[Character] No default character found")
    return None, None


# --------------------------------
# API Endpoint
# --------------------------------
@app.post("/generate-story")
async def generate_story(
    language: str = Form(...),
    gender: str = Form(...),
    story_code: str = Form(...),
    user_name: str = Form(...),
    character_image: UploadFile | None = File(None)
):
    try:
        logger.info("========== NEW STORY REQUEST ==========")
        logger.info(f"Input | language={language}, gender={gender}, story={story_code}, user={user_name}")

        # -----------------------------
        # Story & Character
        # -----------------------------
        story_folder = get_story_folder(gender, story_code)
        gender_folder = get_gender_folder(gender)

        if character_image:
            logger.info("[Upload] Character image uploaded")
            upload_dir = os.path.join(BASE_DIR, "TempUploads")
            os.makedirs(upload_dir, exist_ok=True)

            image_path = os.path.join(upload_dir, character_image.filename)
            with open(image_path, "wb") as f:
                f.write(await character_image.read())

            character_image_path = image_path
            character_name = os.path.splitext(character_image.filename)[0]
        else:
            logger.info("[Upload] No image uploaded, using default character")
            character_image_path, character_name = get_default_character(gender_folder)

        if not character_image_path:
            raise HTTPException(400, "No character image available")

        logger.info(f"[Character] Using image: {character_image_path}")

        # -----------------------------
        # Read Story Info
        # -----------------------------
        logger.info("[Story] Reading info file")
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

        # -----------------------------
        # Select Text File
        # -----------------------------
        if language.lower() == "en":
            text_file = os.path.join(translations, "en_text_data.txt")
            pdf_name = en_story_name or "Story_EN"
        else:
            ar_files = [f for f in os.listdir(translations) if f.startswith("ar_") and f.endswith(".txt")]
            if not ar_files:
                raise HTTPException(400, "Arabic translation missing")
            text_file = os.path.join(translations, ar_files[0])
            pdf_name = ar_story_name or "Story_AR"

        logger.info(f"[Text] Using text file: {text_file}")

        # -----------------------------
        # Read Text
        # -----------------------------
        text_data = read_text_data(text_file, user_name=user_name, language=language)

        # -----------------------------
        # Qt App
        # -----------------------------
        logger.info("[Qt] Initializing QApplication")
        qt_app = QApplication.instance() or QApplication(sys.argv)

        fonts_loaded = load_custom_fonts(
            language=language,
            first_slide_font_path=first_slide_font if language == "en" else ar_first_slide_font,
            rest_slides_font_path=rest_slides_font if language == "en" else ar_rest_slides_font,
            base_dir=BASE_DIR
        )

        logger.info("[Fonts] Fonts loaded successfully")

        # -----------------------------
        # Head Swap
        # -----------------------------
        logger.info("[Images] Starting head swap")
        processed_images, original_dims = process_head_swap(
            clean_images_folder=None,
            character_image_path=character_image_path,
            character_name=character_name,
            story_folder=story_folder
        )

        # -----------------------------
        # Apply Text
        # -----------------------------
        logger.info("[Images] Applying text to images")
        images_with_text = apply_text_to_images(
            images_dict=processed_images,
            text_data=text_data,
            original_dims_dict=original_dims,
            app=qt_app,
            fonts_loaded=fonts_loaded,
            language=language
        )

        # -----------------------------
        # Resize
        # -----------------------------
        logger.info("[Images] Applying resolution")
        final_images = (
            apply_resolution_to_images(images_with_text, resolution_slides)
            if resolution_slides
            else [images_with_text[k] for k in sorted(images_with_text)]
        )

        # -----------------------------
        # Generate PDF
        # -----------------------------
        os.makedirs(RESULT_FOLDER, exist_ok=True)

        pdf_filename = (
            pdf_name.replace("Name", user_name)
            if language.lower() == "en"
            else pdf_name.replace("الاسم", user_name)
        )

        pdf_path = os.path.join(RESULT_FOLDER, f"{pdf_filename}.pdf")

        logger.info(f"[PDF] Generating PDF: {pdf_path}")

        if not create_pdf_from_images(final_images, pdf_path):
            raise HTTPException(500, "PDF generation failed")

        logger.info("✅ STORY GENERATED SUCCESSFULLY")

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path)
        )

    except Exception as e:
        logger.exception("❌ ERROR DURING STORY GENERATION")
        raise HTTPException(500, f"Error: {str(e)}")
