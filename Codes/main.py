# -*- coding: utf-8 -*-
"""
🎭 Stories Production
"""

import sys
import os
from PySide6.QtWidgets import QApplication

from config import RESULT_FOLDER
from utils import read_info_file
from ui_selector import select_language, select_gender, select_story, show_character_images, get_user_name
from text_handler import load_custom_fonts, read_text_data, render_image
from image_processor import process_head_swap, apply_text_to_images, apply_resolution_to_images
from pdf_generator import create_pdf_from_images


def main():
    print("\n" + "="*30)
    print("Stories Production - Reversed")
    print("="*30)
    
    language = select_language()
    print(f"Language: {language}")
    
    gender, gender_folder = select_gender()
    
    story_folder = select_story(gender)
    if not story_folder:
        sys.exit(1)
    
    character_image_path, character_name = show_character_images(gender_folder)
    if not character_image_path:
        sys.exit(1)
    
    user_name = get_user_name(language)
    if not user_name:
        sys.exit(1)
    
    en_story_name, ar_story_name, resolution_slides, first_slide_font, rest_slides_font, ar_first_slide_font, ar_rest_slides_font = read_info_file(story_folder)

    print(f"\nStory Info:")
    if en_story_name: print(f"   EN: {en_story_name}")
    
    
    translations_folder = os.path.join(story_folder, "Translations")
    
    if language == 'en':
        text_file = os.path.join(translations_folder, "en_text_data.txt")
        pdf_name = en_story_name if en_story_name else "Story_EN"
    else:
        ar_files = [f for f in os.listdir(translations_folder) if f.startswith('ar_')]
        if ar_files:
            text_file = os.path.join(translations_folder, ar_files[0])
        else:
            sys.exit(1)
        pdf_name = ar_story_name if ar_story_name else "Story_AR"
    
    print(f"\nReading: {os.path.basename(text_file)}")
    text_data = read_text_data(text_file, user_name=user_name, language=language)
    
    if not text_data:
        sys.exit(1)
    
    # لا تنشئ QApplication إذا كنا سنستخدم المعالجة المتوازية
    # لأن كل worker سينشئ QApplication خاص به
    from config import USE_PARALLEL_TEXT_PROCESSING
    
    app = None
    fonts_loaded = None
    
    # فقط إذا كنا سنستخدم المعالجة التسلسلية
    if not USE_PARALLEL_TEXT_PROCESSING or len(text_data) <= 1:
        app = QApplication(sys.argv)
        
        from config import BASE_DIR
        selected_first_font = first_slide_font if language == 'en' else ar_first_slide_font
        selected_rest_font = rest_slides_font if language == 'en' else ar_rest_slides_font
        
        fonts_loaded = load_custom_fonts(
            language=language,
            first_slide_font_path=selected_first_font,
            rest_slides_font_path=selected_rest_font,
            base_dir=BASE_DIR
        )
    
    print("\nHead Swap Phase...")
    processed_images_dict, original_dims_dict = process_head_swap(
        clean_images_folder=None,  # Not used with WaveSpeed API
        character_image_path=character_image_path,
        character_name=character_name,
        story_folder=story_folder
    )

    if not processed_images_dict:
        sys.exit(1)

    print("\nAdding Text...")
    images_with_text = apply_text_to_images(
        images_dict=processed_images_dict,
        text_data=text_data,
        original_dims_dict=original_dims_dict,
        app=app,
        fonts_loaded=fonts_loaded,
        language=language
    )
    
    if not images_with_text:
        sys.exit(1)

    print("\nResizing...")
    final_images = []

    if not resolution_slides:
        sorted_image_names = sorted(images_with_text.keys())
        for image_name in sorted_image_names:
            final_images.append(images_with_text[image_name])
    else:
        final_images = apply_resolution_to_images(
            images_dict=images_with_text,
            resolution_slides=resolution_slides
        )

    if not final_images:
        sys.exit(1)

    print("\nGenerating PDF...")
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    if language == 'en':
        pdf_filename = pdf_name.replace('Name', user_name).replace('name', user_name).replace('NAME', user_name.upper())
    else:
        pdf_filename = pdf_name.replace('الاسم', user_name).replace('اسم', user_name)

    pdf_filename = f"{pdf_filename}.pdf"
    pdf_path = os.path.join(RESULT_FOLDER, pdf_filename)

    success = create_pdf_from_images(final_images, pdf_path)

    if app:
        app.quit()

    if success:
        print("\n" + "="*30)
        print(f"DONE! PDF: {pdf_path}")
        print("="*30 + "\n")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

