# -*- coding: utf-8 -*-
"""
🎮 UI Selector Module
================================================
واجهة المستخدم والاختيارات التفاعلية
"""

import os
from config import STORIES_FOLDER, CHARACTERS_FOLDER


def select_language():
    """اختيار اللغة"""
    print("\n" + "="*60)
    print("🌍 اختر اللغة / Choose Language:")
    print("="*60)
    print("1. عربي / Arabic")
    print("2. إنجليزي / English")
    
    while True:
        choice = input("\n👉 اختيارك (1 أو 2): ").strip()
        if choice == '1':
            return 'ar'
        elif choice == '2':
            return 'en'
        else:
            print("❌ اختيار غير صحيح! اختر 1 أو 2")


def select_gender():
    """اختيار الجنس"""
    print("\n" + "="*60)
    print("👤 اختر الجنس / Choose Gender:")
    print("="*60)
    print("1. ولد / Boy")
    print("2. بنت / Girl")
    
    while True:
        choice = input("\n👉 اختيارك (1 أو 2): ").strip()
        if choice == '1':
            return 'boy', 'Boys'
        elif choice == '2':
            return 'girl', 'Girls'
        else:
            print("❌ اختيار غير صحيح! اختر 1 أو 2")


def get_available_stories(gender):
    """الحصول على القصص المتاحة حسب الجنس"""
    if not os.path.isdir(STORIES_FOLDER):
        return []
    
    # تحديد المجلد حسب الجنس: Boys أو Girls
    gender_folder_name = "Boys" if gender == 'boy' else "Girls"
    gender_folder_path = os.path.join(STORIES_FOLDER, gender_folder_name)
    
    if not os.path.isdir(gender_folder_path):
        return []
    
    # الحصول على جميع المجلدات داخل Boys أو Girls
    stories = []
    for item in os.listdir(gender_folder_path):
        story_path = os.path.join(gender_folder_path, item)
        if os.path.isdir(story_path):
            stories.append(item)
    
    return stories


def select_story(gender):
    """اختيار القصة حسب الجنس"""
    stories = get_available_stories(gender)
    
    if not stories:
        gender_ar = "الأولاد" if gender == 'boy' else "البنات"
        print(f"❌ لا توجد قصص متاحة لـ {gender_ar}!")
        return None
    
    gender_ar = "الأولاد" if gender == 'boy' else "البنات"
    gender_en = "Boys" if gender == 'boy' else "Girls"
    
    print("\n" + "="*60)
    print(f"📚 قصص {gender_ar} المتاحة / Available {gender_en} Stories:")
    print("="*60)
    
    for idx, story in enumerate(stories, 1):
        print(f"{idx}. {story}")
    
    while True:
        try:
            choice = int(input(f"\n👉 اختر رقم القصة (1-{len(stories)}): ").strip())
            if 1 <= choice <= len(stories):
                selected_story = stories[choice - 1]
                # المسار الكامل: Stories/Boys أو Girls/اسم القصة
                gender_folder_name = "Boys" if gender == 'boy' else "Girls"
                story_path = os.path.join(STORIES_FOLDER, gender_folder_name, selected_story)
                print(f"✅ تم اختيار: {selected_story}")
                return story_path
            else:
                print(f"❌ اختر رقم بين 1 و {len(stories)}")
        except ValueError:
            print("❌ أدخل رقماً صحيحاً!")


def show_character_images(gender_folder):
    """عرض صور الشخصيات المتاحة"""
    from config import TEMP_CROPPED_FOLDER
    from utils import crop_face_only
    
    char_path = os.path.join(CHARACTERS_FOLDER, gender_folder)
    
    if not os.path.isdir(char_path):
        print(f"❌ المجلد '{char_path}' غير موجود!")
        return None, None
    
    images = [f for f in os.listdir(char_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"❌ لا توجد صور في '{char_path}'")
        return None, None
    
    print(f"\n📸 الصور المتاحة في {gender_folder}:")
    for idx, img in enumerate(images, 1):
        print(f"   {idx}. {img}")
    
    while True:
        try:
            choice = int(input(f"\n👉 اختر رقم الصورة (1-{len(images)}): ").strip())
            if 1 <= choice <= len(images):
                selected_image = images[choice - 1]
                selected_image_path = os.path.join(char_path, selected_image)
                character_name = os.path.splitext(selected_image)[0]
                print(f"✅ تم اختيار: {selected_image}")
                
                # قص الوجه تلقائياً
                print("\n✂️  قص الوجه من الصورة...")
                os.makedirs(TEMP_CROPPED_FOLDER, exist_ok=True)
                cropped_image_path = os.path.join(TEMP_CROPPED_FOLDER, f"cropped_{selected_image}")
                
                result_path = crop_face_only(selected_image_path, cropped_image_path, padding=2)
                
                if result_path:
                    print(f"✅ تم حفظ الوجه المقصوص في: {cropped_image_path}")
                    return result_path, character_name
                else:
                    print("⚠️  فشل قص الوجه، سيتم استخدام الصورة الأصلية")
                    return selected_image_path, character_name
            else:
                print(f"❌ اختر رقم بين 1 و {len(images)}")
        except ValueError:
            print("❌ أدخل رقماً صحيحاً!")


def get_user_name(language):
    """طلب اسم المستخدم"""
    print("\n" + "="*60)
    name_prompt = "👤 أدخل اسم البطل/البطلة:" if language == 'ar' else "👤 Enter the hero/heroine name:"
    user_name = input(f"{name_prompt} ").strip()
    
    if not user_name:
        print("❌ لم يتم إدخال اسم!")
        return None
    
    print(f"✅ تم استلام الاسم: {user_name}")
    return user_name
