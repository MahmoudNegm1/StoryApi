# -*- coding: utf-8 -*-
"""
🖼️ Image Processor Module
"""

import os
import cv2
import time
import shutil

from Codes.config import HEAD_SWAP_DELAY
from Codes.api_client import perform_head_swap
from Codes.text_handler import render_image
from Codes.utils import get_image_dimensions


def resize_image_to_resolution(image, target_width, target_height):
    """تغيير حجم الصورة للدقة المطلوبة"""
    current_h, current_w = image.shape[:2]
    if current_w == target_width and current_h == target_height:
        return image
    if target_width < current_w or target_height < current_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def apply_resolution_to_images(images_dict, resolution_slides, use_parallel=None):
    """
    تطبيق الدقة المطلوبة على الصور
    
    Args:
        images_dict: قاموس الصور {اسم: صورة}
        resolution_slides: قائمة [(اسم, عرض, ارتفاع), ...]
        use_parallel: غير مستخدم (للتوافق مع الكود القديم)
    
    Returns:
        list: قائمة الصور بعد تغيير الحجم
    """
    resized_images = []
    for slide_name, target_w, target_h in resolution_slides:
        if slide_name in images_dict:
            img = images_dict[slide_name]
            resized_img = resize_image_to_resolution(img, target_w, target_h)
            resized_images.append(resized_img)
    return resized_images


def apply_text_to_images(images_dict, text_data, original_dims_dict, app, fonts_loaded, language, use_parallel=None):
    """
    إضافة النص على الصور مع دعم المعالجة المتوازية
    
    Args:
        images_dict: قاموس الصور {اسم: صورة}
        text_data: بيانات النصوص
        original_dims_dict: الأبعاد الأصلية للصور
        app: QApplication instance
        fonts_loaded: الخطوط المحملة
        language: اللغة
        use_parallel: استخدام المعالجة المتوازية (None = من config, True/False = تحديد يدوي)
    
    Returns:
        dict: قاموس الصور مع النصوص
    """
    from Codes.config import USE_PARALLEL_TEXT_PROCESSING, MAX_TEXT_WORKERS, BASE_DIR
    
    # تحديد ما إذا كنا سنستخدم المعالجة المتوازية
    if use_parallel is None:
        use_parallel = USE_PARALLEL_TEXT_PROCESSING
    
    # إذا كانت المعالجة المتوازية مفعلة وعدد الصور > 1
    if use_parallel and len(images_dict) > 1:
        print(f"\n🚀 استخدام المعالجة المتوازية ({MAX_TEXT_WORKERS} workers)...")
        return _apply_text_parallel(images_dict, text_data, original_dims_dict, language)
    else:
        # المعالجة التسلسلية
        return _apply_text_sequential(images_dict, text_data, original_dims_dict, app, fonts_loaded, language)



def _apply_text_sequential(images_dict, text_data, original_dims_dict, app, fonts_loaded, language):
    """
    المعالجة التسلسلية للنصوص (الطريقة القديمة)
    """
    processed_images = {}
    
    for image_name, img in images_dict.items():
        current_h, current_w = img.shape[:2]
        
        # إرجاع الصورة لأبعادها الأصلية إذا كانت مختلفة
        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            
            if current_w != orig_w or current_h != orig_h:
                img = resize_image_to_resolution(img, orig_w, orig_h)
                print(f"   ↩️  Restored {image_name} to original: {orig_w}x{orig_h}")
        
        # إضافة النصوص
        if image_name not in text_data:
            processed_images[image_name] = img
            continue
            
        labels_list = text_data[image_name]
        is_first = (image_name == 'slide_01' or image_name == list(text_data.keys())[0])
        
        img_with_text = render_image(
            image_name=image_name,
            text_data_list=labels_list,
            app=app,
            fonts_loaded=fonts_loaded,
            is_first_slide=is_first,
            image_data=img
        )
        
        processed_images[image_name] = img_with_text if img_with_text is not None else img
    
    return processed_images


def _restore_image_worker(args):
    """
    Worker function لإرجاع صورة واحدة لأبعادها الأصلية
    """
    image_name, img, orig_w, orig_h = args
    current_h, current_w = img.shape[:2]
    
    if current_w != orig_w or current_h != orig_h:
        img = resize_image_to_resolution(img, orig_w, orig_h)
        return (image_name, img, f"↩️  Restored to {orig_w}x{orig_h}")
    else:
        return (image_name, img, None)


def _apply_text_parallel(images_dict, text_data, original_dims_dict, language):
    """
    المعالجة المتوازية للنصوص - استخدام الملف المستقل
    """
    from Codes.config import MAX_TEXT_WORKERS, BASE_DIR
    from Codes.parallel_text_processor import apply_text_parallel
    from Codes.utils import read_info_file
    from multiprocessing import Pool, cpu_count
    import os
    
    # استرجاع الصور لأبعادها الأصلية بشكل متوازي
    restored_images = {}
    restore_tasks = []
    
    for image_name, img in images_dict.items():
        if image_name in original_dims_dict:
            orig_w, orig_h = original_dims_dict[image_name]
            restore_tasks.append((image_name, img, orig_w, orig_h))
        else:
            # لا يحتاج restore
            restored_images[image_name] = img
    
    # معالجة متوازية للـ restore
    if restore_tasks:
        print(f"\n🔄 Restoring {len(restore_tasks)} images to original dimensions...")
        num_workers = min(MAX_TEXT_WORKERS, len(restore_tasks))
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(_restore_image_worker, restore_tasks)
        
        for image_name, img, message in results:
            restored_images[image_name] = img
            if message:
                print(f"   {message}: {image_name}")
    
    # تحديد مسارات الخطوط
    from Codes.config import EN_FIRST_SLIDE_FONT, EN_REST_SLIDES_FONT, AR_FIRST_SLIDE_FONT, AR_REST_SLIDES_FONT
    
    if language == 'en':
        first_font_path = EN_FIRST_SLIDE_FONT
        rest_font_path = EN_REST_SLIDES_FONT
    else:
        first_font_path = AR_FIRST_SLIDE_FONT
        rest_font_path = AR_REST_SLIDES_FONT
    
    # استدعاء المعالجة المتوازية
    processed_images = apply_text_parallel(
        images_dict=restored_images,
        text_data=text_data,
        first_font_path=first_font_path,
        rest_font_path=rest_font_path,
        num_workers=MAX_TEXT_WORKERS
    )
    
    return processed_images


def _upload_worker(args):
    """Worker function لرفع الصور"""
    path, = args
    from Codes.api_client import upload_to_imgbb
    return upload_to_imgbb(path)

def _head_swap_worker(args):
    """Worker function لمعالجة Head Swap"""
    scene_path, face_url, output_path, max_retries, retry_delay = args
    from Codes.api_client import perform_head_swap
    from Codes.config import SIMILARITY_THRESHOLD
    from Codes.utils import compare_images_similarity
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = perform_head_swap(
                target_image_path=scene_path,
                face_image_path=None,  # Not needed when using face_url
                output_filename=output_path,
                face_url_cached=face_url
            )
            
            if result and os.path.exists(output_path):
                # التحقق من نجاح العملية عن طريق مقارنة التشابه
                modified_img = cv2.imread(output_path)
                original_img = cv2.imread(scene_path)
                
                if modified_img is not None and original_img is not None:
                    similarity = compare_images_similarity(original_img, modified_img)
                    
                    if similarity is not None:
                        # إذا التشابه أقل من العتبة، يعني الوجه تغير (نجاح)
                        if similarity <= SIMILARITY_THRESHOLD:
                             return (output_path, True, f"✅ Done (Sim: {similarity*100:.1f}%)")
                        else:
                            # الوجه لم يتغير كثيراً
                             if retry_count < max_retries - 1:
                                 os.remove(output_path)  # حذف المحاولة الفاشلة
                                 time.sleep(retry_delay)
                    else:
                        return (output_path, True, "✅ Done (Sim check failed)")
                else:
                    return (output_path, True, "✅ Done (Read failed)")
            
        except Exception as e:
            pass
            
        retry_count += 1
        if retry_count < max_retries:
            time.sleep(retry_delay)
            
    return (output_path, False, "❌ Failed after retries")


def process_head_swap(clean_images_folder, character_image_path, character_name, story_folder, prompts_dict=None, use_parallel=None):
    """
    معالجة Head Swap باستخدام WaveSpeed API (نسخة محسنة وسريعة ⚡)
    """
    from Codes.api_client import upload_to_imgbb
    from Codes.config import HEAD_SWAP_DELAY, MAX_RETRIES, RETRY_DELAY, API_WORKERS, UPLOAD_WORKERS
    from multiprocessing import Pool
    
    # إنشاء مجلد الحفظ
    head_swap_folder = os.path.join(story_folder, "Head_swap")
    os.makedirs(head_swap_folder, exist_ok=True)
    char_output_folder = os.path.join(head_swap_folder, character_name)
    os.makedirs(char_output_folder, exist_ok=True)
    
    # المجلدات
    api_images_folder = os.path.join(story_folder, "api_images")
    normal_images_folder = os.path.join(story_folder, "normal_images")
    
    # قراءة الملفات
    api_images = []
    if os.path.exists(api_images_folder):
        api_images = [f for f in os.listdir(api_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    normal_images = []
    if os.path.exists(normal_images_folder):
        normal_images = [f for f in os.listdir(normal_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    all_images = sorted(api_images + normal_images)
    if not all_images:
        print("   ❌ No images found")
        return None, None
    
    processed_images_dict = {}
    original_dims_dict = {}
    
    print(f"\n📊 Processing {len(all_images)} images...")
    print(f"   🔹 API images: {len(api_images)}")
    print(f"   🔹 Normal images: {len(normal_images)}")
    
    # 1. رفع صورة الوجه (الشخصية) مرة واحدة
    face_url = None
    if api_images:
        print(f"\n☁️  Uploading face image...")
        face_url = upload_to_imgbb(character_image_path)
        if not face_url:
            print(f"   ❌ Failed to upload face image")
            return None, None
    
    # 2. تحديد المهام وتجهيز الصور العادية
    api_tasks_prep = [] # (filename, full_path, output_path)
    
    for idx, filename in enumerate(all_images, 1):
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(char_output_folder, f"{name_no_ext}.jpg")
        
        # تحديد المسار
        if filename in api_images:
            src_path = os.path.join(api_images_folder, filename)
            is_api = True
        else:
            src_path = os.path.join(normal_images_folder, filename)
            is_api = False
            
        # حفظ الأبعاد
        orig_w, orig_h = get_image_dimensions(src_path)
        if orig_w:
            original_dims_dict[name_no_ext] = (orig_w, orig_h)
            
        # التحقق من الوجود مسبقاً
        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            if img is not None:
                processed_images_dict[name_no_ext] = img
                print(f"   ✅ Found existing: {filename}")
                continue
        
        # معالجة الصور العادية فوراً
        if not is_api:
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(output_path, img)
                processed_images_dict[name_no_ext] = img
                print(f"   ⏩ Normal image: {filename}")
            continue
            
        # إضافة لقائمة مهام API
        api_tasks_prep.append((filename, src_path, output_path))
    
    # 3. معالجة صور API بشكل متوازي
    if api_tasks_prep and face_url:
        print(f"\n🚀 Starting Parallel API Processing (Images: {len(api_tasks_prep)})...")
        print(f"   🔄 Processing {len(api_tasks_prep)} images with {API_WORKERS} workers...")
        
        swap_args = []
        for filename, src_path, out_path in api_tasks_prep:
            # نعتمد على أن كل worker سيقوم برفع صورته الخاصة
            # هذا يحقق الرفع المتوازي والمعالجة المتوازية في آن واحد
            swap_args.append((src_path, face_url, out_path, MAX_RETRIES, RETRY_DELAY))
            
        with Pool(processes=API_WORKERS) as pool:
            # نستخدم imap_unordered للحصول على النتائج أولاً بأول
            for i, (out_path, success, msg) in enumerate(pool.imap_unordered(_head_swap_worker, swap_args), 1):
                name = os.path.basename(out_path)
                status_icon = "✅" if success else "❌"
                print(f"   [{i}/{len(api_tasks_prep)}] {status_icon} {name} - {msg}")
                
                if success and os.path.exists(out_path):
                    img = cv2.imread(out_path)
                    if img is not None:
                        name_no_ext = os.path.splitext(name)[0]
                        processed_images_dict[name_no_ext] = img
        
    if processed_images_dict:
        return processed_images_dict, original_dims_dict
    
    return None, None
