# -*- coding: utf-8 -*-
"""
🎨 Parallel Text Processor - Standalone
معالجة النصوص المتوازية - ملف مستقل
"""

import sys
import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import re
from PIL import Image

# ============================================================
# إعدادات الظل
# ============================================================
ENABLE_TEXT_SHADOW = True
TEXT_SHADOW_STYLE = "2px 2px 4px rgba(0, 0, 0, 0.7)"


def inject_font_family(html_text, font_family):
    """حقن اسم الخط في HTML"""
    if not font_family:
        return html_text
    
    html_text = re.sub(r"font-family:\s*[^;'\"]+[;\"]", "", html_text)
    html_text = re.sub(r"font-family:\s*'[^']+'[;\"]?", "", html_text)
    html_text = re.sub(r'font-family:\s*"[^"]+"[;\"]?', "", html_text)
    
    def add_font_to_style(match):
        style_content = match.group(1)
        new_style = f"font-family: '{font_family}' !important; "
        
        if ENABLE_TEXT_SHADOW:
            new_style += f"text-shadow: {TEXT_SHADOW_STYLE}; "
        
        new_style += style_content
        return f'style="{new_style}"'
    
    html_text = re.sub(r'style="([^"]*)"', add_font_to_style, html_text)
    
    base_style = f"font-family: '{font_family}' !important;"
    if ENABLE_TEXT_SHADOW:
        base_style += f" text-shadow: {TEXT_SHADOW_STYLE};"
    
    html_text = re.sub(r'<p(\s|>)', f'<p style="{base_style}"\\1', html_text)
    html_text = re.sub(r'<span(\s|>)', f'<span style="{base_style}"\\1', html_text)
    html_text = re.sub(r'<div(\s|>)', f'<div style="{base_style}"\\1', html_text)
    
    return html_text


def scale_font_sizes(html_text, global_font):
    """تكبير أو تصغير كل أحجام الخطوط"""
    if not global_font or global_font == 0:
        return html_text
    
    def replace_font_size(match):
        original_size = float(match.group(1))
        unit = match.group(2) if len(match.groups()) > 1 else 'pt'
        new_size = int(original_size * global_font)
        if new_size < 1:
            new_size = 1
        return f'font-size:{new_size}{unit}'
    
    return re.sub(r'font-size:(\d+(?:\.\d+)?)(pt|px)?', replace_font_size, html_text)


def process_single_image_worker(args):
    """
    Worker function للمعالجة المتوازية
    معالجة صورة واحدة في process منفصل
    """
    (image_name, image_path, text_data_list, is_first_slide,
     first_font_path, rest_font_path) = args
    
    try:
        from PySide6.QtWidgets import QApplication, QLabel
        from PySide6.QtGui import QPixmap, QPainter, QFontDatabase
        from PySide6.QtCore import Qt, QPoint, QBuffer, QIODevice
        
        # إنشاء QApplication في كل process
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # تحميل الخطوط
        fonts_loaded = {}
        
        if os.path.exists(first_font_path):
            font_id = QFontDatabase.addApplicationFont(first_font_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    fonts_loaded['first'] = families[0]
        
        if os.path.exists(rest_font_path):
            font_id = QFontDatabase.addApplicationFont(rest_font_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    fonts_loaded['rest'] = families[0]
        
        # اختيار الخط المناسب
        font_family = None
        if is_first_slide and 'first' in fonts_loaded:
            font_family = fonts_loaded['first']
        elif not is_first_slide and 'rest' in fonts_loaded:
            font_family = fonts_loaded['rest']
        
        # تحميل الصورة مباشرة من الملف (مثل write_text.py)
        base_pixmap = QPixmap(str(image_path))
        if base_pixmap.isNull():
            return (image_name, None, "فشل تحميل الصورة")
        
        # إنشاء صورة جديدة
        result_pixmap = QPixmap(base_pixmap.size())
        result_pixmap.fill(Qt.transparent)
        
        painter = QPainter(result_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)
        
        # رسم النصوص
        for element in text_data_list:
            html = element.get('html', '')
            x = element.get('x', 0)
            y = element.get('y', 0)
            width = element.get('width', 400)
            height = element.get('height', 200)
            global_font = element.get('global_font', 0)
            
            # حقن الخط في HTML
            if font_family:
                html = inject_font_family(html, font_family)
            
            # تعديل حجم الخط
            if global_font != 0:
                html = scale_font_sizes(html, global_font)
            
            # إنشاء label
            label = QLabel()
            label.setText(html)
            label.setWordWrap(True)
            label.setStyleSheet("background: transparent;")
            label.setGeometry(x, y, width, height)
            
            # رسم
            label.render(painter, QPoint(x, y))
        
        painter.end()
        
        # تحويل لـ bytes
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        result_pixmap.save(buffer, "PNG")
        buffer.close()
        
        result_bytes = bytes(buffer.data())
        
        return (image_name, result_bytes, "✅")
        
    except Exception as e:
        import traceback
        error_msg = f"خطأ: {str(e)}\n{traceback.format_exc()}"
        return (image_name, None, error_msg)


def apply_text_parallel(images_dict, text_data, first_font_path, rest_font_path, num_workers=None):
    """
    معالجة متوازية لإضافة النصوص على الصور
    
    Args:
        images_dict: قاموس الصور {اسم: صورة}
        text_data: بيانات النصوص
        first_font_path: مسار خط السلايد الأول
        rest_font_path: مسار خط باقي السلايدات
        num_workers: عدد الـ workers (None = تلقائي)
    
    Returns:
        dict: قاموس الصور مع النصوص
    """
    import tempfile
    import shutil
    from pathlib import Path
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"\n🚀 معالجة متوازية - {num_workers} workers")
    
    # إنشاء مجلد مؤقت للصور (مثل write_text.py)
    temp_dir = tempfile.mkdtemp(prefix="parallel_text_")
    temp_path = Path(temp_dir)
    
    try:
        # تحضير المهام
        tasks = []
        
        for idx, (image_name, img) in enumerate(images_dict.items()):
            # إذا لم يكن هناك نصوص لهذه الصورة، نتخطاها
            if image_name not in text_data:
                continue
            
            labels_list = text_data[image_name]
            is_first = (idx == 0)
            
            # حفظ الصورة كملف مؤقت (بدلاً من تحويلها لـ bytes)
            temp_image_path = temp_path / f"{image_name}.png"
            cv2.imwrite(str(temp_image_path), img)
            
            # إضافة المهمة (نمرر المسار بدلاً من bytes)
            tasks.append((
                image_name,
                str(temp_image_path),
                labels_list,
                is_first,
                first_font_path,
                rest_font_path
            ))
        
        if not tasks:
            print("   ⚠️  لا توجد مهام للمعالجة")
            return images_dict
        
        print(f"✅ تم تحضير {len(tasks)} مهمة\n")
        print(f"🔄 بدء المعالجة...\n")
        
        start_time = time.time()
        processed_images = {}
        
        # معالجة متوازية
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_image_worker, tasks)
        
        # تحويل النتائج من bytes إلى images
        completed = 0
        failed = 0
        
        for image_name, image_bytes, status in results:
            completed += 1
            if image_bytes is not None:
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    processed_images[image_name] = img
                    print(f"[{completed}/{len(tasks)}] ✅ {image_name}")
                else:
                    failed += 1
                    print(f"[{completed}/{len(tasks)}] ❌ {image_name} - فشل فك تشفير")
                    # استخدام الصورة الأصلية
                    if image_name in images_dict:
                        processed_images[image_name] = images_dict[image_name]
            else:
                failed += 1
                print(f"[{completed}/{len(tasks)}] ❌ {image_name} - {status}")
                # استخدام الصورة الأصلية
                if image_name in images_dict:
                    processed_images[image_name] = images_dict[image_name]
        
        # إضافة الصور التي لم تحتاج معالجة نصوص
        for image_name, img in images_dict.items():
            if image_name not in processed_images:
                processed_images[image_name] = img
        
        elapsed = time.time() - start_time
        success_count = len(tasks) - failed
        
        print(f"\n{'='*60}")
        print(f"✅ انتهت المعالجة!")
        print(f"📊 النجاح: {success_count}/{len(tasks)}")
        if failed > 0:
            print(f"⚠️  الفشل: {failed}/{len(tasks)}")
        print(f"⏱️  الوقت: {elapsed:.2f} ثانية")
        if elapsed > 0:
            print(f"⚡ السرعة: {len(tasks)/elapsed:.2f} صورة/ثانية")
        print(f"{'='*60}\n")
        
        return processed_images
    
    finally:
        # تنظيف الملفات المؤقتة
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️  تحذير: فشل حذف المجلد المؤقت: {e}")


def create_pdf_from_images(images_list, output_path):
    """
    إنشاء PDF من قائمة الصور باستخدام PIL
    """
    if not images_list:
        print("ERROR: No images for PDF")
        return False
    
    print("\nCreating PDF...")
    
    # تحويل OpenCV images إلى PIL Images
    pil_images = []
    
    for idx, img in enumerate(images_list, 1):
        # تحويل BGR (OpenCV) → RGB (PIL)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # تحويل RGBA → RGB إذا لزم الأمر
        if pil_img.mode == 'RGBA':
            rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            pil_images.append(rgb_img)
        else:
            pil_images.append(pil_img.convert('RGB'))
        
        print(f"   Converting image {idx}/{len(images_list)}")
    
    if not pil_images:
        print("ERROR: No valid images to save")
        return False
    
    # حفظ كـ PDF
    print("Writing PDF...")
    try:
        pil_images[0].save(
            output_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else None
        )
        
        print(f"Done: {output_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create PDF - {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 Parallel Text Processor - Standalone")
    print("="*60 + "\n")
    
    print("هذا ملف مساعد للمعالجة المتوازية")
    print("استخدمه من الكود الرئيسي عن طريق:")
    print("  from parallel_text_processor import apply_text_parallel")
