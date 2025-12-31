# -*- coding: utf-8 -*-
"""
✍️ Text Handler Module
================================================
معالجة النصوص والخطوط وإضافتها على الصور
"""

import os
import json
import re
import cv2
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QPainter, QFontDatabase
from PySide6.QtCore import Qt, QBuffer, QIODevice

from Codes.config import (
    EN_FIRST_SLIDE_FONT, EN_REST_SLIDES_FONT,
    AR_FIRST_SLIDE_FONT, AR_REST_SLIDES_FONT,
    ENABLE_TEXT_SHADOW, TEXT_SHADOW_STYLE
)


def load_custom_fonts(language, first_slide_font_path=None, rest_slides_font_path=None, base_dir=None):
    """
    تحميل الخطوط المخصصة حسب اللغة
    
    Args:
        language: اللغة (en أو ar)
        first_slide_font_path: مسار خط السلايد الأول (من info.txt)
        rest_slides_font_path: مسار خط باقي السلايدات (من info.txt)
        base_dir: المسار الأساسي للمشروع (لتحويل المسارات النسبية إلى مطلقة)
    
    Returns:
        dict: قاموس يحتوي على الخطوط المحملة
    """
    fonts_loaded = {}
    
    # إذا تم تمرير مسارات من info.txt، استخدمها
    if first_slide_font_path and base_dir:
        # تحويل المسار النسبي إلى مطلق
        first_font = os.path.join(base_dir, first_slide_font_path)
    elif language == 'en':
        first_font = EN_FIRST_SLIDE_FONT
    else:
        first_font = AR_FIRST_SLIDE_FONT
    
    if rest_slides_font_path and base_dir:
        # تحويل المسار النسبي إلى مطلق
        rest_font = os.path.join(base_dir, rest_slides_font_path)
    elif language == 'en':
        rest_font = EN_REST_SLIDES_FONT
    else:
        rest_font = AR_REST_SLIDES_FONT
    
    # تحميل خط السلايد الأول
    if os.path.exists(first_font):
        font_id = QFontDatabase.addApplicationFont(first_font)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                fonts_loaded['first'] = families[0]
                print(f"✅ تم تحميل خط السلايد الأول: {families[0]} من {os.path.basename(first_font)}")
        else:
            print(f"⚠️ فشل تحميل خط السلايد الأول: {first_font}")
    else:
        print(f"⚠️ خط السلايد الأول غير موجود: {first_font}")
    
    # تحميل خط باقي السلايدات
    if os.path.exists(rest_font):
        font_id = QFontDatabase.addApplicationFont(rest_font)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                fonts_loaded['rest'] = families[0]
                print(f"✅ تم تحميل خط باقي السلايدات: {families[0]} من {os.path.basename(rest_font)}")
        else:
            print(f"⚠️ فشل تحميل خط باقي السلايدات: {rest_font}")
    else:
        print(f"⚠️ خط باقي السلايدات غير موجود: {rest_font}")
    
    return fonts_loaded


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
        unit = match.group(2)  # px or pt
        new_size = int(original_size * global_font)
        if new_size < 1:
            new_size = 1
        return f'font-size:{new_size}{unit}'
    
    # دعم pt و px
    return re.sub(r'font-size:(\d+(?:\.\d+)?)(pt|px)', replace_font_size, html_text)


def replace_name_in_html(html_text, user_name, is_first_slide=False, language='en'):
    """استبدال [*NAME*] أو [*الاسم*] بالاسم المُدخل"""
    
    if language == 'en' and '[*NAME*]' in html_text:
        if is_first_slide:
            replacement_name = user_name.upper()
        else:
            replacement_name = user_name
        html_text = html_text.replace('[*NAME*]', replacement_name)
    
    elif language == 'ar' and '[*الاسم*]' in html_text:
        if is_first_slide:
            replacement_name = user_name.upper()
        else:
            replacement_name = user_name
        html_text = html_text.replace('[*الاسم*]', replacement_name)
    
    return html_text



def detect_format_type(text_data):
    """
    كشف نوع التنسيق
    Format 1: {"slide_01": [labels], "slide_02": [labels]}
    Format 2: {"slide_01": [labels], "slide_02": [labels]} but with different structure
    """
    if not isinstance(text_data, dict):
        return None
    
    # جرب أول key
    first_key = list(text_data.keys())[0] if text_data else None
    if not first_key:
        return None
    
    first_value = text_data[first_key]
    
    # تحقق إنه list فيه objects
    if isinstance(first_value, list) and len(first_value) > 0:
        if isinstance(first_value[0], dict):
            return "format_standard"  # كلا التنسيقين متشابهين في الهيكل
    
    return None


def read_text_data(file_path, user_name='', language='en'):
    """قراءة بيانات النص من الملف مع دعم التنسيقين واستبدال الاسم"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            if not raw_content.strip():
                return None
            
            # 🔥 Pre-processing: نصلح المشاكل الشائعة قبل JSON parsing
            
            # Strategy: نلف على كل character ونحدد متى نكون جوه HTML string
            result = []
            i = 0
            
            while i < len(raw_content):
                # نشوف لو وصلنا لـ "html":
                if raw_content[i:i+7] == '"html":':
                    result.append(raw_content[i:i+7])
                    i += 7
                    
                    # نتخطى المسافات
                    while i < len(raw_content) and raw_content[i] in ' \t':
                        result.append(raw_content[i])
                        i += 1
                    
                    # لو بدأ بـ " يبقى ده HTML string
                    if i < len(raw_content) and raw_content[i] == '"':
                        result.append('"')
                        i += 1
                        
                        # دلوقتي احنا جوه HTML string
                        # نقرأ لحد ما نلاقي closing " (مش escaped)
                        html_chars = []
                        
                        while i < len(raw_content):
                            char = raw_content[i]
                            
                            # لو لقينا "
                            if char == '"':
                                # نشوف لو هي closing quote فعلاً
                                # نتأكد من الـ context بعدها
                                peek_ahead = raw_content[i+1:i+20].lstrip()
                                
                                # لو بعدها , أو } يبقى دي نهاية HTML
                                if peek_ahead.startswith(',') or peek_ahead.startswith('}'):
                                    # دي نهاية HTML string
                                    # نحفظ الـ HTML ونكمل
                                    cleaned_html = ''.join(html_chars)
                                    
                                    # 🔥 نظف الـ HTML من كل المشاكل
                                    # 1. استبدل \" بـ '
                                    cleaned_html = cleaned_html.replace('\\"', "'")
                                    cleaned_html = cleaned_html.replace("\\'", "'")
                                    
                                    # 2. استبدل أي " بـ ' (ماعدا اللي في attributes)
                                    # نستخدم regex ذكي
                                    import re
                                    # نستبدل " جوه النص (مش في attributes)
                                    # Pattern: " اللي مش بعد = ومش قبل >
                                    cleaned_html = re.sub(r'(?<!=)"(?![>\s])', "'", cleaned_html)
                                    
                                    # 3. نظف escape sequences تانية
                                    cleaned_html = cleaned_html.replace('\\n', ' ')
                                    cleaned_html = cleaned_html.replace('\\t', ' ')
                                    cleaned_html = cleaned_html.replace('\\r', '')
                                    cleaned_html = cleaned_html.replace('\\/', '/')
                                    
                                    # 4. إصلاح حالة خاصة: ," داخل النص
                                    # نستبدلها بـ ,'
                                    cleaned_html = cleaned_html.replace(',"', ",'")
                                    cleaned_html = cleaned_html.replace('",', "',")
                                    
                                    result.append(cleaned_html)
                                    result.append('"')
                                    i += 1
                                    break
                                else:
                                    # مش نهاية، دي " عادية جوه النص
                                    html_chars.append("'")  # نحولها لـ '
                                    i += 1
                            
                            elif char == '\\' and i + 1 < len(raw_content):
                                next_char = raw_content[i + 1]
                                if next_char == '"':
                                    # \" نحولها لـ '
                                    html_chars.append("'")
                                    i += 2
                                elif next_char == "'":
                                    # \' نحولها لـ '
                                    html_chars.append("'")
                                    i += 2
                                elif next_char == '\\':
                                    # \\ نخليها \
                                    html_chars.append('\\')
                                    i += 2
                                elif next_char in 'ntr':
                                    # \n \t \r نحولهم لمسافة
                                    html_chars.append(' ')
                                    i += 2
                                else:
                                    # باقي الـ escapes نشيل الـ \
                                    i += 1
                            else:
                                html_chars.append(char)
                                i += 1
                        
                        continue
                
                # لو مش HTML، نكتب عادي
                result.append(raw_content[i])
                i += 1
            
            content = ''.join(result)
            
            # Parse as JSON
            data = json.loads(content)
            
            # استبدال الاسم (User Name)
            if user_name:
                slide_index = 0
                for image_name, labels_list in data.items():
                    if isinstance(labels_list, list):
                        for label in labels_list:
                            if 'html' in label:
                                is_first = (slide_index == 0)
                                label['html'] = replace_name_in_html(label['html'], user_name, is_first, language)
                    slide_index += 1
            
            return data
                
    except FileNotFoundError:
        print(f"❌ الملف غير موجود: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️ خطأ في تنسيق JSON: {e}")
        # محاولة طباعة معلومات الخطأ للمساعدة
        if hasattr(e, 'lineno'):
             print(f"   السطر {e.lineno}, العمود {e.colno}, الموقع {e.pos}")
        return None
    except Exception as e:
        print(f"⚠️ خطأ في قراءة الملف: {e}")
        import traceback
        traceback.print_exc()
        return None


def scale_text_positions(labels_list, ratio_x, ratio_y):
    """
    تطبيق النسب على مواضع النصوص
    
    Args:
        labels_list: قائمة النصوص
        ratio_x: نسبة التغيير في العرض
        ratio_y: نسبة التغيير في الارتفاع
    
    Returns:
        list: النصوص المعدلة
    """
    scaled_list = []
    
    # حساب العامل المشترك لتغيير حجم الخط
    # نستخدم الجذر التربيعي (geometric mean) بدلاً من المتوسط الحسابي
    # لأنه يعطي نتيجة أفضل عند تصغير/تكبير الصور
    # مثال: لو ratio_x = ratio_y = 0.25
    #   - المتوسط الحسابي = 0.25 (الخط يصغر جداً!)
    #   - الجذر التربيعي = sqrt(0.25 * 0.25) = 0.25 (نفس النتيجة في هذه الحالة)
    # لكن لو ratio_x = 0.5 و ratio_y = 0.5
    #   - المتوسط الحسابي = 0.5
    #   - الجذر التربيعي = sqrt(0.5 * 0.5) = 0.5 (نفس النتيجة)
    # 
    # في الواقع، الفرق يظهر لما النسب تكون مختلفة
    # لكن الأهم هو إننا نستخدم نسبة معقولة تحافظ على قراءة الخط
    import math
    font_ratio = math.sqrt(ratio_x * ratio_y)
    
    for item in labels_list:
        new_item = item.copy()
        
        # تطبيق النسب على الإحداثيات والأبعاد
        new_item['x'] = int(item.get('x', 0) * ratio_x)
        new_item['y'] = int(item.get('y', 0) * ratio_y)
        new_item['width'] = int(item.get('width', 400) * ratio_x)
        new_item['height'] = int(item.get('height', 200) * ratio_y)
        
        # تطبيق النسبة على حجم الخط
        original_global_font = item.get('global_font', 0)
        if original_global_font != 0:
            new_item['global_font'] = original_global_font * font_ratio
            
        scaled_list.append(new_item)
        
    return scaled_list


def render_image(image_path=None, image_name="", text_data_list=None, app=None, fonts_loaded=None, is_first_slide=False, image_data=None, scale_x=1.0, scale_y=1.0, silent=False):
    """
    إضافة النصوص على الصورة
    
    Args:
        image_path: مسار الصورة (اختياري إذا تم تمرير image_data)
        image_name: اسم الصورة
        text_data_list: قائمة البيانات النصية
        app: تطبيق QT
        fonts_loaded: الخطوط المحملة
        is_first_slide: هل هي الشريحة الأولى
        image_data: بيانات الصورة (numpy array) بدلاً من قراءتها من المسار
        scale_x: نسبة التكبير/التصغير الأفقي (للمعلومات فقط، التطبيق يتم قبل هذه الدالة عادة)
        scale_y: نسبة التكبير/التصغير العمودي
        silent: إذا كان True، لا تطبع أي رسائل (للمعالجة المتوازية)
    """
    if not silent:
        print(f"\n🖼️  Rendering Text: {image_name}")
    
    # تحميل الصورة إما من المسار أو من البيانات المباشرة
    if image_data is not None:
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        # تحويل من BGR (OpenCV) إلى RGB (Qt)
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # تحويل numpy array إلى QImage
        from PySide6.QtGui import QImage
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
    elif image_path:
        pixmap = QPixmap(image_path)
    else:
        if not silent:
            print("❌ Error: No image path or data provided")
        return None

    if pixmap.isNull():
        if not silent:
            print(f"   ❌ Failed to load image for text rendering")
        return None
    
    font_family = None
    if is_first_slide and 'first' in fonts_loaded:
        font_family = fonts_loaded['first']
    elif not is_first_slide and 'rest' in fonts_loaded:
        font_family = fonts_loaded['rest']
    
    final_pixmap = QPixmap(pixmap.size())
    final_pixmap.fill(Qt.transparent)
    
    painter = QPainter(final_pixmap)
    painter.drawPixmap(0, 0, pixmap)
    
    # تطبيق scaling للنصوص إذا لم يتم تطبيقها مسبقاً (يتم تطبيقها عادة خارج هذه الدالة الآن)
    # لكن يمكن استخدام scale_x, scale_y إذا أردنا تطبيقها هنا
    
    for idx, item in enumerate(text_data_list, 1):
        html = item.get('html', '')
        x = item.get('x', 0)
        y = item.get('y', 0)
        w = item.get('width', 400)
        h = item.get('height', 200)
        global_font = item.get('global_font', 0)
        
        # ملاحظة: نفترض أن x, y, w, h, global_font تم تعديلهم بالفعل 
        # باستخدام scale_text_positions قبل استدعاء هذه الدالة
        
        if font_family:
            html = inject_font_family(html, font_family)
        
        if global_font != 0:
            html = scale_font_sizes(html, global_font)
        
        label = QLabel()
        label.setText(html)
        label.setWordWrap(True)
        label.setStyleSheet("background: transparent;")
        label.setGeometry(x, y, w, h)
        
        label.render(painter, label.pos())
        if not silent:
            print(f"   ✓ Label {idx}: ({x}, {y}) [{w}x{h}] FontScale: {global_font:.2f}")
    
    painter.end()
    
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    final_pixmap.save(buffer, "PNG")
    buffer.close()
    
    arr = np.frombuffer(buffer.data(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    return img



def render_image_worker(args):
    """
    Worker function للمعالجة المتوازية
    تعمل في process منفصل - بدون طباعة رسائل
    
    Args:
        args: tuple يحتوي على:
            - image_name: اسم الصورة
            - image_bytes: بيانات الصورة كـ bytes
            - text_data_list: قائمة النصوص
            - is_first_slide: هل هي الشريحة الأولى
            - first_font_path: مسار خط السلايد الأول
            - rest_font_path: مسار خط باقي السلايدات
            - language: اللغة
            - base_dir: المسار الأساسي
    
    Returns:
        tuple: (image_name, image_bytes, status_message)
    """
    (image_name, image_bytes, text_data_list, is_first_slide,
     first_font_path, rest_font_path, language, base_dir) = args
    
    try:
        # إنشاء QApplication في كل process
        from PySide6.QtWidgets import QApplication, QLabel
        from PySide6.QtGui import QPixmap, QPainter, QFontDatabase
        from PySide6.QtCore import Qt, QBuffer, QIODevice
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # تحميل الخطوط بدون طباعة رسائل
        fonts_loaded = {}
        
        # تحديد مسارات الخطوط
        if first_font_path and base_dir:
            first_font = os.path.join(base_dir, first_font_path)
        elif language == 'en':
            from config import EN_FIRST_SLIDE_FONT
            first_font = EN_FIRST_SLIDE_FONT
        else:
            from config import AR_FIRST_SLIDE_FONT
            first_font = AR_FIRST_SLIDE_FONT
        
        if rest_font_path and base_dir:
            rest_font = os.path.join(base_dir, rest_font_path)
        elif language == 'en':
            from config import EN_REST_SLIDES_FONT
            rest_font = EN_REST_SLIDES_FONT
        else:
            from config import AR_REST_SLIDES_FONT
            rest_font = AR_REST_SLIDES_FONT
        
        # تحميل خط السلايد الأول
        if os.path.exists(first_font):
            font_id = QFontDatabase.addApplicationFont(first_font)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    fonts_loaded['first'] = families[0]
        
        # تحميل خط باقي السلايدات
        if os.path.exists(rest_font):
            font_id = QFontDatabase.addApplicationFont(rest_font)
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
        
        # تحويل bytes إلى QPixmap
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            return (image_name, None, "فشل تحويل الصورة")
        
        # تحويل OpenCV إلى QPixmap
        height, width, channel = img_cv.shape
        bytes_per_line = 3 * width
        rgb_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        from PySide6.QtGui import QImage
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        base_pixmap = QPixmap.fromImage(q_img)
        
        if base_pixmap.isNull():
            return (image_name, None, "فشل تحويل QPixmap")
        
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
            from PySide6.QtCore import QPoint
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



