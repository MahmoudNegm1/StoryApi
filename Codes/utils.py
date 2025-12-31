# -*- coding: utf-8 -*-
"""
🛠️ Utility Functions
================================================
"""

import os
import json
import cv2  # New import for image dimensions

def read_info_file(folder_path):
    """قراءة ملف info.txt من المجلد المحدد"""
    info_file_path = os.path.join(folder_path, "info.txt")
    
    en_story_name = None
    ar_story_name = None
    resolution_slides = None
    first_slide_font = None
    rest_slides_font = None
    ar_first_slide_font = None
    ar_rest_slides_font = None
    
    if os.path.exists(info_file_path):
        try:
            with open(info_file_path, 'r', encoding='utf-8') as f:
                # قراءة المحتوى كنص أولاً
                content = f.read()
                
                # استبدال علامة = بـ : لجعل الملف JSON صحيح
                content = content.replace('"FIRST_SLIDE_FONT" =', '"FIRST_SLIDE_FONT":')
                content = content.replace('"REST_SLIDES_FONT" =', '"REST_SLIDES_FONT":')
                content = content.replace('"AR_FIRST_SLIDE_FONT" =', '"AR_FIRST_SLIDE_FONT":')
                content = content.replace('"AR_REST_SLIDES_FONT" =', '"AR_REST_SLIDES_FONT":')
                
                # إزالة الفواصل المزدوجة إذا وجدت
                content = content.replace('""', '"')
                
                # تحويل إلى JSON
                data = json.loads(content)
                
                en_story_name = data.get('en')
                ar_story_name = data.get('ar')
                resolution_slides = data.get('resolution_slides')
                first_slide_font = data.get('FIRST_SLIDE_FONT')
                rest_slides_font = data.get('REST_SLIDES_FONT')
                ar_first_slide_font = data.get('AR_FIRST_SLIDE_FONT')
                ar_rest_slides_font = data.get('AR_REST_SLIDES_FONT')
                
        except Exception as e:
            print(f"⚠️ Error reading info.txt: {e}")
            
    return en_story_name, ar_story_name, resolution_slides, first_slide_font, rest_slides_font, ar_first_slide_font, ar_rest_slides_font


def get_image_dimensions(image_path):
    """
    الحصول على أبعاد الصورة
    
    Args:
        image_path: مسار الصورة
    
    Returns:
        (width, height) أو None في حالة الفشل
    """
    if not os.path.exists(image_path):
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w = img.shape[:2]
    return w, h


def compare_images_similarity(image1_path, image2_path):
    """
    مقارنة التشابه بين صورتين باستخدام SSIM
    
    Args:
        image1_path: مسار الصورة الأولى (أو numpy array)
        image2_path: مسار الصورة الثانية (أو numpy array)
    
    Returns:
        float: نسبة التشابه من 0.0 إلى 1.0 (1.0 = متطابقة تماماً)
               أو None في حالة الفشل
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # قراءة الصور
        if isinstance(image1_path, str):
            img1 = cv2.imread(image1_path)
        else:
            img1 = image1_path
            
        if isinstance(image2_path, str):
            img2 = cv2.imread(image2_path)
        else:
            img2 = image2_path
        
        if img1 is None or img2 is None:
            return None
        
        # تحويل للرمادي لتسريع المقارنة
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # التأكد من أن الصور بنفس الحجم
        if gray1.shape != gray2.shape:
            # تغيير حجم الصورة الثانية لتطابق الأولى
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # حساب SSIM
        similarity_index = ssim(gray1, gray2)
        
        return similarity_index
        
    except ImportError:
        print("   ⚠️  scikit-image not installed. Install with: pip install scikit-image")
        return None
    except Exception as e:
        print(f"   ⚠️  Error comparing images: {str(e)}")
        return None


def crop_face_only(image_path, output_path, padding=2):
    """
    قص الصورة على الوجه فقط باستخدام Haar Cascade
    مع محاولة تدوير الصورة إذا فشل الاكتشاف
    
    Args:
        image_path: مسار الصورة الأصلية
        output_path: مسار حفظ الصورة المقصوصة
        padding: مقدار المساحة حول الوجه (2 = 200% من حجم الوجه)
    
    Returns:
        str: مسار الصورة المقصوصة، أو None في حالة الفشل
    """
    import numpy as np
    
    def rotate_image(image, angle):
        """تدوير الصورة بزاوية معينة"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # حساب الأبعاد الجديدة بعد التدوير
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # تعديل مصفوفة التدوير للحفاظ على الصورة كاملة
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255, 255, 255))
        return rotated
    
    def detect_and_crop(img, angle_name="الأصلية"):
        """محاولة اكتشاف وقص الوجه"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        
        # اكتشاف الوجوه
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        # أخذ أول وجه (الأكبر عادة)
        x, y, width, height = faces[0]
        
        # إضافة padding حول الوجه
        pad_w = int(width * (padding - 1) / 2)
        pad_h = int(height * (padding - 1) / 2)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + width + pad_w)
        y2 = min(h, y + height + pad_h)
        
        # قص الوجه
        cropped = img[y1:y2, x1:x2]
        print(f"   ✂️  الوجه اتقص من ({x1},{y1}) لـ ({x2},{y2}) - الزاوية: {angle_name}")
        
        return cropped
    
    try:
        # قراءة الصورة الأصلية
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ فشل قراءة الصورة: {image_path}")
            return None
        
        # المحاولة 1: الصورة الأصلية
        print("   🔍 محاولة 1: الصورة الأصلية...")
        cropped = detect_and_crop(img, "الأصلية")
        
        if cropped is not None:
            cv2.imwrite(output_path, cropped)
            return output_path
        
        # المحاولة 2: دوران 45° مع عقارب الساعة
        print("   🔍 محاولة 2: دوران 45° مع عقارب الساعة...")
        rotated_cw = rotate_image(img, -45)  # سالب = مع عقارب الساعة
        cropped = detect_and_crop(rotated_cw, "45° مع عقارب الساعة")
        
        if cropped is not None:
            cv2.imwrite(output_path, cropped)
            return output_path
        
        # المحاولة 3: دوران 45° عكس عقارب الساعة
        print("   🔍 محاولة 3: دوران 45° عكس عقارب الساعة...")
        rotated_ccw = rotate_image(img, 45)  # موجب = عكس عقارب الساعة
        cropped = detect_and_crop(rotated_ccw, "45° عكس عقارب الساعة")
        
        if cropped is not None:
            cv2.imwrite(output_path, cropped)
            return output_path
        
        # إذا فشلت كل المحاولات
        print("   ⚠️  مفيش وجه اتلقى في كل المحاولات، هستخدم الصورة الأصلية")
        cv2.imwrite(output_path, img)
        return output_path
        
    except Exception as e:
        print(f"   ❌ خطأ في قص الوجه: {str(e)}")
        return None
