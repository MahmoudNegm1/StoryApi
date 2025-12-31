# -*- coding: utf-8 -*-
"""
📄 PDF Generator Module
تحويل الصور إلى PDF باستخدام PIL
"""

import cv2
from PIL import Image


def create_pdf_from_images(images_list, output_path, use_parallel=None):
    """
    إنشاء PDF من قائمة الصور باستخدام PIL
    
    Args:
        images_list: قائمة الصور (OpenCV format - BGR)
        output_path: مسار ملف PDF الناتج
        use_parallel: غير مستخدم (للتوافق مع الكود القديم)
    
    Returns:
        bool: True إذا نجح الإنشاء، False إذا فشل
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
            # إنشاء خلفية بيضاء
            rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
            # لصق الصورة مع استخدام قناة الشفافية كـ mask
            rgb_img.paste(pil_img, mask=pil_img.split()[3])
            pil_images.append(rgb_img)
        else:
            # تحويل إلى RGB
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

