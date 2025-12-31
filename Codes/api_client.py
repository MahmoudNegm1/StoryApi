# -*- coding: utf-8 -*-
"""
🌐 API Client Module - WaveSpeed Integration
"""

import os
import base64
import requests

from Codes.config import (
    WAVESPEED_API_KEY,
    WAVESPEED_API_URL,
    WAVESPEED_OUTPUT_FORMAT,
    WAVESPEED_SYNC_MODE,
    WAVESPEED_TIMEOUT,
    IMGBB_API_KEY,
    IMGBB_UPLOAD_URL
)


def upload_to_imgbb(image_path):
    """
    رفع صورة على ImgBB والحصول على URL
    
    Args:
        image_path: مسار الصورة المحلية
        
    Returns:
        str: رابط الصورة على ImgBB أو None في حالة الفشل
    """
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read())
        
        response = requests.post(
            IMGBB_UPLOAD_URL,
            data={
                "key": IMGBB_API_KEY,
                "image": encoded
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "url" in result["data"]:
                return result["data"]["url"]
            else:
                print(f"   ❌ ImgBB response missing URL")
                return None
        else:
            print(f"   ❌ ImgBB upload failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ❌ ImgBB upload error: {str(e)}")
        return None


def perform_head_swap(target_image_path, face_image_path, output_filename, face_url_cached=None):
    """
    تنفيذ Head Swap باستخدام WaveSpeed API
    
    Args:
        target_image_path: مسار الصورة الأساسية (المشهد)
        face_image_path: مسار صورة الوجه (الشخصية)
        output_filename: مسار حفظ النتيجة
        face_url_cached: (اختياري) رابط صورة الوجه المرفوع مسبقاً على ImgBB
        
    Returns:
        str: مسار الملف المحفوظ أو None في حالة الفشل
    """
    try:
        # الخطوة 1: رفع الصورة الأساسية على ImgBB
        print(f"   ☁️  Uploading target image...")
        target_url = upload_to_imgbb(target_image_path)
        if not target_url:
            print(f"   ❌ Failed to upload target image")
            return None
        
        # الخطوة 2: استخدام face_url المحفوظ أو رفع صورة جديدة
        if face_url_cached:
            face_url = face_url_cached
            # لا نطبع شيء هنا لأن الرفع تم مسبقاً
        else:
            print(f"   ☁️  Uploading face image...")
            face_url = upload_to_imgbb(face_image_path)
            if not face_url:
                print(f"   ❌ Failed to upload face image")
                return None
        
        # الخطوة 3: إرسال الطلب لـ WaveSpeed API
        print(f"   🔄 Processing with WaveSpeed API...")
        
        payload = {
            "image": target_url,
            "face_image": face_url,
            "output_format": WAVESPEED_OUTPUT_FORMAT,
            "enable_sync_mode": WAVESPEED_SYNC_MODE
        }
        
        headers = {
            "Authorization": f"Bearer {WAVESPEED_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            WAVESPEED_API_URL,
            headers=headers,
            json=payload,
            timeout=WAVESPEED_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # استخراج رابط النتيجة
            if "data" in result and "outputs" in result["data"] and len(result["data"]["outputs"]) > 0:
                result_url = result["data"]["outputs"][0]
                
                # الخطوة 4: تحميل الصورة النهائية
                print(f"   ⬇️  Downloading result...")
                img_response = requests.get(result_url, timeout=WAVESPEED_TIMEOUT)
                
                if img_response.status_code == 200:
                    with open(output_filename, 'wb') as f:
                        f.write(img_response.content)
                    print(f"   ✅ Saved: {os.path.basename(output_filename)}")
                    return output_filename
                else:
                    print(f"   ❌ Failed to download result: {img_response.status_code}")
                    return None
            else:
                print(f"   ❌ No output in API response")
                return None
        else:
            print(f"   ❌ WaveSpeed API Error: {response.status_code}")
            if response.text:
                print(f"   📄 Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return None

