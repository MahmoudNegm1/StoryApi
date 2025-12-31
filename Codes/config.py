# -*- coding: utf-8 -*-
"""
🔧 Configuration Module
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================== WaveSpeed API Configuration ==================
WAVESPEED_API_KEY = "45ec4b8a031581b78369f3b80bf4f2eb7a30e2b73efefc67ac91cd1a329f0f8d"
WAVESPEED_API_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/image-head-swap"
WAVESPEED_OUTPUT_FORMAT = "jpeg"
WAVESPEED_SYNC_MODE = True
WAVESPEED_TIMEOUT = 60 * 3  # 3 minutes timeout

# ================== ImgBB Configuration ==================
IMGBB_API_KEY = "3b4cd701f4471dee2c2c67a0d13d711e"
IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"

# ================== Font Paths ==================
EN_FIRST_SLIDE_FONT = os.path.join(BASE_DIR, "Fonts/english fonts/Irish_Grover/IrishGrover-Regular.ttf")
EN_REST_SLIDES_FONT = os.path.join(BASE_DIR, "Fonts/english fonts/static/PlayfairDisplay-Bold.ttf")
AR_FIRST_SLIDE_FONT = os.path.join(BASE_DIR, "Fonts/arabic fonts/alfont_com_KAF-GULZAR-PC.otf")
AR_REST_SLIDES_FONT = os.path.join(BASE_DIR, "Fonts/arabic fonts/alfont_com_Al-Haroni-Mashnab-Salawat.ttf")

# ================== Folder Paths ==================
STORIES_FOLDER = os.path.join(BASE_DIR, "Stories")
CHARACTERS_FOLDER = os.path.join(BASE_DIR, "characters")
RESULT_FOLDER = os.path.join(BASE_DIR, "Result")
TEMP_CROPPED_FOLDER = os.path.join(BASE_DIR, "temp_cropped_faces")

# ================== Text Rendering ==================
ENABLE_TEXT_SHADOW = True
TEXT_SHADOW_STYLE = "2px 2px 4px rgba(0, 0, 0, 0.7)"



# ================== Processing Settings ==================
HEAD_SWAP_DELAY = 0.2  # Delay between API calls in seconds (Reduced from 1s)
RETRY_DELAY = 0.5      # Delay before retrying failed API calls
MAX_RETRIES = 2        # Maximum number of retries for API calls
SIMILARITY_THRESHOLD = 0.97  # Threshold for considering face unchanged (0.0 to 1.0)

# ================== Parallel Processing Settings ==================
API_WORKERS = 3       # Number of simultaneous API calls
UPLOAD_WORKERS = 5    # Number of simultaneous image uploads
USE_PARALLEL_TEXT_PROCESSING = True  # Enable parallel text rendering
from multiprocessing import cpu_count
MAX_TEXT_WORKERS = max(1, cpu_count() - 1)  # Number of parallel workers for text processing



