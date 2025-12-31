@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
python -m uvicorn Api.api_story:app --host 0.0.0.0 --port 8000
pause
