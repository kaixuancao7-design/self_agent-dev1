@echo off
cd /d "%~dp0"
cd ..
pip install fastapi uvicorn python-multipart -q
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload