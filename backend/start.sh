#!/bin/bash
# FastAPI 服务启动脚本
cd "$(dirname "$0")"
cd ..
pip install fastapi uvicorn python-multipart -q
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload