#!/bin/bash
# Startup script for Streamlit app

echo "🚀 Starting Oncology Drug Response Prediction App..."
echo ""
echo "📍 Access the app at: http://localhost:8501"
echo "⏹️  To stop: Press Ctrl+C"
echo ""

CUDA_VISIBLE_DEVICES=1 streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
