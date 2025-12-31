#!/bin/bash
# Simple test script to start dashboard and see output

cd dashboard
source ../backend/venv/bin/activate

echo "🚀 Starting Streamlit Dashboard..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "If the browser doesn't open automatically, manually navigate to:"
echo "   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
