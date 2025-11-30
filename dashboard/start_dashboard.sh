#!/bin/bash
# Start Dashboard with proper error handling

echo "🚀 Starting MedInsight Dashboard..."
echo "==================================="

cd "$(dirname "$0")"

# Activate venv
if [ -d "../backend/venv" ]; then
    source ../backend/venv/bin/activate
else
    echo "❌ Backend virtual environment not found!"
    exit 1
fi

# Kill any existing Streamlit
pkill -f streamlit 2>/dev/null
sleep 1

# Check if port is free
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is in use. Trying to free it..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo "✅ Starting Streamlit on http://localhost:8501"
echo ""
echo "📝 Tips:"
echo "   - If you see a white page, try:"
echo "     1. Hard refresh: Cmd+Shift+R"
echo "     2. Clear Safari cache: Cmd+Option+E"
echo "     3. Check terminal for errors"
echo ""
echo "Press Ctrl+C to stop"
echo "==================================="
echo ""

streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless false \
    --browser.gatherUsageStats false
