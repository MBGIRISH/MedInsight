#!/bin/bash
# Start both MedInsight Backend and Dashboard
# Run from root directory: ./start.sh

echo "🚀 Starting MedInsight System..."
echo "================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if ports are already in use and stop them
if lsof -ti :8000 > /dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use. Stopping existing backend..."
    kill $(lsof -ti :8000) 2>/dev/null
    sleep 2
    # Force kill if still running
    if lsof -ti :8000 > /dev/null 2>&1; then
        kill -9 $(lsof -ti :8000) 2>/dev/null
    fi
    echo "✅ Stopped existing backend"
fi

if lsof -ti :8501 > /dev/null 2>&1; then
    echo "⚠️  Port 8501 is already in use. Stopping existing dashboard..."
    kill $(lsof -ti :8501) 2>/dev/null
    sleep 2
    # Force kill if still running
    if lsof -ti :8501 > /dev/null 2>&1; then
        kill -9 $(lsof -ti :8501) 2>/dev/null
    fi
    echo "✅ Stopped existing dashboard"
fi

# Also kill by process name (backup)
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f streamlit 2>/dev/null
sleep 1
echo ""

# Check MongoDB
echo "📊 Checking MongoDB..."
if ! mongosh --eval "db.version()" > /dev/null 2>&1; then
    echo "⚠️  MongoDB not running. Attempting to start..."
    if command -v brew > /dev/null; then
        brew services start mongodb-community 2>/dev/null || echo "⚠️  Please start MongoDB manually"
    else
        echo "⚠️  Please start MongoDB manually"
    fi
    sleep 2
fi
echo "✅ MongoDB check complete"
echo ""

# Start Backend in background
echo "🔧 Starting Backend (FastAPI)..."
cd backend

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Start backend in background
nohup uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 > ../backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid
echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   📝 Logs: backend.log"
echo "   🌐 URL: http://localhost:8000"
echo "   📚 Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is responding
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️  Backend may still be starting..."
fi
echo ""

# Start Dashboard
echo "📊 Starting Dashboard (Streamlit)..."
cd ../dashboard

# Use backend's venv
source ../backend/venv/bin/activate

# Install streamlit if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📥 Installing Streamlit..."
    pip install streamlit plotly requests pandas > /dev/null 2>&1
fi

echo "✅ Dashboard starting..."
echo "   🌐 URL: http://localhost:8501"
echo ""
echo "📝 The dashboard will open automatically in your browser."
echo "   If it doesn't, manually open: http://localhost:8501"
echo ""

# Start dashboard (foreground so user can see output)
# Use --server.headless false to ensure browser opens
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless false \
    --browser.gatherUsageStats false

