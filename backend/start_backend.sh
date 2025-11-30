#!/bin/bash
# Start MedInsight Backend (run from backend directory)

echo "🚀 Starting MedInsight Backend..."
echo "================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# Check MongoDB
echo "🔍 Checking MongoDB..."
if ! mongosh --eval "db.version()" > /dev/null 2>&1; then
    echo "⚠️  MongoDB not running. Attempting to start..."
    if command -v brew > /dev/null; then
        brew services start mongodb-community 2>/dev/null || echo "⚠️  Please start MongoDB manually: brew services start mongodb-community"
    else
        echo "⚠️  Please start MongoDB manually: sudo systemctl start mongod"
    fi
    sleep 2
fi

# Start FastAPI server
echo ""
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo "📊 Health check: http://localhost:8000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

