#!/bin/bash
# Start MedInsight Dashboard (run from root directory)

echo "🚀 Starting MedInsight Dashboard..."
echo "==================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/dashboard"

# Check if backend venv exists
if [ -d "../backend/venv" ]; then
    echo "🔧 Activating backend virtual environment..."
    source ../backend/venv/bin/activate
else
    echo "⚠️  Backend virtual environment not found. Please run start_backend.sh first."
    exit 1
fi

# Install streamlit if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📥 Installing Streamlit..."
    pip install streamlit plotly requests pandas
fi

# Start Streamlit
echo "✅ Starting Streamlit dashboard on http://localhost:8501"
echo ""
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
