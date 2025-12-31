#!/bin/bash
# Stop MedInsight Backend and Dashboard

echo "🛑 Stopping MedInsight Services..."
echo ""

# Stop Backend (port 8000)
BACKEND_PID=$(lsof -ti :8000 2>/dev/null)
if [ ! -z "$BACKEND_PID" ]; then
    echo "🔧 Stopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    sleep 2
    # Force kill if still running
    if lsof -ti :8000 > /dev/null 2>&1; then
        kill -9 $BACKEND_PID 2>/dev/null
    fi
    echo "✅ Backend stopped"
else
    echo "⚠️  No backend process found on port 8000"
fi

# Stop Dashboard (port 8501)
DASHBOARD_PID=$(lsof -ti :8501 2>/dev/null)
if [ ! -z "$DASHBOARD_PID" ]; then
    echo "📊 Stopping Dashboard (PID: $DASHBOARD_PID)..."
    kill $DASHBOARD_PID 2>/dev/null
    sleep 2
    # Force kill if still running
    if lsof -ti :8501 > /dev/null 2>&1; then
        kill -9 $DASHBOARD_PID 2>/dev/null
    fi
    echo "✅ Dashboard stopped"
else
    echo "⚠️  No dashboard process found on port 8501"
fi

# Also kill by process name (backup method)
pkill -f "uvicorn app.main:app" 2>/dev/null && echo "✅ Killed uvicorn processes"
pkill -f streamlit 2>/dev/null && echo "✅ Killed streamlit processes"

# Clean up PID files
rm -f backend.pid dashboard.pid 2>/dev/null

echo ""
echo "✅ All services stopped"

