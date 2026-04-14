#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "  Rulebook Dashboard Startup"
echo "=================================================="
echo ""

# Check if backend dependencies are installed
if [ ! -d "backend/venv" ] && ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing backend dependencies..."
    cd backend
    pip install -r requirements.txt
    cd ..
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "🚀 Starting servers..."
echo ""

# Start backend
echo "Starting backend (port 5000)..."
cd backend
python server.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend
echo "Starting frontend (port 5173)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "=================================================="
echo "✅ Both servers started!"
echo "=================================================="
echo ""
echo "📊 Dashboard:  http://localhost:5173"
echo "🔌 Backend:    http://127.0.0.1:5000"
echo "💾 Database:   rulebook_pipeline.db"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
