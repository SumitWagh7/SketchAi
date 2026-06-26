@echo off
echo =========================================
echo Starting SketchAI Local AI Engine...
echo =========================================

if not exist "venv" (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)

echo Activating virtual environment...
call venv\Scripts\activate


echo Starting AI Server on port 8000...
python ai_engine.py

pause
