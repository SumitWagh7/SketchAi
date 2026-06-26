@echo off
echo =========================================
echo Setting up Python Environment
echo =========================================

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH! Please install Python 3.10+.
    pause
    exit /b
)

echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Environment ready.
