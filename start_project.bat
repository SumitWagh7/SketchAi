@echo off
echo =========================================
echo Starting SketchAI Full Stack Environment
echo =========================================

echo 1. Starting Java Spring Boot Backend...
start cmd /k "mvn spring-boot:run"

echo 2. Starting Python AI Engine...
if exist "venv\Scripts\activate.bat" (
    echo Activating Virtual Environment...
    start cmd /k "call venv\Scripts\activate.bat && python ai_engine.py"
) else (
    echo Venv not found. Running Python globally...
    start cmd /k "python ai_engine.py"
)

echo =========================================
echo Both servers are starting in separate windows.
echo Frontend will be available at: http://localhost:8080
echo AI Engine will be available at: http://localhost:8000
echo =========================================
