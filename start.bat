@echo off
echo =========================================
echo Starting SketchAI Servers...
echo =========================================

echo Starting Python AI Engine...
start cmd /k "call venv\Scripts\activate && python ai_engine.py"

echo Starting Spring Boot Server...
start cmd /k "mvn spring-boot:run"

echo Both servers are starting in new windows.
echo Once Spring Boot is ready, go to http://localhost:8080
pause
