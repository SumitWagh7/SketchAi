@echo off
echo =========================================
echo SketchAI Local Environment Installer
echo =========================================

call setup_env.bat

echo Checking for NVIDIA GPU (CUDA)...
python -c "import torch; print(torch.cuda.is_available())" > temp_gpu.txt
set /p GPU_AVAIL=<temp_gpu.txt
del temp_gpu.txt

if "%GPU_AVAIL%"=="True" (
    echo NVIDIA GPU detected. Installing CUDA-compatible PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected. Installing CPU version of PyTorch...
    pip install torch torchvision torchaudio
)

echo Installing required packages...
pip install diffusers==0.27.2 transformers==4.39.3 accelerate==0.28.0 huggingface_hub==0.22.2
pip install rembg onnxruntime-gpu opencv-python pillow numpy scipy
pip install fastapi uvicorn python-multipart requests
pip install realesrgan basicsr facexlib gfpgan
pip install sentencepiece safetensors

echo Validating and downloading models...
python model_downloader.py
python check_models.py

echo Installation Complete!
