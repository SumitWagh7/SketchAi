import os
import sys

MODEL_DIR = os.path.join(os.getcwd(), "models")

def check_models():
    print("Validating required models...")
    required_files = [
        "u2net.onnx",
        "RealESRGAN_x4plus.pth",
        "GFPGANv1.4.pth"
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            missing.append(f)
            
    if missing:
        print(f"Missing models: {missing}")
        print("Please run install.bat or model_downloader.py first.")
        sys.exit(1)
    
    print("All required local weights found!")

if __name__ == "__main__":
    check_models()
