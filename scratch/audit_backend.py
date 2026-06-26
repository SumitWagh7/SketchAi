import os
import sys
import time
import gc
import json
import traceback
from PIL import Image
import numpy as np

# Add parent directory to sys.path to import model_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We will collect all results in this dict
audit_results = {
    "missing_dependencies": [],
    "broken_imports": [],
    "missing_models": [],
    "failed_model_loads": [],
    "endpoint_failures": [],
    "gpu_info": {},
    "model_devices": {},
    "models_verified": {}
}

# 1. Dependency check
print("=== 1. Checking Python Dependencies ===")
required_imports = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "rembg": "rembg",
    "onnxruntime": "onnxruntime",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "numpy": "numpy",
    "realesrgan": "realesrgan",
    "basicsr": "basicsr",
    "facexlib": "facexlib",
    "gfpgan": "gfpgan",
    "python-multipart": "multipart",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "requests": "requests",
    "pydantic": "pydantic",
    "simple-lama-inpainting": "simple_lama_inpainting",
    "fire": "fire"
}

for pkg, imp_name in required_imports.items():
    try:
        __import__(imp_name)
        print(f"[OK] {pkg} -> imported successfully as {imp_name}")
    except ImportError as e:
        print(f"[FAIL] {pkg} (import name: {imp_name}) failed to import: {e}")
        audit_results["broken_imports"].append(f"{pkg} ({imp_name}): {str(e)}")
        # Check if installed but broken, or just missing
        audit_results["missing_dependencies"].append(pkg)

# 2. GPU Detection
print("\n=== 2. Checking GPU / CUDA Configuration ===")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    audit_results["gpu_info"]["cuda_available"] = cuda_available
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        audit_results["gpu_info"]["gpu_name"] = gpu_name
        audit_results["gpu_info"]["vram_gb"] = f"{vram_gb:.2f} GB"
        print(f"CUDA Available: Yes")
        print(f"GPU Name: {gpu_name}")
        print(f"VRAM: {vram_gb:.2f} GB")
    else:
        audit_results["gpu_info"]["gpu_name"] = "N/A"
        audit_results["gpu_info"]["vram_gb"] = "N/A"
        print("CUDA Available: No (Running on CPU)")
except Exception as e:
    print(f"Error checking GPU: {e}")
    audit_results["gpu_info"]["cuda_available"] = False
    audit_results["gpu_info"]["error"] = str(e)

# 3. Model file paths check
print("\n=== 3. Checking Local Model File Paths ===")
MODEL_DIR = os.path.join(os.getcwd(), "models")
models_to_check = {
    "SDXL Turbo": os.path.join(MODEL_DIR, "stable_diffusion", "models--stabilityai--sdxl-turbo"),
    "Studio Ghibli LoRA": os.path.join(MODEL_DIR, "stable_diffusion", "StudioGhibli.safetensors"),
    "MODNet": os.path.join(MODEL_DIR, "background", "matting_modnet_portrait.pth"),
    "BiRefNet": os.path.join(MODEL_DIR, "background", "birefnet-general.onnx"),
    "SimpleLama": os.path.join(MODEL_DIR, "object_removal", "big-lama.pt"),
    "RealESRGAN x4 / x2 (Weights)": os.path.join(MODEL_DIR, "upscale", "RealESRGAN_x4plus.pth"),
    "GFPGAN v1.4 (Face Enhancer)": os.path.join(MODEL_DIR, "upscale", "GFPGANv1.4.pth")
}

for model_name, path in models_to_check.items():
    exists = os.path.exists(path)
    audit_results["models_verified"][model_name] = {
        "path": path,
        "exists": exists
    }
    if exists:
        print(f"[OK] {model_name} found at: {path}")
    else:
        print(f"[FAIL] {model_name} NOT found at: {path}")
        audit_results["missing_models"].append(model_name)

# 4. Active Device logging by loading models one-by-one to avoid VRAM OOM
print("\n=== 4. Testing Model Loading and Active Device ===")
if not audit_results["broken_imports"]:
    from model_manager import model_manager, gpu_manager
    
    # Define verification function for each model type
    def test_model_load(name, load_fn, device_check_fn):
        print(f"Loading {name}...")
        start = time.perf_counter()
        try:
            model = load_fn()
            elapsed = time.perf_counter() - start
            device = device_check_fn(model)
            audit_results["model_devices"][name] = {
                "status": "Loaded successfully",
                "load_time_sec": f"{elapsed:.2f}s",
                "device": str(device)
            }
            print(f"[OK] Loaded {name} on device '{device}' in {elapsed:.2f}s")
            
            # Clean up immediately
            del model
            gpu_manager.flush_memory()
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"[FAIL] Failed to load {name}: {e}")
            tb_str = traceback.format_exc()
            traceback.print_exc()
            audit_results["failed_model_loads"].append(name)
            audit_results["model_devices"][name] = {
                "status": f"Load failed: {str(e)}",
                "load_time_sec": f"{elapsed:.2f}s",
                "device": "N/A",
                "traceback": tb_str
            }

    # MODNet
    test_model_load(
        "MODNet",
        model_manager.get_modnet,
        lambda m: next(m.parameters()).device
    )

    # BiRefNet (rembg birefnet-general session)
    def load_birefnet():
        import rembg
        return rembg.new_session("birefnet-general", providers=gpu_manager.get_onnx_providers())
    
    test_model_load(
        "BiRefNet",
        load_birefnet,
        lambda s: s.inner_session.get_providers() if hasattr(s, "inner_session") else "ONNX CPU/GPU"
    )

    # SimpleLama
    test_model_load(
        "SimpleLama",
        model_manager.get_simple_lama,
        lambda m: m.device
    )

    # RealESRGAN
    test_model_load(
        "RealESRGAN x4/x2",
        model_manager.get_upscaler,
        lambda m: m.device
    )

    # SDXL Turbo
    test_model_load(
        "SDXL Turbo",
        model_manager.get_ghibli_pipe,
        lambda pipe: pipe.device
    )

else:
    print("Skipping model loading tests due to broken imports.")
    for name in ["MODNet", "BiRefNet", "SimpleLama", "RealESRGAN x4/x2", "SDXL Turbo"]:
        audit_results["failed_model_loads"].append(name)
        audit_results["model_devices"][name] = {
            "status": "Skipped (broken imports)",
            "load_time_sec": "0s",
            "device": "N/A"
        }

# Write results to json for endpoint testing script to read
with open("scratch/audit_results_temp.json", "w") as f:
    json.dump(audit_results, f, indent=4)

print("\nBackend audit verification complete. Temporary results saved to scratch/audit_results_temp.json.")
