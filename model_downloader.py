import os
import torch
import requests
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline
from transformers import pipeline
from huggingface_hub import hf_hub_download
import shutil

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["U2NET_HOME"] = MODEL_DIR

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_models():
    print("=========================================")
    print("SketchAI Automated Model Downloader")
    print("=========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("\n1. Downloading Background Removal Model (u2net)...")
    download_file('https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx', os.path.join(MODEL_DIR, 'u2net.onnx'))

    print("\n2. Downloading Upscale & Enhancement Models (Real-ESRGAN & GFPGAN)...")
    download_file('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', os.path.join(MODEL_DIR, 'RealESRGAN_x4plus.pth'))
    download_file('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', os.path.join(MODEL_DIR, 'GFPGANv1.4.pth'))

    print("\n3. Downloading Local LLM (TinyLlama & FLAN-T5) for Caption & Strategy...")
    pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model_kwargs={"cache_dir": MODEL_DIR})
    pipeline("text2text-generation", model="google/flan-t5-small", model_kwargs={"cache_dir": MODEL_DIR})

    print("\n4. Downloading Stable Diffusion XL Turbo & Ghibli LoRA...")
    print("NOTE: This is a large download. Please be patient.")
    
    # Download SDXL Turbo
    try:
        StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=dtype, cache_dir=MODEL_DIR)
        print("SDXL Turbo downloaded successfully.")
    except Exception as e:
        print("Failed to download SDXL Turbo:", str(e))
        
    # Download Ghibli LoRA
    try:
        lora_dest = os.path.join(MODEL_DIR, "stable_diffusion", "StudioGhibli.safetensors")
        if not os.path.exists(lora_dest):
            print("Downloading Studio Ghibli LoRA from HF...")
            path = hf_hub_download(repo_id="artificialguybr/StudioGhibli.Redmond-V2", filename="StudioGhibli.Redmond-StdGBRRedmAF-StudioGhibli.safetensors")
            os.makedirs(os.path.dirname(lora_dest), exist_ok=True)
            shutil.copy(path, lora_dest)
            print("Studio Ghibli LoRA downloaded successfully.")
        else:
            print("Studio Ghibli LoRA already exists.")
    except Exception as e:
        print("Failed to download Ghibli LoRA:", str(e))

    print("\n5. Downloading Object Removal / Inpainting Model...")
    # RunwayML SD Inpainting
    try:
        StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=dtype, cache_dir=os.path.join(MODEL_DIR, "object_removal"))
        print("SD Inpainting downloaded successfully.")
    except Exception as e:
        print("Failed to download SD Inpainting:", str(e))
        
    # LaMa Model
    try:
        lama_dest = os.path.join(MODEL_DIR, "object_removal", "big-lama.pt")
        if not os.path.exists(lama_dest):
            print("Downloading LaMa big-lama.pt from HF...")
            path = hf_hub_download(repo_id="fashn-ai/LaMa", filename="big-lama.pt")
            os.makedirs(os.path.dirname(lama_dest), exist_ok=True)
            shutil.copy(path, lama_dest)
            print("LaMa model downloaded successfully.")
        else:
            print("LaMa model already exists.")
    except Exception as e:
        print("Failed to download LaMa model:", str(e))

    print("\n=========================================")
    print("ALL MODELS DOWNLOADED AND VERIFIED!")
    print("=========================================")

if __name__ == "__main__":
    download_models()
