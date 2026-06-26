import os
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

MODEL_DIR = os.path.join(os.getcwd(), "models")
cache_dir = os.path.join(MODEL_DIR, "stable_diffusion")
os.makedirs(cache_dir, exist_ok=True)

print("Downloading SDXL Turbo model to:", cache_dir)
try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir,
        use_safetensors=True
    )
    print("SDXL Turbo model downloaded and cached successfully!")
except Exception as e:
    print("Failed to download SDXL Turbo:", str(e))
