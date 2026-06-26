import os
import shutil
import sys

# Clear locks
locks_dir = "models/stable_diffusion/.locks"
if os.path.exists(locks_dir):
    print("Clearing cache locks...")
    try:
        shutil.rmtree(locks_dir)
        print("Locks cleared.")
    except Exception as e:
        print(f"Failed to clear locks: {e}")

print("Importing diffusers & torch...")
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

cache_dir = "models/stable_diffusion"
os.makedirs(cache_dir, exist_ok=True)

print(f"Starting SDXL Turbo download to {cache_dir}...")
try:
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir,
        use_safetensors=True,
        local_files_only=False
    )
    print("SUCCESS: SDXL Turbo has been successfully downloaded and cached!")
except Exception as e:
    print(f"ERROR: Failed to download SDXL Turbo: {e}")
    sys.exit(1)
