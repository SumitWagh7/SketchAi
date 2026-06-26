import os
import torch
import numpy as np
from PIL import Image

os.environ["LAMA_MODEL"] = "models/object_removal/big-lama.pt"

# Create a sample image (e.g. gradient)
w, h = 250, 250
img_np = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        img_np[y, x] = [x % 256, y % 256, (x+y) % 256]
img = Image.fromarray(img_np)

# Create a mask with a square in the middle
mask = Image.new("L", (w, h), 0)
for y in range(100, 150):
    for x in range(100, 150):
        mask.putpixel((x, y), 255)

from simple_lama_inpainting import SimpleLama

# Test on CPU
print("--- Testing on CPU ---")
try:
    lama_cpu = SimpleLama(device="cpu")
    out_cpu = lama_cpu(img, mask)
    out_cpu_np = np.array(out_cpu)
    crop_cpu = out_cpu_np[100:150, 100:150]
    mean_val = np.mean(crop_cpu)
    std_val = np.std(crop_cpu)
    print(f"CPU Inpainted Region: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
    print("Any NaNs in CPU output:", np.isnan(out_cpu_np).any())
    print("Is the inpainted region completely black?", (crop_cpu == 0).all())
except Exception as e:
    print("CPU Test Failed:", str(e))

# Test on GPU if available
if torch.cuda.is_available():
    print("\n--- Testing on GPU (CUDA) ---")
    try:
        lama_gpu = SimpleLama(device="cuda")
        out_gpu = lama_gpu(img, mask)
        out_gpu_np = np.array(out_gpu)
        crop_gpu = out_gpu_np[100:150, 100:150]
        mean_val = np.mean(crop_gpu)
        std_val = np.std(crop_gpu)
        print(f"GPU Inpainted Region: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        print("Any NaNs in GPU output:", np.isnan(out_gpu_np).any())
        print("Is the inpainted region completely black?", (crop_gpu == 0).all())
    except Exception as e:
        print("GPU Test Failed:", str(e))
else:
    print("\nGPU (CUDA) is not available.")
