import os
import time
import numpy as np
from PIL import Image
import torch

# Test imports and GPU availability
print("GPU CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Verify SimpleLama
print("\n--- Testing SimpleLama (Object Removal) ---")
os.environ["LAMA_MODEL"] = "models/object_removal/big-lama.pt"
try:
    from simple_lama_inpainting import SimpleLama
    start_time = time.perf_counter()
    lama = SimpleLama(device=device)
    print(f"SimpleLama loaded in {time.perf_counter() - start_time:.2f}s")
    
    # Create a dummy image and mask
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    mask = Image.new("L", (256, 256), 0)
    # Draw a white square in the mask
    for x in range(100, 150):
        for y in range(100, 150):
            mask.putpixel((x, y), 255)
            
    start_time = time.perf_counter()
    out = lama(img, mask)
    print(f"SimpleLama processed image in {time.perf_counter() - start_time:.2f}s")
    print("SimpleLama output image size:", out.size)
except Exception as e:
    print("SimpleLama test failed:", str(e))

# 2. Verify Real-ESRGAN
print("\n--- Testing Real-ESRGAN (Upscaling) ---")
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    model_path = "models/upscale/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        raise Exception(f"Model not found at {model_path}")
        
    start_time = time.perf_counter()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False,
        device=device
    )
    print(f"Real-ESRGAN loaded in {time.perf_counter() - start_time:.2f}s")
    
    # Create a dummy image
    img_np = np.zeros((128, 128, 3), dtype=np.uint8)
    
    start_time = time.perf_counter()
    out, _ = upscaler.enhance(img_np, outscale=2)
    print(f"Real-ESRGAN 2X upscale processed in {time.perf_counter() - start_time:.2f}s")
    print("Real-ESRGAN 2X output size:", out.shape)
    
    start_time = time.perf_counter()
    out, _ = upscaler.enhance(img_np, outscale=4)
    print(f"Real-ESRGAN 4X upscale processed in {time.perf_counter() - start_time:.2f}s")
    print("Real-ESRGAN 4X output size:", out.shape)
except Exception as e:
    print("Real-ESRGAN test failed:", str(e))
