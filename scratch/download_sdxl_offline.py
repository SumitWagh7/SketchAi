import os
import shutil
import sys

local_dir = "models/stable_diffusion/sdxl_turbo"
os.makedirs(local_dir, exist_ok=True)

# Clear locks
locks_dir = "models/stable_diffusion/.locks"
if os.path.exists(locks_dir):
    print("Clearing cache locks...")
    try:
        shutil.rmtree(locks_dir)
        print("Locks cleared.")
    except Exception as e:
        print(f"Failed to clear locks: {e}")

print(f"Starting snapshot download of stabilityai/sdxl-turbo to {local_dir}...")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="stabilityai/sdxl-turbo",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.ckpt", "*.onnx", "*.pb"]  # Prefer safetensors to save bandwidth
    )
    print("SUCCESS: SDXL Turbo model downloaded via snapshot_download.")
except Exception as e:
    print(f"snapshot_download failed: {e}. Trying fallback download via diffusers from_pretrained...")
    try:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe.save_pretrained(local_dir)
        print("SUCCESS: SDXL Turbo model downloaded via fallback save_pretrained.")
    except Exception as ex:
        print(f"ERROR: Both download methods failed. Root cause: {ex}")
        sys.exit(1)
