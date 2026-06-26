import os
import sys
import torch
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLImg2ImgPipeline

def main():
    MODEL_DIR = os.path.join(os.getcwd(), "models")
    cache_dir = os.path.join(MODEL_DIR, "stable_diffusion")
    
    print("=====================================================")
    print("Sequential SDXL-Turbo Downloader for SketchAI")
    print(f"Target cache directory: {cache_dir}")
    print("=====================================================")
    
    allow_patterns = [
        "*.json",
        "*.txt",
        "*model.fp16.safetensors",
        "*diffusion_pytorch_model.fp16.safetensors"
    ]
    
    print(f"\n[Step 1] Downloading SDXL-Turbo repo sequentially...")
    try:
        snapshot_download(
            repo_id="stabilityai/sdxl-turbo",
            cache_dir=cache_dir,
            max_workers=1,
            allow_patterns=allow_patterns,
            local_files_only=False
        )
        print("[Step 1 SUCCESS] Sequential snapshot download completed.")
    except Exception as e:
        print(f"[Step 1 ERROR] Failed during snapshot download: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Step 2: Verify offline load works
    print("\n[Step 2] Verifying offline model loading...")
    try:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True
        )
        print("[SUCCESS] SDXL Turbo model successfully verified and loaded in OFFLINE mode!")
    except Exception as e_verify:
        print(f"[ERROR] Offline verification failed: {e_verify}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
