import os
import sys
import torch
from PIL import Image

# Add root folder to sys.path to import model_manager
sys.path.append(os.getcwd())

from model_manager import model_manager, gpu_manager

def run_experiment():
    print("=====================================================")
    print("Ghibli Experiment Run (Memory Optimized)")
    print("=====================================================")

    # 1. Load the original male face image
    input_path = "scratch/debug_input_to_pipeline.png"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist!")
        sys.exit(1)
        
    init_image = Image.open(input_path).convert("RGB")
    print(f"Loaded input image: {input_path} (size: {init_image.size})")

    # 2. Get the SDXL Turbo pipeline with 4GB GPU optimizations but NO LoRA loaded yet
    print("\n--- Initializing SDXL Turbo Pipeline (No LoRA yet) ---")
    cache_dir = "models/stable_diffusion"
    from diffusers import StableDiffusionXLImg2ImgPipeline
    
    print("Loading pipeline with sequential CPU offload...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        use_safetensors=True,
        variant="fp16",
        low_cpu_mem_usage=True
    )
    
    # Apply 4GB VRAM memory saving options
    print("Enabling sequential CPU offloading and VAE/attention slicing...")
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    
    # 3. Test 1: LoRA Disabled, Strength = 0.50
    print("\n--- Running Test 1: LoRA Disabled, Strength = 0.50 ---")
    prompt = "sunset, Studio Ghibli style, Redmond Ghibli style, anime art, watercolor style, masterpiece, high quality"
    print(f"Prompt: {prompt}")
    
    with torch.inference_mode():
        out_no_lora = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.50,
            guidance_scale=0.0,
            num_inference_steps=4
        ).images[0]
        
    out_no_lora_path = "scratch/output_lora_disabled.png"
    out_no_lora.save(out_no_lora_path)
    print(f"Saved Test 1 output to: {out_no_lora_path}")

    # 4. Load LoRA weights
    lora_path = os.path.join(cache_dir, "StudioGhibli.safetensors")
    print(f"\n--- Loading LoRA weights from {lora_path} ---")
    pipe.load_lora_weights(lora_path)
    print("LoRA weights loaded successfully.")

    # 5. Test 2: LoRA Enabled, Strength = 0.25
    print("\n--- Running Test 2: LoRA Enabled, Strength = 0.25 ---")
    with torch.inference_mode():
        out_strength_025 = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.25,
            guidance_scale=0.0,
            num_inference_steps=4
        ).images[0]
        
    out_strength_025_path = "scratch/output_strength_025.png"
    out_strength_025.save(out_strength_025_path)
    print(f"Saved Test 2 output to: {out_strength_025_path}")

    # 6. Test 3: LoRA Enabled, Strength = 0.50
    print("\n--- Running Test 3: LoRA Enabled, Strength = 0.50 ---")
    with torch.inference_mode():
        out_strength_050 = pipe(
            prompt=prompt,
            image=init_image,
            strength=0.50,
            guidance_scale=0.0,
            num_inference_steps=4
        ).images[0]
        
    out_strength_050_path = "scratch/output_strength_050.png"
    out_strength_050.save(out_strength_050_path)
    print(f"Saved Test 3 output to: {out_strength_050_path}")

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    run_experiment()
