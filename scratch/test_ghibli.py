import os
import sys
import time
import torch
from PIL import Image

# Add root folder to sys.path to import model_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import model_manager, gpu_manager

def main():
    print("=====================================================")
    print("SketchAI Ghibli Standalone Test")
    print("=====================================================")
    
    # 1. Prepare Input Image (using non-square size 640x360 to test proportional aspect ratio scaling)
    img_path = "scratch/benchmark_results/original_gradient.png"
    if os.path.exists(img_path):
        print(f"Loading input image: {img_path}")
        init_image = Image.open(img_path).convert("RGB").resize((640, 360))
    else:
        print("Input image not found. Creating dummy gradient image...")
        init_image = Image.new("RGB", (640, 360))
        for y in range(360):
            for x in range(640):
                init_image.putpixel((x, y), (x // 3, y // 2, (x + y) // 4))
    
    # Save input representation
    os.makedirs("scratch/test_results", exist_ok=True)
    input_save_path = "scratch/test_results/ghibli_input_test.png"
    init_image.save(input_save_path)
    print(f"Saved input image to: {input_save_path}")
    
    # 2. Get Ghibli Pipeline
    start_load = time.perf_counter()
    print("\n[STEP] Loading Ghibli Pipeline...")
    try:
        pipe = model_manager.get_ghibli_pipe()
        elapsed_load = time.perf_counter() - start_load
        print(f"[STEP SUCCESS] Pipeline loaded in {elapsed_load:.2f}s")
    except Exception as e:
        print(f"[STEP CRITICAL ERROR] Failed to load pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Generate image
    prompt = "Studio Ghibli style, Redmond Ghibli style, anime art, watercolor style, masterpiece, high quality, sunset landscape"
    print(f"\n[STEP] Generating Ghibli Art...")
    print(f"Prompt: {prompt}")
    print(f"Device: {gpu_manager.device}")
    
    start_gen = time.perf_counter()
    try:
        # Run pipeline
        output_image, meta = model_manager.run_ghibli_inference(
            prompt=prompt, 
            image=init_image, 
            strength=0.50, 
            guidance_scale=0.0,
            num_inference_steps=4
        )
        elapsed_gen = time.perf_counter() - start_gen
        print(f"[STEP SUCCESS] Ghibli Art generated in {elapsed_gen:.2f}s")
        
        # 4. Save and Verify
        output_path = "scratch/test_ghibli_output.png"
        output_image.save(output_path)
        print(f"Saved output image to: {output_path}")
        
        # Save alternative path as well if needed
        alt_path = "scratch/test_results/ghibli_output_test.png"
        output_image.save(alt_path)
        print(f"Saved output image to alternative path: {alt_path}")
        
        # Verify size
        print(f"Input size: {init_image.size}, Output size: {output_image.size}")
        if torch.cuda.is_available():
            vram_end = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"Active GPU VRAM usage at end: {vram_end:.1f} MB")
            print(f"Peak VRAM usage reported: {meta['peak_vram_mb']:.1f} MB")
            print(f"Sequential CPU offload used: {meta['cpu_offload_used']}")
            
        expected_w, expected_h = model_manager.get_proportional_size(init_image.size, gpu_manager.max_dimension)
        if output_image.size == (expected_w, expected_h):
            print(f"\n[VERIFICATION PASSED] Ghibli model works successfully! Output size is {output_image.size} as expected (matches VRAM classification proportional constraints).")
        else:
            print(f"\n[VERIFICATION FAILED] Output size is {output_image.size}, expected ({expected_w}, {expected_h})!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[STEP CRITICAL ERROR] Generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
