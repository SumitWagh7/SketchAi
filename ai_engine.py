import time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from rembg import remove

# Import the centralized service engine
from model_manager import (
    gpu_manager,
    model_manager,
    task_queue,
    ImageProcessor
)

app = FastAPI()

class AiRequest(BaseModel):
    prompt: Optional[str] = ""
    image: str # base64
    mask: Optional[str] = None # base64 for inpainting
    strength: Optional[float] = 0.60
    upscale_factor: Optional[int] = 2
    mode: Optional[str] = "balanced"

class CaptionRequest(BaseModel):
    prompt: str

# --- REST ENDPOINTS WIREUP TO QUEUES & MANAGERS ---

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": gpu_manager.device_name}

@app.post("/api/python/ghibli")
async def generate_ghibli(req: AiRequest):
    def task():
        start_time = time.perf_counter()
        print(f"[Request Received] /api/python/ghibli | Prompt: {req.prompt}")
        
        # Log telemetry details
        print(f"[Python API] Received image base64 length: {len(req.image)}")
        try:
            import hashlib
            import base64 as pybase64
            img_bytes = pybase64.b64decode(req.image.split(',')[-1])
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()
            print(f"[Python API] Decoded image size: {len(img_bytes)} bytes")
            print(f"[Python API] Input image SHA-256 hash: {sha256_hash}")
        except Exception as e:
            print(f"[Python API ERROR] Error calculating hash: {e}")

        # Load and verify input image
        print("[Ghibli Process] Loading input image...")
        init_image = ImageProcessor.base64_to_pil(req.image).convert("RGB")
        w, h = init_image.size
        print(f"[Python API] Decoded image dimensions W={w}, H={h}")
        max_dim = gpu_manager.max_dimension
        new_w, new_h = model_manager.get_proportional_size((w, h), max_dim)
        print(f"[Ghibli Process] Resizing input image from {w}x{h} to {new_w}x{new_h} based on VRAM limit {max_dim} (aspect ratio preserved)")
        init_image = init_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print("[Ghibli Process] Image loaded successfully.")
        
        prompt = req.prompt + ", Studio Ghibli style, Redmond Ghibli style, anime art, watercolor style, masterpiece, high quality"
        strength = req.strength if req.strength is not None else 0.50
        
        # SDXL Turbo: 4 steps, guidance_scale=0.0
        print(f"[Ghibli Process] Generation started (guidance_scale=0.0, strength={strength})...")
        image, meta = model_manager.run_ghibli_inference(
            prompt=prompt, 
            image=init_image, 
            strength=strength, 
            guidance_scale=0.0,
            num_inference_steps=4
        )
        print("[Ghibli Process] Generation completed successfully.")
        
        gpu_manager.flush_memory()
        elapsed = time.perf_counter() - start_time
        print(f"[Success] generate_ghibli took {elapsed:.2f}s")
        return {
            "success": True, 
            "image": ImageProcessor.pil_to_base64(image), 
            "execution_time_sec": elapsed,
            "peak_vram_mb": meta.get("peak_vram_mb"),
            "cpu_offload_used": meta.get("cpu_offload_used")
        }

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import sys
        import traceback
        print("[Ghibli Process ERROR] Failure in /api/python/ghibli:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/caption")
async def generate_caption(req: CaptionRequest):
    def task():
        start_time = time.perf_counter()
        print(f"[Request Received] /api/python/caption | Prompt: {req.prompt}")
        
        print("Model Manager: Loading TinyLlama...")
        pipe = model_manager.get_llm_pipe()
        
        prompt = f"<|system|>\nYou are an aesthetic art critic. Generate a poetic Instagram caption with hashtags for the following description.<|end|>\n<|user|>\n{req.prompt}<|end|>\n<|assistant|>\n"
        print("Generating caption text...")
        result = pipe(prompt, max_new_tokens=60, return_full_text=False)
        caption = result[0]['generated_text'].strip()
        
        elapsed = time.perf_counter() - start_time
        print(f"[Success] generate_caption took {elapsed:.2f}s")
        return {"success": True, "text": caption, "execution_time_sec": elapsed}

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/caption:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/strategy")
async def edit_strategy(req: CaptionRequest):
    def task():
        start_time = time.perf_counter()
        print(f"[Request Received] /api/python/edit/strategy | Prompt: {req.prompt}")
        
        print("Model Manager: Loading TinyLlama...")
        pipe = model_manager.get_llm_pipe()
        
        prompt = f"<|system|>\nYou are a professional image editor. Suggest 3 specific editing improvements (color, lighting, composition) for this scene.<|end|>\n<|user|>\n{req.prompt}<|end|>\n<|assistant|>\n"
        print("Generating editing strategy...")
        result = pipe(prompt, max_new_tokens=100, return_full_text=False)
        strategy = result[0]['generated_text'].strip()
        
        elapsed = time.perf_counter() - start_time
        print(f"[Success] edit_strategy took {elapsed:.2f}s")
        return {"success": True, "text": strategy, "execution_time_sec": elapsed}

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/edit/strategy:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/remove-bg")
async def remove_bg(req: AiRequest):
    def task():
        start_time = time.perf_counter()
        print("[Request Received] /api/python/edit/remove-bg")
        
        input_image = ImageProcessor.base64_to_pil(req.image)
        w, h = input_image.size
        print(f"Preprocessing input image (size: {w}x{h}, mode: {input_image.mode})...")
        
        print("Model Manager: Running MODNet...")
        output_image = model_manager.remove_bg_modnet(input_image)
        
        elapsed = time.perf_counter() - start_time
        print(f"[Success] remove_bg took {elapsed:.2f}s")
        return {"success": True, "image": ImageProcessor.pil_to_base64(output_image), "execution_time_sec": elapsed}

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/edit/remove-bg:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/bg-replace")
async def replace_bg(req: AiRequest):
    def task():
        start_time = time.perf_counter()
        print(f"[Request Received] /api/python/edit/bg-replace | Prompt: {req.prompt}")
        
        pil_img = ImageProcessor.base64_to_pil(req.image)
        w, h = pil_img.size
        print(f"Preprocessing input image (size: {w}x{h}, mode: {pil_img.mode})...")
        
        print("Model Manager: Segmenting foreground via MODNet...")
        fg = model_manager.remove_bg_modnet(pil_img)
        
        prompt = req.prompt + ", beautiful landscape, masterpiece, high resolution"
        max_dim = gpu_manager.max_dimension
        bg_w, bg_h = model_manager.get_proportional_size((w, h), max_dim)
        print(f"[BG Replace] Using VRAM target resolution for background: {bg_w}x{bg_h} (matching input aspect ratio)")
        blank = Image.new("RGB", (bg_w, bg_h), (255, 255, 255))
        
        print("Generating new background...")
        bg, meta = model_manager.run_ghibli_inference(
            prompt=prompt, 
            image=blank, 
            strength=1.0, 
            num_inference_steps=4,
            guidance_scale=0.0
        )
        
        print("Compositing foreground cutout onto generated background...")
        bg = bg.resize((w, h))
        bg.paste(fg, (0, 0), fg)
        
        gpu_manager.flush_memory()
        elapsed = time.perf_counter() - start_time
        print(f"[Success] replace_bg took {elapsed:.2f}s")
        return {
            "success": True, 
            "image": ImageProcessor.pil_to_base64(bg), 
            "execution_time_sec": elapsed,
            "peak_vram_mb": meta.get("peak_vram_mb"),
            "cpu_offload_used": meta.get("cpu_offload_used")
        }

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/edit/bg-replace:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/upscale")
async def upscale_image(req: AiRequest):
    def task():
        start_time = time.perf_counter()
        scale = req.upscale_factor if req.upscale_factor is not None else 2
        if scale not in [2, 4]:
            scale = 2
            
        print(f"[Request Received] /api/python/edit/upscale | Scale: {scale}X")
        
        print("Model Manager: Loading Real-ESRGAN...")
        upscaler = model_manager.get_upscaler()
        
        print("Running Real-ESRGAN upscaling...")
        img = ImageProcessor.base64_to_cv2(req.image)
        output, _ = upscaler.enhance(img, outscale=scale)
        
        gpu_manager.flush_memory()
        elapsed = time.perf_counter() - start_time
        print(f"[Success] upscale ({scale}X) took {elapsed:.2f}s")
        return {"success": True, "image": ImageProcessor.cv2_to_base64(output), "execution_time_sec": elapsed}

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/edit/upscale:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/inpaint")
async def inpaint_object(req: AiRequest):
    init_image = ImageProcessor.base64_to_pil(req.image)
    mask_image = ImageProcessor.base64_to_pil(req.mask)
    
    # Preprocess mask to binary format
    print(f"Processing mask (mode: {mask_image.mode})...")
    if mask_image.mode in ('RGBA', 'LA') or (mask_image.mode == 'P' and 'transparency' in mask_image.info):
        alpha = mask_image.convert('RGBA').split()[-1]
        mask_l = alpha.point(lambda x: 255 if x > 10 else 0)
    else:
        mask_l = mask_image.convert("L").point(lambda x: 255 if x > 10 else 0)

    # 1. Automatic Mask Expansion (Dilation by 5px, kernel 11)
    mask_np = np.array(mask_l)
    kernel_size = 11
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
    
    # 2. Feathering: Smooth mask borders (15px Gaussian blur)
    mask_feathered = cv2.GaussianBlur(mask_dilated, (15, 15), 0)
    mask_feathered_float = mask_feathered.astype(np.float32) / 255.0
    
    mask_dilated_pil = Image.fromarray(mask_dilated)
    
    # Selected quality mode
    mode = req.mode or "balanced"
    print(f"Inpaint: Mode selected: {mode}, Original Size: {init_image.size}")

    def task():
        nonlocal mode
        start_time = time.perf_counter()
        orig_size = init_image.size
        init_image_rgb = init_image.convert("RGB")
        model_used = "OpenCV Inpaint"
        output = None
        
        # Mode implementation & Fallback hierarchy
        if mode == "high":
            try:
                print("Inpaint Task: Attempting Stable Diffusion descriptive inpainting...")
                init_image_resized = init_image.convert("RGB").resize((512, 512))
                mask_dilated_resized = mask_dilated_pil.resize((512, 512)).point(lambda x: 255 if x > 10 else 0)
                pipe = model_manager.get_inpaint_pipe()
                prompt_str = req.prompt or "remove object"
                output = pipe(
                    prompt=prompt_str, 
                    image=init_image_resized, 
                    mask_image=mask_dilated_resized,
                    num_inference_steps=20
                ).images[0]
                output = output.resize(orig_size)
                model_used = "Stable Diffusion"
            except Exception as e:
                import traceback
                print(f"Stable Diffusion inpainting failed: {e}. Falling back to SimpleLama...")
                traceback.print_exc()
                mode = "balanced"
                
        if mode == "balanced" and output is None:
            try:
                print("Inpaint Task: Attempting SimpleLama...")
                lama = model_manager.get_simple_lama()
                output = lama(init_image_rgb, mask_dilated_pil)
                model_used = "SimpleLama"
            except Exception as e:
                import traceback
                print(f"SimpleLama failed: {e}. Falling back to OpenCV...")
                traceback.print_exc()
                mode = "fast"
                
        if (mode == "fast" or output is None) and output is None:
            print("Inpaint Task: Running OpenCV Inpainting...")
            output = ImageProcessor.opencv_inpaint(init_image_rgb, mask_dilated_pil)
            model_used = "OpenCV Inpaint"

        # Crop to original size to remove padding if size changed by models (like SimpleLama)
        if output.size != orig_size:
            output = output.crop((0, 0, orig_size[0], orig_size[1]))

        # 3. Feathered Blending to prevent hard transitions
        print("Inpaint Task: Blending outputs using feathered mask...")
        img_np = np.array(init_image_rgb).astype(np.float32)
        out_np = np.array(output).astype(np.float32)
        mask_blend = np.expand_dims(mask_feathered_float, axis=-1)
        
        blended_np = img_np * (1.0 - mask_blend) + out_np * mask_blend
        blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
        output = Image.fromarray(blended_np)
            
        # Restore transparency if input had an alpha channel
        if init_image.mode == "RGBA":
            print("Restoring transparency channel in inpainted output...")
            r, g, b, _ = output.convert("RGBA").split()
            _, _, _, a = init_image.split()
            output = Image.merge("RGBA", (r, g, b, a))
            
        gpu_manager.flush_memory()
        elapsed = time.perf_counter() - start_time
        print(f"[Success] {model_used} inpaint took {elapsed:.2f}s")
        
        return {
            "success": True, 
            "image": ImageProcessor.pil_to_base64(output), 
            "execution_time_sec": elapsed, 
            "model_used": model_used,
            "processing_time": f"{elapsed:.2f}s",
            "image_size": f"{orig_size[0]}x{orig_size[1]}"
        }

    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in Inpaint task execution:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/python/edit/sketch")
async def enhance_sketch(req: AiRequest):
    def task():
        start_time = time.perf_counter()
        print("[Request Received] /api/python/edit/sketch")
        
        img = ImageProcessor.base64_to_cv2(req.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
        
        elapsed = time.perf_counter() - start_time
        print(f"[Success] sketch took {elapsed:.2f}s")
        return {"success": True, "image": ImageProcessor.cv2_to_base64(sketch), "execution_time_sec": elapsed}
        
    try:
        return await task_queue.submit(task)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error in /api/python/edit/sketch:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    model_manager.verify_local_paths()
    model_manager.preload_models()
    print(f"Starting AI Engine on device: {gpu_manager.device_name}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
