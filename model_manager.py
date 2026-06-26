import base64
import os
import gc
import cv2
import torch
import numpy as np
import threading
import queue
from io import BytesIO
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Advanced Deep Learning Libraries
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from transformers import pipeline
from rembg import remove

# Root directory setups
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["U2NET_HOME"] = os.path.join(MODEL_DIR, "background")

# 1. GPU Manager
class GPUManager:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        self.dtype = torch.float16 if self.cuda_available else torch.float32
        self.is_4gb_gpu = False
        self.max_dimension = 512
        self.target_resolution = (512, 512)
        
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Manager: NVIDIA GPU detected -> {self.device_name} ({vram_gb:.1f} GB VRAM). CUDA and FP16 enabled.")
            if vram_gb <= 4.5:
                self.is_4gb_gpu = True
                self.max_dimension = 512
                self.target_resolution = (512, 512)
                print("GPU Manager: GPU has 4.5GB or less VRAM. Activating memory optimizations. max_dimension=512")
            elif vram_gb <= 8.5:
                self.is_4gb_gpu = False
                self.max_dimension = 768
                self.target_resolution = (768, 768)
                print("GPU Manager: GPU has VRAM > 4.5GB and <= 8.5GB. max_dimension=768")
            else:
                self.is_4gb_gpu = False
                self.max_dimension = 1024
                self.target_resolution = (1024, 1024)
                print("GPU Manager: GPU has VRAM > 8.5GB. max_dimension=1024")
        else:
            self.device_name = "CPU"
            print("GPU Manager: No NVIDIA GPU detected. Falling back to CPU mode. max_dimension=512")

        if os.environ.get("FORCE_4GB_OPTIMIZATION") == "1":
            self.is_4gb_gpu = True
            self.max_dimension = 512
            self.target_resolution = (512, 512)
            print("GPU Manager: FORCE_4GB_OPTIMIZATION=1 override active. max_dimension=512")
        elif os.environ.get("FORCE_8GB_OPTIMIZATION") == "1":
            self.is_4gb_gpu = False
            self.max_dimension = 768
            self.target_resolution = (768, 768)
            print("GPU Manager: FORCE_8GB_OPTIMIZATION=1 override active. max_dimension=768")
        elif os.environ.get("FORCE_12GB_OPTIMIZATION") == "1":
            self.is_4gb_gpu = False
            self.max_dimension = 1024
            self.target_resolution = (1024, 1024)
            print("GPU Manager: FORCE_12GB_OPTIMIZATION=1 override active. max_dimension=1024")

    def get_onnx_providers(self):
        if self.cuda_available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def flush_memory(self):
        if self.cuda_available:
            torch.cuda.empty_cache()
        gc.collect()

# 2. Image Processor
class ImageProcessor:
    @staticmethod
    def base64_to_cv2(b64_str):
        img_data = base64.b64decode(b64_str.split(',')[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def cv2_to_base64(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def base64_to_pil(b64_str):
        return Image.open(BytesIO(base64.b64decode(b64_str.split(',')[-1])))

    @staticmethod
    def pil_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def opencv_inpaint(img_pil, mask_pil):
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        mask = np.array(mask_pil)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        return Image.fromarray(dst)

# 3. Cache Manager
class CacheManager:
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.cache.get(key)

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest = self.keys.pop(0)
                self.cache.pop(oldest, None)
            self.cache[key] = value
            self.keys.append(key)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.keys.clear()
        gpu_manager.flush_memory()

# 4. Task Queue
class TaskQueue:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def submit(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, fn)

gpu_manager = GPUManager()
cache_manager = CacheManager()
task_queue = TaskQueue()

# 5. Model Manager
class ModelManager:
    def __init__(self):
        self.models = {
            "ghibli": None,
            "llm": None,
            "inpaint": None,
            "upscaler": None,
            "face_enhancer_2x": None,
            "face_enhancer_4x": None,
            "rembg_session": None,
            "modnet": None,
            "simple_lama": None
        }
        self.paths = {
            "background": os.path.join(MODEL_DIR, "background"),
            "object_removal": os.path.join(MODEL_DIR, "object_removal"),
            "upscale": os.path.join(MODEL_DIR, "upscale"),
            "stable_diffusion": os.path.join(MODEL_DIR, "stable_diffusion")
        }

    def verify_local_paths(self):
        u2net_path = os.path.join(self.paths["background"], "u2net.onnx")
        realesrgan_path = os.path.join(self.paths["upscale"], "RealESRGAN_x4plus.pth")
        gfpgan_path = os.path.join(self.paths["upscale"], "GFPGANv1.4.pth")
        
        missing = []
        if not os.path.exists(u2net_path):
            missing.append("background/u2net.onnx")
        if not os.path.exists(realesrgan_path):
            missing.append("upscale/RealESRGAN_x4plus.pth")
        if not os.path.exists(gfpgan_path):
            missing.append("upscale/GFPGANv1.4.pth")
            
        if missing:
            print(f"Warning: Missing local weights in subdirectories: {missing}")
        else:
            print("Model Manager: All local weights verified successfully.")

    def get_rembg_session(self):
        if self.models["rembg_session"] is None:
            import rembg
            providers = gpu_manager.get_onnx_providers()
            print(f"Model Manager: Initializing rembg session on {providers}")
            self.models["rembg_session"] = rembg.new_session("u2net", providers=providers)
        return self.models["rembg_session"]

    def get_modnet(self):
        if self.models["modnet"] is None:
            print("Model Manager: Loading MODNet for Background Removal...")
            weight_path = os.path.join(self.paths["background"], "matting_modnet_portrait.pth")
            
            if not os.path.exists(weight_path):
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                url = "https://github.com/xinntao/facexlib/releases/download/v0.2.0/matting_modnet_portrait.pth"
                print(f"Model Manager: Downloading MODNet weights from {url}...")
                import requests
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(weight_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model Manager: MODNet weights downloaded successfully.")
                
            from facexlib.matting.modnet import MODNet
            model = MODNet(backbone_pretrained=False)
            state_dict = torch.load(weight_path, map_location="cpu")
            
            # Strip module. prefix if weights were saved with DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict)
            model = model.to(gpu_manager.device, dtype=torch.float32)
            model.eval()
            self.models["modnet"] = model
        return self.models["modnet"]

    def remove_bg_modnet(self, img_pil):
        model = self.get_modnet()
        w, h = img_pil.size
        target_w, target_h = 512, 512
        
        # Fast preprocessing using cv2/numpy
        img_np = np.array(img_pil.convert("RGB"))
        img_resized = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = (img_resized - 0.5) / 0.5
        img_resized = np.transpose(img_resized, (2, 0, 1))
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).to(gpu_manager.device, dtype=torch.float32)
        
        with torch.no_grad():
            _, _, matte = model(img_tensor, True)
            
        matte = matte.squeeze(0).cpu().numpy()
        matte = np.transpose(matte, (1, 2, 0))
        matte = (matte * 255.0).astype(np.uint8)
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if len(matte.shape) == 2:
            alpha_channel = matte
        else:
            alpha_channel = matte[:, :, 0]
            
        rgba_img = img_pil.convert("RGBA")
        rgba_img.putalpha(Image.fromarray(alpha_channel, mode='L'))
        return rgba_img

    def get_ghibli_pipe(self):
        if self.models["ghibli"] is None:
            import sys
            import traceback
            print("[Model Loader] =====================================================")
            print("[Model Loader] Beginning Studio Ghibli SDXL Turbo Pipeline load...")
            print("[Model Loader] =====================================================")
            
            # 1. GPU / CUDA and VRAM Diagnostics
            is_low_vram = False
            vram_gb = 0.0
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2) # MB
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
                allocated_vram = torch.cuda.memory_allocated(0) / (1024**2) # MB
                cached_vram = torch.cuda.memory_reserved(0) / (1024**2) # MB
                print(f"[Model Loader] CUDA Device detected: {torch.cuda.get_device_name(0)}")
                print(f"[Model Loader] VRAM Status: Total={total_vram:.1f}MB ({vram_gb:.2f} GB), Allocated={allocated_vram:.1f}MB, Cached={cached_vram:.1f}MB")
                if vram_gb < 5.0:
                    is_low_vram = True
                    print(f"[Model Loader] WARNING: GPU has low VRAM ({vram_gb:.2f} GB). Memory-saving features will be enforced.")
            else:
                print("[Model Loader] Running on CPU (CUDA unavailable)")

            # 2. Path Verification
            cache_dir = self.paths["stable_diffusion"]
            print(f"[Model Loader] Target model path: {cache_dir}")
            print(f"[Model Loader] Configured local_files_only = True (offline mode)")
            
            # Check if cache dir exists
            if not os.path.exists(cache_dir):
                print(f"[Model Loader] WARNING: Model directory {cache_dir} does not exist!")
            else:
                # Log files in the cache to help with diagnostics
                files = []
                for r, d, fs in os.walk(cache_dir):
                    for f in fs:
                        if f.endswith(('.safetensors', '.bin', '.onnx', 'json')):
                            files.append(os.path.relpath(os.path.join(r, f), cache_dir))
                print(f"[Model Loader] Available cached files: {files}")

            from diffusers import StableDiffusionXLImg2ImgPipeline
            
            # 3. GPU Loading Attempt
            try:
                print("[Model Loader] Stage: Loading SDXL Turbo pipeline on GPU...")
                vram_before = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
                
                if gpu_manager.is_4gb_gpu:
                    print("[Model Loader] Enforcing 4GB VRAM GPU optimizations for SDXL Turbo.")
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
                    # Enable sequential CPU offload instead of moving full model to GPU
                    pipe.enable_sequential_cpu_offload()
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                    pipe.enable_attention_slicing()
                    pipe.cpu_offload_used = True
                else:
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        "stabilityai/sdxl-turbo", 
                        torch_dtype=gpu_manager.dtype, 
                        cache_dir=cache_dir,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True,
                        use_safetensors=True,
                        variant="fp16",
                        low_cpu_mem_usage=True
                    ).to(gpu_manager.device)
                    pipe.cpu_offload_used = False
                    
                    if is_low_vram:
                        print("[Model Loader] Low VRAM safety fallback: Enforcing attention slicing.")
                        pipe.enable_attention_slicing()
                    else:
                        try:
                            pipe.enable_xformers_memory_efficient_attention()
                            print("[Model Loader] Enabled xformers memory efficient attention.")
                        except Exception as ex_attn:
                            print(f"[Model Loader] xformers memory efficient attention not available: {ex_attn}. Enabling attention slicing instead as fallback.")
                            pipe.enable_attention_slicing()
                
                vram_after = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
                print(f"[Model Loader] SDXL Turbo pipeline successfully loaded. VRAM consumption delta: +{vram_after - vram_before:.1f} MB (Current: {vram_after:.1f} MB)")
                
                # Load Ghibli LoRA if present
                lora_path = os.path.join(self.paths["stable_diffusion"], "StudioGhibli.safetensors")
                if os.path.exists(lora_path):
                    print(f"[Model Loader] Stage: Loading Studio Ghibli LoRA from {lora_path}...")
                    pipe.load_lora_weights(lora_path)
                    print("[Model Loader] Studio Ghibli LoRA loaded successfully.")
                else:
                    print(f"[Model Loader] Warning: Studio Ghibli LoRA not found at {lora_path}.")
                    
                self.models["ghibli"] = pipe
                
            except Exception as e_gpu:
                print(f"[Model Loader ERROR] Failed to load SDXL Turbo on GPU: {e_gpu}. Falling back to CPU...", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                
                # 4. CPU Fallback Attempt
                try:
                    print("[Model Loader] Stage: Loading SDXL Turbo pipeline on CPU...")
                    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        "stabilityai/sdxl-turbo", 
                        torch_dtype=torch.float32, 
                        cache_dir=cache_dir,
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=True,
                        use_safetensors=True,
                        variant="fp16",
                        low_cpu_mem_usage=True
                    ).to("cpu")
                    pipe.cpu_offload_used = False
                    print("[Model Loader] SDXL Turbo pipeline successfully loaded on CPU.")
                    
                    print("[Model Loader] CPU load safety fallback: Enforcing attention slicing.")
                    pipe.enable_attention_slicing()
                    
                    lora_path = os.path.join(self.paths["stable_diffusion"], "StudioGhibli.safetensors")
                    if os.path.exists(lora_path):
                        print(f"[Model Loader] Stage: Loading Studio Ghibli LoRA from {lora_path} (CPU)...")
                        pipe.load_lora_weights(lora_path)
                        print("[Model Loader] Studio Ghibli LoRA loaded successfully on CPU.")
                        
                    self.models["ghibli"] = pipe
                except Exception as e_cpu:
                    print(f"[Model Loader CRITICAL] Failed to load SDXL Turbo on CPU: {e_cpu}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    raise e_cpu
                    
        return self.models["ghibli"]

    def get_llm_pipe(self):
        if self.models["llm"] is None:
            print("Model Manager: Loading Local LLM (TinyLlama)...")
            cache_dir = self.paths["stable_diffusion"]
            self.models["llm"] = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                cache_dir=cache_dir,
                device=0 if gpu_manager.cuda_available else -1,
                local_files_only=True
            )
        return self.models["llm"]

    def get_inpaint_pipe(self):
        if self.models["inpaint"] is None:
            print("Model Manager: Loading SD Inpainting Pipeline...")
            cache_dir = self.paths["object_removal"]
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting", 
                    torch_dtype=gpu_manager.dtype,
                    cache_dir=cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                ).to(gpu_manager.device)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("Model Manager: Enabled xformers memory efficient attention for Inpainting.")
                except Exception:
                    pipe.enable_attention_slicing()
                self.models["inpaint"] = pipe
            except Exception as e:
                print(f"Model Manager: Failed to load SD Inpainting on GPU: {e}. Falling back to CPU.")
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting", 
                    torch_dtype=torch.float32,
                    cache_dir=cache_dir,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                ).to("cpu")
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_attention_slicing()
                self.models["inpaint"] = pipe
        return self.models["inpaint"]

    def get_upscaler(self):
        if self.models["upscaler"] is None:
            print("Model Manager: Loading Real-ESRGAN...")
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model_path = os.path.join(self.paths["upscale"], "RealESRGAN_x4plus.pth")
            if not os.path.exists(model_path):
                raise Exception(f"Model not found: {model_path}")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.models["upscaler"] = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if gpu_manager.cuda_available else False,
                device=gpu_manager.device
            )
        return self.models["upscaler"]

    def get_face_enhancer(self, scale=2):
        key = f"face_enhancer_{scale}x"
        if self.models[key] is None:
            print(f"Model Manager: Loading GFPGAN for {scale}x upscale...")
            from gfpgan import GFPGANer
            model_path = os.path.join(self.paths["upscale"], "GFPGANv1.4.pth")
            if not os.path.exists(model_path):
                raise Exception(f"Model not found: {model_path}")
            upscaler = self.get_upscaler()
            self.models[key] = GFPGANer(
                model_path=model_path,
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upscaler
            )
        return self.models[key]

    def get_simple_lama(self):
        if self.models["simple_lama"] is None:
            print("Model Manager: Loading SimpleLama for Object Removal...")
            from simple_lama_inpainting import SimpleLama
            os.environ["LAMA_MODEL"] = os.path.join(self.paths["object_removal"], "big-lama.pt")
            if not os.path.exists(os.environ["LAMA_MODEL"]):
                raise Exception(f"Lama model not found at: {os.environ['LAMA_MODEL']}")
            self.models["simple_lama"] = SimpleLama(device=gpu_manager.device)
        return self.models["simple_lama"]

    def get_proportional_size(self, size, max_dim):
        """
        Calculates width and height proportionally such that the longest side <= max_dim.
        Ensures both dimensions are rounded to the nearest multiple of 8 for Stable Diffusion VAE compatibility.
        """
        w, h = size
        longest = max(w, h)
        scale = min(1.0, max_dim / longest)
        
        new_w = max(8, int(round(w * scale / 8) * 8))
        new_h = max(8, int(round(h * scale / 8) * 8))
        return new_w, new_h

    def run_ghibli_inference(self, prompt, image, strength=0.5, guidance_scale=0.0, num_inference_steps=4):
        """
        Runs inference using the Ghibli pipeline, enforcing memory-saving parameters
        for 4GB VRAM GPUs (max 4 steps, max 512x512 image size during testing/execution),
        logging performance metrics (peak VRAM, inference time, CPU offload status),
        and preventing CUDA Out Of Memory errors.
        """
        import time
        import hashlib
        from io import BytesIO
        
        # Log input image details
        print(f"[Model Manager] Input PIL image dimensions: {image.size[0]}x{image.size[1]}")
        try:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            sha256_hash = hashlib.sha256(img_bytes).hexdigest()
            print(f"[Model Manager] Input image size passed to inference: {len(img_bytes)} bytes")
            print(f"[Model Manager] Input image SHA-256 hash passed to inference: {sha256_hash}")
        except Exception as e:
            print(f"[Model Manager ERROR] Error calculating hash: {e}")
            
        print(f"[Model Manager] Confirming active image object (ID: {id(image)}) is passed directly to pipeline. No placeholder image used.")

        pipe = self.get_ghibli_pipe()
        is_4gb = gpu_manager.is_4gb_gpu
        cpu_offload_used = getattr(pipe, "cpu_offload_used", False)
        
        # 1. Limit image size proportionally based on available VRAM classification (longest side constraint, multiples of 8)
        max_dim = gpu_manager.max_dimension
        w, h = image.size
        longest = max(w, h)
        if longest > max_dim or (w % 8 != 0 or h % 8 != 0):
            new_w, new_h = self.get_proportional_size((w, h), max_dim)
            print(f"[Memory Opt] Resizing input image from {w}x{h} to {new_w}x{new_h} based on VRAM limit {max_dim} (aspect ratio preserved).")
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Recalculate hash for resized image
            try:
                img_byte_arr_resized = BytesIO()
                image.save(img_byte_arr_resized, format='PNG')
                img_bytes_resized = img_byte_arr_resized.getvalue()
                sha256_hash_resized = hashlib.sha256(img_bytes_resized).hexdigest()
                print(f"[Model Manager] Resized image size passed to pipeline: {len(img_bytes_resized)} bytes")
                print(f"[Model Manager] Resized image SHA-256 hash passed to pipeline: {sha256_hash_resized}")
                print(f"[Model Manager] Resized image dimensions: {image.size[0]}x{image.size[1]}")
            except Exception as e:
                print(f"[Model Manager ERROR] Error calculating resized hash: {e}")
                
        # 2. Limit inference steps to 4
        if is_4gb:
            if num_inference_steps > 4:
                print(f"[Memory Opt] Limiting inference steps from {num_inference_steps} to 4 for 4GB VRAM GPU.")
            num_inference_steps = min(num_inference_steps, 4)
            
        # 3. Reset Peak VRAM stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.perf_counter()
        
        # --- INVESTIGATION DIAGNOSTICS ---
        print(f"[Investigation] Pipeline type: {type(pipe)}")
        print(f"[Investigation] Inference parameters:")
        print(f"  - strength: {strength}")
        print(f"  - guidance_scale: {guidance_scale}")
        print(f"  - num_inference_steps: {num_inference_steps}")
        print(f"  - prompt: '{prompt}'")
        print(f"  - negative_prompt: None")
        
        # Log LoRA scale (defaults to 1.0 in diffusers unless cross_attention_kwargs contains scale)
        print(f"[Investigation] LoRA scale: 1.0")
        
        # Save the exact image passed to pipeline
        try:
            os.makedirs("scratch", exist_ok=True)
            image.save("scratch/debug_input_to_pipeline.png")
            print("[Investigation] Saved exact input image to: scratch/debug_input_to_pipeline.png")
        except Exception as e:
            print(f"[Investigation ERROR] Failed to save input image: {e}")
            
        print("[Investigation] Confirming input_image is passed to pipe(image=input_image)")
        # ---------------------------------
        
        # 4. Run pipeline, catch OOM or other errors
        try:
            print(f"[Model Manager] Invoking StableDiffusionXLImg2ImgPipeline with strength={strength}, guidance_scale={guidance_scale}, steps={num_inference_steps}...")
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
                generated_image = output.images[0]
                
                # Save exact generated output
                try:
                    generated_image.save("scratch/debug_output_from_pipeline.png")
                    print("[Investigation] Saved exact generated output image to: scratch/debug_output_from_pipeline.png")
                except Exception as e:
                    print(f"[Investigation ERROR] Failed to save output image: {e}")
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"[Memory Opt ERROR] CUDA Out of Memory during inference! Clearing cache and throwing error: {oom_err}")
            gpu_manager.flush_memory()
            raise RuntimeError("CUDA Out of Memory during image generation. Try reducing image size or complexity.") from oom_err
        except Exception as e:
            print(f"[Inference ERROR] Failed during generation: {e}")
            raise e
            
        inference_time = time.perf_counter() - start_time
        
        # Log output details
        try:
            out_byte_arr = BytesIO()
            generated_image.save(out_byte_arr, format='PNG')
            out_bytes = out_byte_arr.getvalue()
            out_sha256 = hashlib.sha256(out_bytes).hexdigest()
            print(f"[Model Manager] Generated output image dimensions: {generated_image.size[0]}x{generated_image.size[1]}")
            print(f"[Model Manager] Generated image size: {len(out_bytes)} bytes")
            print(f"[Model Manager] Generated image SHA-256 hash: {out_sha256}")
        except Exception as e:
            print(f"[Model Manager ERROR] Error calculating output hash: {e}")
        
        # 5. Log metrics
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"[Performance Log] Peak VRAM usage: {peak_vram_mb:.2f} MB")
        else:
            print("[Performance Log] Peak VRAM usage: N/A (Running on CPU)")
            
        print(f"[Performance Log] Inference time: {inference_time:.2f}s")
        print(f"[Performance Log] Sequential CPU offload used: {cpu_offload_used}")
        
        metadata = {
            "peak_vram_mb": peak_vram_mb,
            "inference_time_sec": inference_time,
            "cpu_offload_used": cpu_offload_used,
            "steps_used": num_inference_steps,
            "image_size_used": image.size
        }
        
        return generated_image, metadata

    def preload_models(self):
        print("Model Manager: Lazy preloading enabled. Models will load on-demand.")

model_manager = ModelManager()
