import os
import requests
import base64
import time
from PIL import Image
from io import BytesIO

SB_URL = "http://localhost:8080"
results_dir = "scratch/benchmark_results"
os.makedirs(results_dir, exist_ok=True)

# 1. Authenticate with Spring Boot
print("Authenticating with Spring Boot backend...")
login_url = SB_URL + "/api/auth/login"
payload = {
    "email": "testuser_fixes@example.com",
    "password": "Password123!"
}
resp = requests.post(login_url, json=payload)
if resp.status_code != 200:
    print("Authentication failed. Registering test user...")
    register_url = SB_URL + "/api/auth/register"
    reg_payload = {
        "username": "benchmark_user",
        "email": "testuser_fixes@example.com",
        "password": "Password123!",
        "phone": "1234567890"
    }
    requests.post(register_url, json=reg_payload)
    resp = requests.post(login_url, json=payload)

token = resp.json().get("data", {}).get("token")
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
print("Authenticated successfully. Token acquired.")

# Helper to run inpainting
def run_inpaint_benchmark(img_name, img_pil, mask_pil, mode, prompt="remove object"):
    # Convert image and mask to base64
    buffered_img = BytesIO()
    img_pil.save(buffered_img, format="PNG")
    img_b64 = base64.b64encode(buffered_img.getvalue()).decode("utf-8")
    
    buffered_mask = BytesIO()
    mask_pil.save(buffered_mask, format="PNG")
    mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")
    
    payload = {
        "image": img_b64,
        "mask": mask_b64,
        "prompt": prompt,
        "mode": mode
    }
    
    print(f"  Running [{mode}] mode on {img_name}...")
    start_time = time.perf_counter()
    resp = requests.post(SB_URL + "/api/inpaint", json=payload, headers=headers)
    elapsed = time.perf_counter() - start_time
    
    if resp.status_code == 200:
        data = resp.json()
        out_b64 = data.get("image")
        out_img = Image.open(BytesIO(base64.b64decode(out_b64)))
        
        # Save output
        out_path = f"{results_dir}/{img_name.split('.')[0]}_{mode}_output.png"
        out_img.save(out_path)
        
        print(f"    Success! Saved: {out_path}")
        print(f"    Metadata: model_used={data.get('model_used')}, processing_time={data.get('processing_time')}, image_size={data.get('image_size')}")
        print(f"    Client side elapsed: {elapsed:.2f}s")
        return {
            "status": "Success",
            "model_used": data.get("model_used"),
            "processing_time": data.get("processing_time"),
            "image_size": data.get("image_size"),
            "aspect_ratio_ok": out_img.size == img_pil.size,
            "transparency_ok": out_img.mode == img_pil.mode
        }
    else:
        print(f"    Failed! Status: {resp.status_code}, Error: {resp.text}")
        return {"status": "Failed", "error": resp.text}

# Prepare Test Cases
print("\nPreparing test cases...")
# Case 1: Transparent PNG (representing Small object removal with transparency)
print("Creating Transparent PNG Image...")
w, h = 128, 128
png_img = Image.new("RGBA", (w, h), (0, 0, 0, 0)) # fully transparent
for y in range(40, 88):
    for x in range(40, 88):
        png_img.putpixel((x, y), (0, 0, 255, 255)) # blue square in center

png_mask = Image.new("L", (w, h), 0)
for y in range(50, 78):
    for x in range(50, 78):
        png_mask.putpixel((x, y), 255) # erase middle of blue square

# Save originals
png_img.save(f"{results_dir}/original_transparent.png")
png_mask.save(f"{results_dir}/mask_transparent.png")

# Case 2: Complex Background JPG (representing Large object removal / Complex bg)
print("Creating Complex background JPG Image...")
jpg_img = Image.new("RGB", (300, 300))
# Fill with gradient background
for y in range(300):
    for x in range(300):
        jpg_img.putpixel((x, y), (x % 256, y % 256, (x + y) // 2))
# Add large target object to erase
for y in range(100, 200):
    for x in range(100, 200):
        jpg_img.putpixel((x, y), (255, 0, 0)) # large red box

jpg_mask = Image.new("L", (300, 300), 0)
for y in range(95, 205):
    for x in range(95, 205):
        jpg_mask.putpixel((x, y), 255)

# Save originals
jpg_img.save(f"{results_dir}/original_gradient.png")
jpg_mask.save(f"{results_dir}/mask_gradient.png")


# Execute benchmarks for both test cases in all 3 quality modes
benchmarks = {}

print("\n--- Running Transparent PNG benchmarks ---")
benchmarks["transparent_fast"] = run_inpaint_benchmark("transparent.png", png_img, png_mask, "fast")
benchmarks["transparent_balanced"] = run_inpaint_benchmark("transparent.png", png_img, png_mask, "balanced")
benchmarks["transparent_high"] = run_inpaint_benchmark("transparent.png", png_img, png_mask, "high")

print("\n--- Running Complex Gradient JPG benchmarks ---")
benchmarks["gradient_fast"] = run_inpaint_benchmark("gradient.jpg", jpg_img, jpg_mask, "fast")
benchmarks["gradient_balanced"] = run_inpaint_benchmark("gradient.jpg", jpg_img, jpg_mask, "balanced")
benchmarks["gradient_high"] = run_inpaint_benchmark("gradient.jpg", jpg_img, jpg_mask, "high")

print("\n=== Benchmark Summary ===")
for k, v in benchmarks.items():
    print(f"{k}: {v}")
