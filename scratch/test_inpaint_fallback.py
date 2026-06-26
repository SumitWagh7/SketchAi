import os
import requests
import base64
from PIL import Image
from io import BytesIO

# Port 8000 is FastAPI backend
API_URL = "http://localhost:8000/api/python/edit/inpaint"

# 1. Create a mock image
w, h = 100, 100
mock_img = Image.new("RGB", (w, h), (128, 128, 128))
for y in range(30, 70):
    for x in range(30, 70):
        mock_img.putpixel((x, y), (255, 0, 0)) # Red square in middle

buffered_img = BytesIO()
mock_img.save(buffered_img, format="PNG")
img_b64 = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

# 2. Create a mask
mask_img = Image.new("L", (w, h), 0)
for y in range(40, 60):
    for x in range(40, 60):
        mask_img.putpixel((x, y), 255)

buffered_mask = BytesIO()
mask_img.save(buffered_mask, format="PNG")
mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

# Test 1: With SimpleLama model weights present
print("--- Test 1: Inpainting with SimpleLama (Weights Present) ---")
payload = {
    "image": img_b64,
    "mask": mask_b64,
    "prompt": "remove object"
}
try:
    resp = requests.post(API_URL, json=payload, timeout=30)
    print("Status:", resp.status_code)
    data = resp.json()
    print("Success:", data.get("success"))
    print("Model Used:", data.get("model_used"))
except Exception as e:
    print("Test 1 Failed:", e)

# Test 2: Simulating SimpleLama Failure by moving weights
print("\n--- Test 2: Inpainting Fallback to OpenCV (Weights Missing) ---")
model_path = "models/object_removal/big-lama.pt"
backup_path = "models/object_removal/big-lama.pt.bak"

weights_exist = os.path.exists(model_path)
if weights_exist:
    print("Renaming weights to simulate missing model...")
    os.rename(model_path, backup_path)

try:
    resp = requests.post(API_URL, json=payload, timeout=30)
    print("Status:", resp.status_code)
    data = resp.json()
    print("Success:", data.get("success"))
    print("Model Used:", data.get("model_used"))
    if data.get("model_used") == "OpenCV Inpaint":
        print("PASS: Fallback to OpenCV Inpaint was successful!")
    else:
        print("FAIL: Fallback did not use OpenCV!")
except Exception as e:
    print("Test 2 Failed:", e)
finally:
    if weights_exist and os.path.exists(backup_path):
        print("Restoring weights file...")
        os.rename(backup_path, model_path)
