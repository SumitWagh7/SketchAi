import os
import time
import json
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

SPRING_BOOT_URL = "http://localhost:8080"

def test_large_noise():
    print("=== TESTING LARGE REQUEST LIMIT WITH RANDOM NOISE ===")
    
    # 1. Login to get token
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{SPRING_BOOT_URL}/api/auth/login", json={
        "email": "audit@example.com",
        "password": "Password123!"
    }, headers=headers, timeout=10)
    
    token = resp.json()["data"]["token"]
    
    # 2. Create a random noise image (1500x1500px)
    print("Generating 1500x1500px random noise image (high entropy)...")
    arr = np.random.randint(0, 256, (1500, 1500, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    raw_size = len(img_bytes)
    
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    payload_size = len(img_b64)
    print(f"Generated PNG raw size: {raw_size} bytes ({raw_size / (1024*1024):.2f} MB)")
    print(f"Base64 string size: {payload_size} characters ({payload_size / (1024*1024):.2f} MB)")
    
    payload = {
        "prompt": "Ghibli style noise test",
        "image": img_b64,
        "strength": 0.60
    }
    
    headers_auth = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    start_time = time.perf_counter()
    try:
        print("Sending large request to Spring Boot `/api/ghibli`...")
        resp = requests.post(
            f"{SPRING_BOOT_URL}/api/ghibli",
            json=payload,
            headers=headers_auth,
            timeout=180
        )
        duration = time.perf_counter() - start_time
        print(f"Response status: {resp.status_code}")
        print(f"Response duration: {duration:.3f}s")
        print(f"Response body: {resp.text[:500]}")
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"Request failed: {e} after {duration:.3f}s")

if __name__ == "__main__":
    test_large_noise()
