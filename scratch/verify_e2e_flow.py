import os
import sys
import time
import json
import base64
import requests
from PIL import Image
from io import BytesIO

SPRING_BOOT_URL = "http://localhost:8080"
PYTHON_AI_URL = "http://localhost:8000"

def log_stage(stage, request_size, response_size, duration, status, exception="None"):
    print(f"\n[{stage}]")
    print(f"  Request Size:  {request_size}")
    print(f"  Response Size: {response_size}")
    print(f"  Duration:      {duration:.3f}s")
    print(f"  HTTP Status:   {status}")
    print(f"  Exception:     {exception}")

def run_audit():
    print("=== STARTING END-TO-END GHIBLI INTEGRATION AUDIT ===")

    # Create a test image (512x512)
    print("Creating test image (512x512)...")
    img = Image.new("RGB", (512, 512), color=(73, 109, 137))
    # Add some graphics
    for x in range(150, 362):
        for y in range(150, 362):
            img.putpixel((x, y), (255, 255, 255))
            
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    raw_size_bytes = len(img_bytes)
    
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    b64_size_chars = len(img_b64)
    print(f"Test image generated. Raw size: {raw_size_bytes} bytes. Base64 size: {b64_size_chars} chars.")

    # 1. Authenticate with Spring Boot (Login or Register)
    print("\n--- Authenticators/Users setup on Spring Boot ---")
    headers = {"Content-Type": "application/json"}
    auth_payload = {
        "username": "audit_user",
        "email": "audit@example.com",
        "password": "Password123!",
        "phone": "1234567890"
    }
    
    token = None
    try:
        # Try register
        print("Registering audit user...")
        resp = requests.post(f"{SPRING_BOOT_URL}/api/auth/register", json=auth_payload, headers=headers, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("success"):
            token = data["data"]["token"]
            print("Registration successful.")
        else:
            print(f"Registration status: {resp.status_code}, attempting login instead...")
            resp = requests.post(f"{SPRING_BOOT_URL}/api/auth/login", json={
                "email": "audit@example.com",
                "password": "Password123!"
            }, headers=headers, timeout=10)
            data = resp.json()
            if resp.status_code == 200 and data.get("success"):
                token = data["data"]["token"]
                print("Login successful.")
            else:
                print(f"Login failed: {data}")
    except Exception as e:
        print(f"Failed to authenticate with Spring Boot: {e}")
        return False

    if not token:
        print("Error: Could not retrieve JWT Token for authentication.")
        return False

    # 2. Audit Stage: Frontend -> Spring Boot -> Python -> SDXL Turbo -> Spring Boot -> Frontend
    print("\n--- Auditing Main Ghibli Request Flow ---")
    
    ghibli_payload = {
        "prompt": "Ghibli style, beautiful green meadow, blue skies",
        "image": img_b64,
        "strength": 0.60
    }
    
    payload_str = json.dumps(ghibli_payload)
    request_payload_size_bytes = len(payload_str.encode('utf-8'))
    
    headers_with_auth = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    start_time = time.perf_counter()
    http_status = "N/A"
    response_size = "N/A"
    exception_info = "None"
    response_data = None
    
    try:
        print("Sending /api/ghibli request to Spring Boot...")
        resp = requests.post(
            f"{SPRING_BOOT_URL}/api/ghibli",
            json=ghibli_payload,
            headers=headers_with_auth,
            timeout=180
        )
        duration = time.perf_counter() - start_time
        http_status = str(resp.status_code)
        response_size = f"{len(resp.content)} bytes"
        
        if resp.status_code == 200:
            response_data = resp.json()
            print("Request successful!")
        else:
            exception_info = f"HTTP Error status code: {resp.status_code}. Details: {resp.text[:200]}"
            print(f"Request failed with status: {resp.status_code}")
    except Exception as e:
        duration = time.perf_counter() - start_time
        exception_info = str(e)
        print(f"Request threw exception: {e}")

    # Log Stage 1: Client -> Spring Boot request & response
    log_stage(
        "Client <-> Spring Boot Flow",
        request_size=f"{request_payload_size_bytes} bytes (JSON payload)",
        response_size=response_size,
        duration=duration,
        status=http_status,
        exception=exception_info
    )

    if not response_data or "image" not in response_data:
        print("Flow failed at Spring Boot <-> Python connection or Inference step. Checking direct Python endpoint...")
        # Direct Python Call audit
        test_direct_python(img_b64)
        return False

    # 3. Base64 Verification
    print("\n--- Verifying Returned Image Base64 Decodability ---")
    ret_image_b64 = response_data["image"]
    try:
        decoded_bytes = base64.b64decode(ret_image_b64)
        decoded_img = Image.open(BytesIO(decoded_bytes))
        print(f"[OK] Decoded returned base64 successfully. Format: {decoded_img.format}, Size: {decoded_img.size}")
    except Exception as e:
        print(f"[FAIL] Base64 decoding or Image.open failed: {e}")

    # 4. Large Image Size Limit Edge Case Test
    test_large_image_limit(token)

    return True

def test_direct_python(img_b64):
    print("\n--- Auditing Direct Call to Python AI Service (/api/python/ghibli) ---")
    payload = {
        "prompt": "Ghibli style, direct test",
        "image": img_b64,
        "strength": 0.60
    }
    payload_str = json.dumps(payload)
    headers = {"Content-Type": "application/json"}
    
    start_time = time.perf_counter()
    try:
        resp = requests.post(
            f"{PYTHON_AI_URL}/api/python/ghibli",
            json=payload,
            headers=headers,
            timeout=120
        )
        duration = time.perf_counter() - start_time
        print(f"Direct Python status: {resp.status_code}, time: {duration:.3f}s")
        if resp.status_code != 200:
            print("Response details:", resp.text)
    except Exception as e:
        print(f"Direct Python request failed: {e}")

def test_large_image_limit(token):
    print("\n--- Testing Large Image Payload (Size Limits Validation) ---")
    # Let's generate a larger image to test limits (e.g. 3000x3000px)
    print("Generating large test image (3000x3000)...")
    large_img = Image.new("RGB", (3000, 3000), color=(100, 100, 100))
    large_byte_arr = BytesIO()
    large_img.save(large_byte_arr, format='PNG')
    large_bytes = large_byte_arr.getvalue()
    
    large_b64 = base64.b64encode(large_bytes).decode('utf-8')
    payload = {
        "prompt": "Ghibli style large test",
        "image": large_b64,
        "strength": 0.60
    }
    
    payload_size = len(json.dumps(payload).encode('utf-8'))
    print(f"Large Payload Size: {payload_size / (1024*1024):.2f} MB ({payload_size} bytes)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    start_time = time.perf_counter()
    try:
        resp = requests.post(
            f"{SPRING_BOOT_URL}/api/ghibli",
            json=payload,
            headers=headers,
            timeout=180
        )
        duration = time.perf_counter() - start_time
        print(f"Large Payload Response Code: {resp.status_code}, time: {duration:.3f}s")
        if resp.status_code != 200:
            print(f"Error Details: {resp.text[:500]}")
    except Exception as e:
        duration = time.perf_counter() - start_time
        print(f"Large request failed with exception: {e} after {duration:.3f}s")

if __name__ == "__main__":
    run_audit()
