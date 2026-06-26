import os
import sys
import time
import json
import base64
import requests
import subprocess
from io import BytesIO
from PIL import Image

# 1. Create mock images for testing endpoints
print("Creating mock image and mask for endpoint verification...")
w, h = 128, 128
mock_img = Image.new("RGB", (w, h), (128, 128, 128))
# Put a colored square in the center
for y in range(40, 88):
    for x in range(40, 88):
        mock_img.putpixel((x, y), (255, 0, 0))

buffered_img = BytesIO()
mock_img.save(buffered_img, format="PNG")
img_b64 = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

# Mask for inpainting
mock_mask = Image.new("L", (w, h), 0)
for y in range(48, 80):
    for x in range(48, 80):
        mock_mask.putpixel((x, y), 255)

buffered_mask = BytesIO()
mock_mask.save(buffered_mask, format="PNG")
mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

# Read temp audit results
temp_json_path = "scratch/audit_results_temp.json"
if os.path.exists(temp_json_path):
    with open(temp_json_path, "r") as f:
        audit_results = json.load(f)
else:
    print(f"Error: Temporary audit results file '{temp_json_path}' not found!")
    sys.exit(1)

# Start FastAPI server in background
print("Starting FastAPI backend (ai_engine.py) on http://localhost:8000...")
python_path = os.path.join("venv", "Scripts", "python")
server_log = open("scratch/ai_engine.log", "w", encoding="utf-8")
server_proc = subprocess.Popen([python_path, "-u", "ai_engine.py"], stdout=server_log, stderr=subprocess.STDOUT, text=True)

# Wait for server to start up (max 30 seconds)
server_ready = False
for attempt in range(15):
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            print("FastAPI backend is UP and running!")
            server_ready = True
            break
    except Exception:
        print(f"Waiting for backend... ({attempt+1}/15)")
        time.sleep(2)

if not server_ready:
    print("Error: FastAPI backend failed to start!")
    server_proc.terminate()
    server_log.close()
    try:
        with open("scratch/ai_engine.log", "r", encoding="utf-8") as log_f:
            print("AI Engine Logs:\n", log_f.read())
    except Exception:
        pass
    audit_results["endpoint_failures"].append("All endpoints failed: FastAPI backend failed to start.")
    sys.exit(1)

# Test list
endpoints = [
    {
        "name": "Health Check",
        "url": "http://localhost:8000/health",
        "method": "GET",
        "payload": None
    },
    {
        "name": "Ghibli Image Generation",
        "url": "http://localhost:8000/api/python/ghibli",
        "method": "POST",
        "payload": {
            "prompt": "sunset",
            "image": img_b64,
            "strength": 0.50
        }
    },
    {
        "name": "Background Removal",
        "url": "http://localhost:8000/api/python/edit/remove-bg",
        "method": "POST",
        "payload": {
            "image": img_b64
        }
    },
    {
        "name": "Background Replace",
        "url": "http://localhost:8000/api/python/edit/bg-replace",
        "method": "POST",
        "payload": {
            "prompt": "starry night sky",
            "image": img_b64
        }
    },
    {
        "name": "Object Inpaint",
        "url": "http://localhost:8000/api/python/edit/inpaint",
        "method": "POST",
        "payload": {
            "prompt": "remove object",
            "image": img_b64,
            "mask": mask_b64
        }
    },
    {
        "name": "Upscaling (2X)",
        "url": "http://localhost:8000/api/python/edit/upscale",
        "method": "POST",
        "payload": {
            "image": img_b64,
            "upscale_factor": 2
        }
    }
]

print("\n=== 5. Testing REST Endpoints ===")
for ep in endpoints:
    name = ep["name"]
    url = ep["url"]
    method = ep["method"]
    payload = ep["payload"]
    
    print(f"Testing {name} ({method} {url.replace('http://localhost:8000', '')})...")
    start = time.perf_counter()
    try:
        if method == "GET":
            resp = requests.get(url, timeout=120)
        else:
            resp = requests.post(url, json=payload, timeout=120)
            
        elapsed = time.perf_counter() - start
        
        if resp.status_code == 200:
            print(f"[OK] {name} succeeded in {elapsed:.2f}s")
            # For POST requests, inspect output to verify success key
            if method == "POST":
                data = resp.json()
                if not data.get("success"):
                    raise Exception(f"API returned success=False. Error detail: {data.get('detail')}")
        else:
            raise Exception(f"HTTP Status {resp.status_code}: {resp.text}")
            
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[FAIL] {name} failed: {e}")
        audit_results["endpoint_failures"].append(f"{name}: {str(e)}")

# Terminate FastAPI backend
print("Stopping FastAPI backend...")
server_proc.terminate()
try:
    server_proc.wait(timeout=5)
except Exception:
    server_proc.kill()
server_log.close()
print("FastAPI backend stopped.")

# 6. Generate final markdown report
print("\n=== 6. Generating Final Audit Report ===")
report_md = f"""# SketchAI System Audit and Diagnostic Report

**Generated on**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Target OS**: Windows (Powershell)

---

## 1. Hardware & GPU Capabilities

- **CUDA Available**: {"✅ Yes" if audit_results["gpu_info"].get("cuda_available") else "❌ No"}
- **GPU Name**: {audit_results["gpu_info"].get("gpu_name")}
- **Total VRAM**: {audit_results["gpu_info"].get("vram_gb")}

---

## 2. Python Dependencies

- **Missing Dependencies**: {", ".join(audit_results["missing_dependencies"]) if audit_results["missing_dependencies"] else "None (All required packages are installed)"}
- **Broken Imports**: {", ".join(audit_results["broken_imports"]) if audit_results["broken_imports"] else "None (All imports work correctly)"}

---

## 3. Model Weights Audits

Below is the status of the local model weights paths:

| Model Name | Path | Local Status | Active Execution Device |
| --- | --- | --- | --- |
"""

for model_name, data in audit_results["models_verified"].items():
    exists_str = "✅ Found" if data["exists"] else "❌ Missing"
    
    # Map to the loaded device status
    device_status = "N/A"
    for m_dev, dev_info in audit_results["model_devices"].items():
        if m_dev.lower().replace(" ", "") in model_name.lower().replace(" ", "") or model_name.lower().replace(" ", "") in m_dev.lower().replace(" ", ""):
            device_status = f"{dev_info['device']} ({dev_info['status']})"
            break
            
    report_md += f"| {model_name} | `{data['path']}` | {exists_str} | {device_status} |\n"

report_md += """
---

## 4. API Endpoints Status

| Endpoint Path | Method | Status |
| --- | --- | --- |
| `/health` | GET | {"❌ Failed" if any("Health Check" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |
| `/api/python/ghibli` | POST | {"❌ Failed" if any("Ghibli" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |
| `/api/python/edit/remove-bg` | POST | {"❌ Failed" if any("Background Removal" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |
| `/api/python/edit/bg-replace` | POST | {"❌ Failed" if any("Background Replace" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |
| `/api/python/edit/inpaint` | POST | {"❌ Failed" if any("Inpaint" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |
| `/api/python/edit/upscale` | POST | {"❌ Failed" if any("Upscaling" in x for x in audit_results["endpoint_failures"]) else "✅ Succeeded"} |

### Endpoint Failure Details
"""

if audit_results["endpoint_failures"]:
    for failure in audit_results["endpoint_failures"]:
        report_md += f"- **{failure}**\n"
else:
    report_md += "_No endpoint failures detected._\n"

# Write final report to workspace
report_path = "audit_report.md"

# Read Spring Boot connectivity results
connectivity_section = ""
conn_json_path = "scratch/connectivity_results.json"
if os.path.exists(conn_json_path):
    try:
        with open(conn_json_path, "r") as f:
            conn_data = json.load(f)
        
        status_symbol = "✅ Connected" if conn_data.get("connection_success") else "❌ Connection Failed"
        connectivity_section = f"""
---

## 5. Spring Boot ↔ Python Connectivity Audit

- **Configured Python Backend URL**: `{conn_data.get("python_url")}`
- **Connect Timeout**: `{conn_data.get("connect_timeout_ms")} ms`
- **Read Timeout**: `{conn_data.get("read_timeout_ms")} ms`
- **Retry Logic**: `Up to {conn_data.get("retry_attempts")} attempts with linear backoff`
- **Connection Test Status**: **{status_symbol}**
- **Response Time**: `{conn_data.get("response_time_ms")} ms`
- **HTTP Status**: `{conn_data.get("http_status")}`
- **Response Body**: `{conn_data.get("response_body")}`
"""
        if not conn_data.get("connection_success"):
            connectivity_section += f"\n- **Connection Failure Reason**: `{conn_data.get('error_message')}`\n"
    except Exception as e:
        connectivity_section = f"""
---

## 5. Spring Boot ↔ Python Connectivity Audit

- **Error**: Failed to read connectivity results: `{str(e)}`
"""
else:
    connectivity_section = """
---

## 5. Spring Boot ↔ Python Connectivity Audit

- **Status**: Spring Boot connectivity test results not found. Please run the Spring Boot JUnit test `ConnectivityTest` first.
"""

report_md += connectivity_section

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"Final report saved to: {os.path.abspath(report_path)}")

# Also save final JSON results for referencing if needed
with open("scratch/audit_results_final.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=4)

print("\nAudit report generated successfully!")

