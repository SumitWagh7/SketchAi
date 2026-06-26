import os
import sys
import time
import json
from playwright.sync_api import sync_playwright

def run_browser_test():
    print("=== STARTING FRONTEND BROWSER AUTOMATION ===")
    
    with sync_playwright() as p:
        print("Launching headless Chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        
        # Listen for console logs
        page.on("console", lambda msg: print(f"[Browser Console] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[Browser JS Error] {err}"))
        
        # Listen for network requests & responses to capture size, latency, etc.
        network_data = {}
        
        def handle_request(request):
            if "/api/" in request.url:
                network_data[request.url] = {
                    "method": request.method,
                    "request_size": len(request.post_data or "") if request.post_data else 0,
                    "start_time": time.perf_counter()
                }
        
        def handle_response(response):
            url = response.url
            if "/api/" in url and url in network_data:
                latency = time.perf_counter() - network_data[url]["start_time"]
                try:
                    body = response.body()
                    resp_size = len(body)
                except Exception:
                    resp_size = 0
                network_data[url].update({
                    "status": response.status,
                    "response_size": resp_size,
                    "latency": latency
                })
        
        page.on("request", handle_request)
        page.on("response", handle_response)
        
        # Navigate to host
        print("Navigating to http://localhost:8080...")
        page.goto("http://localhost:8080")
        page.wait_for_timeout(2000)
        
        # Click "Start Sketching"
        print("Clicking 'Start Sketching' button...")
        page.click("button:has-text('Start Sketching')")
        page.wait_for_timeout(1000)
        
        # Login
        print("Logging in...")
        page.fill("#loginEmail", "audit@example.com")
        page.fill("#loginPassword", "Password123!")
        page.click("button:has-text('Connect Session')")
        page.wait_for_timeout(3000)
        
        # Verify studio is active
        if not page.is_visible("#appWorkspace"):
            print("[FAIL] Workspace container #appWorkspace is not visible.")
            page.screenshot(path="scratch/login_failed.png")
            browser.close()
            return
            
        print("[OK] Logged in and workspace loaded successfully.")
        
        # File Upload
        print("Uploading test image...")
        file_input = page.locator("input[id='imageInput']")
        file_input.set_input_files("scratch/test_upload.png")
        page.wait_for_timeout(2000)
        
        # Verify canvas updated
        if page.is_visible("#outputCanvas"):
            print("[OK] Canvas outputCanvas is visible.")
        else:
            print("[FAIL] Canvas not visible after file upload.")
            page.screenshot(path="scratch/upload_failed.png")
            browser.close()
            return
            
        # Click Ghibli button
        print("Clicking Ghibli Art Style button...")
        # Scroll the sidebar to reveal the button if needed
        page.evaluate("document.querySelector('.sidebar').scrollTop = 500")
        page.wait_for_timeout(500)
        
        try:
            page.click("#btnGhibli")
            print("Inference triggered. Waiting for processing loader to disappear...")
            
            # Wait for loader spinner to disappear (max 45s timeout)
            page.wait_for_selector("#mainLoader", state="hidden", timeout=45000)
            page.wait_for_timeout(2000)
            
            # Screenshot the final workspace
            os.makedirs("scratch", exist_ok=True)
            page.screenshot(path="scratch/ghibli_output_workspace.png")
            print("[OK] Screenshot saved to scratch/ghibli_output_workspace.png")
        except Exception as e:
            print(f"[ERROR during click or wait]: {e}")
            os.makedirs("scratch", exist_ok=True)
            page.screenshot(path="scratch/ghibli_timeout_failed.png")
            print("[FAIL] Screenshot saved to scratch/ghibli_timeout_failed.png")
            
            # Diagnose state of elements
            loader_style = page.evaluate("document.getElementById('mainLoader').style.display")
            loader_text = page.evaluate("document.getElementById('loaderText').innerText")
            print(f"mainLoader display: {loader_style}, loaderText: {loader_text}")
            
            toast_text = page.evaluate("Array.from(document.querySelectorAll('.toast span')).map(el => el.innerText)")
            print(f"Toast messages: {toast_text}")
            browser.close()
            raise e
        
        # Print captured Ghibli API request metrics
        ghibli_url = "http://localhost:8080/api/ghibli"
        if ghibli_url in network_data and "status" in network_data[ghibli_url]:
            metrics = network_data[ghibli_url]
            print("\n=== Browser Network Call Telemetry ===")
            print(f"URL:          {ghibli_url}")
            print(f"Method:       {metrics['method']}")
            print(f"Status Code:  {metrics['status']}")
            print(f"Request Size: {metrics['request_size']} bytes")
            print(f"Response Size:{metrics['response_size']} bytes")
            print(f"Latency:      {metrics['latency']:.3f} seconds")
            
            # Save browser metrics to a temp json for the final report
            with open("scratch/browser_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            print("[FAIL] Ghibli API call was not successfully captured in browser network requests.")
            
        browser.close()
        print("=== FRONTEND BROWSER AUTOMATION COMPLETE ===")

if __name__ == "__main__":
    run_browser_test()
