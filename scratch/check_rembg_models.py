import os
import sys

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_manager import gpu_manager, MODEL_DIR
import rembg
import torch

print("U2NET_HOME env:", os.environ.get("U2NET_HOME"))
print("MODEL_DIR exists:", os.path.exists(MODEL_DIR))

# Check background models
bg_dir = os.path.join(MODEL_DIR, "background")
print("Files in background dir:", os.listdir(bg_dir) if os.path.exists(bg_dir) else "Directory not found")

try:
    print("Trying to instantiate a rembg session with birefnet-general...")
    providers = gpu_manager.get_onnx_providers()
    print("Providers:", providers)
    # We can try to new_session with "birefnet-general"
    session = rembg.new_session("birefnet-general", providers=providers)
    print("Successfully created birefnet-general session!")
    print("Files in background dir after load:", os.listdir(bg_dir))
except Exception as e:
    print("Failed to load birefnet-general session:", str(e))
