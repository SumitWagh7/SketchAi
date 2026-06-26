import os
import torch
from simple_lama_inpainting import SimpleLama

# Configure local path for offline safety
os.environ["LAMA_MODEL"] = "models/object_removal/big-lama.pt"

print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing SimpleLama on {device}...")
try:
    lama = SimpleLama(device=device)
    print("SimpleLama initialized successfully!")
    print("Model loaded locally from:", os.environ["LAMA_MODEL"])
except Exception as e:
    print("Failed to initialize SimpleLama:", str(e))
