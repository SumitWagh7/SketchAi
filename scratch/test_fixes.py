import os
import sys
import torch
import numpy as np
from PIL import Image

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_manager import model_manager, gpu_manager

print("Testing MODNet loading...")
try:
    modnet = model_manager.get_modnet()
    print("MODNet loaded successfully!")
    
    # Run background removal test
    img = Image.new("RGB", (256, 256), (100, 200, 100))
    out = model_manager.remove_bg_modnet(img)
    print("MODNet remove_bg completed successfully!")
    print("Output size:", out.size, "Mode:", out.mode)
except Exception as e:
    import traceback
    traceback.print_exc()
