import os
import shutil

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    # Target directories
    dirs = {
        "background": os.path.join(models_dir, "background"),
        "object_removal": os.path.join(models_dir, "object_removal"),
        "upscale": os.path.join(models_dir, "upscale"),
        "stable_diffusion": os.path.join(models_dir, "stable_diffusion")
    }
    
    # Create directories
    for name, path in dirs.items():
        if not os.path.exists(path):
            print(f"Creating directory: {path}")
            os.makedirs(path, exist_ok=True)
            
    # Moves map (source relative to models_dir, dest folder path)
    moves = [
        ("u2net.onnx", dirs["background"]),
        ("RealESRGAN_x4plus.pth", dirs["upscale"]),
        ("GFPGANv1.4.pth", dirs["upscale"]),
        ("models--stablediffusionapi--anything-v5", dirs["stable_diffusion"]),
        ("models--runwayml--stable-diffusion-inpainting", dirs["object_removal"]),
        ("models--TinyLlama--TinyLlama-1.1B-Chat-v1.0", dirs["stable_diffusion"]),
        ("models--google--flan-t5-small", dirs["stable_diffusion"])
    ]
    
    for item, dest_dir in moves:
        src_path = os.path.join(models_dir, item)
        dest_path = os.path.join(dest_dir, item)
        
        if os.path.exists(src_path):
            print(f"Moving {item} to {dest_dir}...")
            try:
                shutil.move(src_path, dest_path)
                print(f"Successfully moved {item}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")
        else:
            if os.path.exists(dest_path):
                print(f"Already moved or exists at target: {dest_path}")
            else:
                print(f"Source not found: {src_path}")
                
    print("Model migration completed successfully.")

if __name__ == "__main__":
    main()
