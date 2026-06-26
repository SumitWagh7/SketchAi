import os
import time
import sys

# Target sizes for the SDXL Turbo fp16 components
targets = {
    "unet/diffusion_pytorch_model.fp16.safetensors": 2620 * 1024 * 1024,
    "text_encoder_2/model.fp16.safetensors": 1390 * 1024 * 1024,
    "vae/diffusion_pytorch_model.fp16.safetensors": 335 * 1024 * 1024,
    "text_encoder/model.fp16.safetensors": 246 * 1024 * 1024
}

def get_component_size(subpath):
    # Check if the file is fully downloaded inside the snapshots directory of huggingface cache
    snapshots_dir = "models/stable_diffusion/models--stabilityai--sdxl-turbo/snapshots"
    if os.path.exists(snapshots_dir):
        for snapshot in os.listdir(snapshots_dir):
            full_path = os.path.join(snapshots_dir, snapshot, subpath)
            if os.path.exists(full_path):
                return os.path.getsize(full_path)
    return 0

# Map active temp files to their target component based on target size matching or active growing
# We'll monitor files in 'models/stable_diffusion' that are larger than 10MB
def get_active_downloads():
    path = "models/stable_diffusion"
    files = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.startswith("tmp"):
                f_path = os.path.join(path, f)
                try:
                    size = os.path.getsize(f_path)
                    if size > 10 * 1024 * 1024:
                        files.append((f, size))
                except Exception:
                    pass
    return sorted(files, key=lambda x: x[1], reverse=True)

def monitor():
    components = [
        ("UNet weights (diffusion_pytorch_model.fp16.safetensors)", "unet/diffusion_pytorch_model.fp16.safetensors", 2620 * 1024 * 1024),
        ("Text Encoder 2 weights (model.fp16.safetensors)", "text_encoder_2/model.fp16.safetensors", 1390 * 1024 * 1024),
        ("VAE weights (diffusion_pytorch_model.fp16.safetensors)", "vae/diffusion_pytorch_model.fp16.safetensors", 335 * 1024 * 1024),
        ("Text Encoder 1 weights (model.fp16.safetensors)", "text_encoder/model.fp16.safetensors", 246 * 1024 * 1024)
    ]

    prev_sizes = {}
    prev_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - prev_time
        if dt < 0.1:
            dt = 0.1
        prev_time = current_time

        # Check which components are completed
        completed = {}
        uncompleted_indices = []
        for i, (name, subpath, target_bytes) in enumerate(components):
            final_size = get_component_size(subpath)
            if final_size > 0:
                completed[i] = final_size
            else:
                uncompleted_indices.append(i)

        # If all completed, print 100% state and exit
        if len(completed) == len(components):
            # Clear console
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()
            print("=============================================================================")
            print("                       BROWSER DOWNLOADS - SDXL TURBO                        ")
            print("=============================================================================")
            print()
            for name, subpath, target_bytes in components:
                bar = '=' * 30
                print(f"File: {name}")
                print(f"Status: {target_bytes / (1024*1024):.1f} MB / {target_bytes / (1024*1024):.1f} MB (100.0%)")
                print(f"[{bar}]  Completed")
                print("-" * 77)
            print("\nALL DOWNLOADS COMPLETED SUCCESSFULLY!")
            break

        # Get active downloads (temp files)
        active = get_active_downloads()

        # Sort uncompleted components by target size descending
        uncompleted_indices_sorted = sorted(uncompleted_indices, key=lambda idx: components[idx][2], reverse=True)

        # Clear console
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()
        
        print("=============================================================================")
        print("                       BROWSER DOWNLOADS - SDXL TURBO                        ")
        print("=============================================================================")
        print()

        # Build output information for all components
        display_states = {}
        
        # 1. Fill completed components
        for idx in completed:
            name, _, target_bytes = components[idx]
            display_states[idx] = {
                "curr_bytes": target_bytes,
                "percent": 100.0,
                "bar": '=' * 30,
                "speed_str": "Completed",
                "eta_str": ""
            }

        # 2. Fill downloading / pending components
        for map_idx, idx in enumerate(uncompleted_indices_sorted):
            name, _, target_bytes = components[idx]
            curr_bytes = 0
            temp_name = ""
            if map_idx < len(active):
                temp_name, curr_bytes = active[map_idx]
            
            percent = min(100.0, (curr_bytes / target_bytes) * 100.0)
            
            # Speed
            speed = 0
            if temp_name:
                if temp_name in prev_sizes:
                    speed = (curr_bytes - prev_sizes[temp_name]) / dt
                prev_sizes[temp_name] = curr_bytes
                
            bar_len = 30
            filled_len = int(round(bar_len * percent / 100.0))
            bar = '=' * filled_len + ' ' * (bar_len - filled_len)
            
            speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 0 else "Connecting..."
            eta_str = "Calculating..."
            if speed > 0:
                remaining = target_bytes - curr_bytes
                eta_sec = remaining / speed
                if eta_sec > 60:
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s remaining"
                else:
                    eta_str = f"{int(eta_sec)}s remaining"
            elif curr_bytes == 0:
                eta_str = "Pending..."
                speed_str = "Queued"

            display_states[idx] = {
                "curr_bytes": curr_bytes,
                "percent": percent,
                "bar": bar,
                "speed_str": speed_str,
                "eta_str": eta_str
            }

        # Print all in original order
        for idx, (name, _, target_bytes) in enumerate(components):
            state = display_states[idx]
            curr_mb = state["curr_bytes"] / (1024 * 1024)
            target_mb = target_bytes / (1024 * 1024)
            print(f"File: {name}")
            print(f"Status: {curr_mb:.1f} MB / {target_mb:.1f} MB ({state['percent']:.1f}%)")
            if state["eta_str"]:
                print(f"[{state['bar']}]  {state['speed_str']} | {state['eta_str']}")
            else:
                print(f"[{state['bar']}]  {state['speed_str']}")
            print("-" * 77)

        print()
        print("Press Ctrl+C to stop panel monitoring.")
        sys.stdout.flush()
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)
