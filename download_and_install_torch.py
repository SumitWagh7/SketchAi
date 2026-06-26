import os
import sys
import urllib.request
import subprocess

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    
    # Custom progress reporter
    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        # Print progress every 5% to avoid polluting log files
        if percent % 5 == 0 and percent > 0 and progress_hook.last_percent != percent:
            print(f"Progress: {percent}% ({downloaded / 1024 / 1024:.1f} MB of {total_size / 1024 / 1024:.1f} MB)", flush=True)
            progress_hook.last_percent = percent
            
    progress_hook.last_percent = -1
    
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print(f"Download complete: {dest}", flush=True)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", flush=True)
        return False

def main():
    torch_url = "https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl"
    vision_url = "https://download.pytorch.org/whl/cu121/torchvision-0.17.2%2Bcu121-cp311-cp311-win_amd64.whl"
    
    torch_dest = "torch-2.2.2+cu121-cp311-cp311-win_amd64.whl"
    vision_dest = "torchvision-0.17.2+cu121-cp311-cp311-win_amd64.whl"
    
    # Check if files already downloaded
    if not os.path.exists(torch_dest):
        if not download_file(torch_url, torch_dest):
            sys.exit(1)
            
    if not os.path.exists(vision_dest):
        if not download_file(vision_url, vision_dest):
            sys.exit(1)
            
    print("Installing wheels...", flush=True)
    pip_path = os.path.join("venv", "Scripts", "pip")
    
    try:
        # Install torch
        print("Installing torch...", flush=True)
        subprocess.run([pip_path, "install", torch_dest, "--no-deps", "--force-reinstall"], check=True)
        
        # Install torchvision
        print("Installing torchvision...", flush=True)
        subprocess.run([pip_path, "install", vision_dest, "--no-deps", "--force-reinstall"], check=True)
        
        print("PyTorch with GPU CUDA support installed successfully!", flush=True)
        
        # Clean up files
        print("Cleaning up wheel files...", flush=True)
        os.remove(torch_dest)
        os.remove(vision_dest)
        print("Cleanup complete.", flush=True)
        
    except Exception as e:
        print(f"Installation failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
