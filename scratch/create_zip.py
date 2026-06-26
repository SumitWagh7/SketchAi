import os
import zipfile

def zip_project():
    zip_name = "sketchai_project_source.zip"
    exclude_dirs = {"venv", "models", "target", ".vscode", ".git", "__pycache__", "data", "gfpgan"}
    exclude_exts = {".whl", ".zip"}
    
    print("Creating zip archive...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to prevent os.walk from entering excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_exts:
                    continue
                
                file_path = os.path.join(root, file)
                # Relative path for the zip archive
                rel_path = os.path.relpath(file_path, '.')
                print(f"Adding: {rel_path}")
                zipf.write(file_path, rel_path)
                
    print(f"Zip archive created successfully: {zip_name}")

if __name__ == "__main__":
    zip_project()
