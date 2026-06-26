import os
import zipfile

def zip_project():
    zip_filename = 'sketchai_project.zip'
    exclude_dirs = {'venv', 'models', 'target', '.git', '.vscode', '__pycache__', 'data'}
    exclude_files = {zip_filename, 'sketchai_project_source.zip'}
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                file_path = os.path.join(root, file)
                
                # Exclude IDE state files and absolute logs
                if '.system_generated' in file_path or '.tempmediaStorage' in file_path:
                    continue
                    
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
    print(f"Project successfully zipped to {zip_filename}")

if __name__ == '__main__':
    zip_project()
