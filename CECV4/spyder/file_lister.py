from pathlib import Path

def get_file_names(folder_name="CM_COMBINED_JSON"):
    # Resolve the folder path relative to this Python script's location
    base_dir = Path(__file__).resolve().parent
    target_folder = base_dir / folder_name

    # Collect only file names (ignoring any subfolders)
    file_names = [item.name for item in target_folder.iterdir() if item.is_file()]
    
    return file_names

if __name__ == "__main__":
    files = get_file_names()
    print(files)