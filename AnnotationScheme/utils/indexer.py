import os
import re
import json
from pathlib import Path

def get_last_index(tracker_path):
    if not tracker_path.exists():
        return 0
    try:
        with open(tracker_path, 'r') as f:
            data = json.load(f)
            return data.get('last_assigned_index', 0)
    except (json.JSONDecodeError, IOError):
        return 0

def save_last_index(tracker_path, index):
    with open(tracker_path, 'w') as f:
        json.dump({'last_assigned_index': index}, f, indent=4)

def index_videos(assets_dir, tracker_path):
    assets_path = Path(assets_dir)
    if not assets_path.exists():
        print(f"Directory {assets_dir} does not exist.")
        return

    last_index = get_last_index(tracker_path)
    print(f"Starting indexing from index: {last_index}")

    video_extensions = ('.mp4', '.mov')
    
    # Regex to check if file is already indexed: starts with v and digits followed by underscore
    indexed_pattern = re.compile(r'^v\d+_')

    for file_path in assets_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            if not indexed_pattern.match(file_path.name):
                last_index += 1
                new_name = f"v{last_index}_{file_path.name}"
                new_path = file_path.with_name(new_name)
                
                print(f"Renaming: {file_path.name} -> {new_name}")
                file_path.rename(new_path)
                
                # Update tracker after each rename to ensure consistency if interrupted
                save_last_index(tracker_path, last_index)
            else:
                print(f"Already indexed: {file_path.name}")

    print(f"Indexing complete. Last index assigned: {last_index}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    assets_directory = PROJECT_ROOT / "assets"
    tracker_file = PROJECT_ROOT / "tracker.json"
    
    index_videos(assets_directory, tracker_file)
