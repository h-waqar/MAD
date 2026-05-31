import json
import os

class IdManager:
    """Manages mapping between SAM2 object IDs and display labels/categories."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all mappings for a new video."""
        self.sam2_to_label = {} # {sam2_id: {"category": str, "instance_num": int}}
        self.category_counts = {} # {category: int}
        self.label_to_sam2 = {} # {label: sam2_id}

    def get_or_create_label(self, sam2_id, category):
        """Get existing label for sam2_id or create a new one for the category."""
        if sam2_id in self.sam2_to_label:
            info = self.sam2_to_label[sam2_id]
            return f"{info['category']} #{info['instance_num']}"
        
        # Create new label
        count = self.category_counts.get(category, 0) + 1
        self.category_counts[category] = count
        
        self.sam2_to_label[sam2_id] = {
            "category": category,
            "instance_num": count
        }
        label = f"{category} #{count}"
        self.label_to_sam2[label] = sam2_id
        return label

    def get_display_label(self, sam2_id):
        """Get the display label for an existing SAM2 ID."""
        if sam2_id in self.sam2_to_label:
            info = self.sam2_to_label[sam2_id]
            return f"{info['category']} #{info['instance_num']}"
        return f"Unknown #{sam2_id}"

    def set_mapping(self, sam2_id, category, instance_num):
        """Manually set a mapping (useful when loading previous state)."""
        self.sam2_to_label[sam2_id] = {
            "category": category,
            "instance_num": instance_num
        }
        label = f"{category} #{instance_num}"
        self.label_to_sam2[label] = sam2_id
        self.category_counts[category] = max(self.category_counts.get(category, 0), instance_num)

    def save_mapping(self, path):
        """Save the current mapping to a JSON file."""
        data = {
            "sam2_to_label": {str(k): v for k, v in self.sam2_to_label.items()},
            "category_counts": self.category_counts
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_mapping(self, path):
        """Load mapping from a JSON file."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.category_counts = data.get("category_counts", {})
            # Convert string keys back to int
            self.sam2_to_label = {int(k): v for k, v in data.get("sam2_to_label", {}).items()}
            self.label_to_sam2 = {f"{v['category']} #{v['instance_num']}": int(k) 
                                 for k, v in data.get("sam2_to_label", {}).items()}
            return True
        except Exception as e:
            print(f" [!] Error loading IdManager mapping: {e}")
            return False
