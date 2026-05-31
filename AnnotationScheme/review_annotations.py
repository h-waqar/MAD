import os
import json
import cv2
import argparse
from pathlib import Path
import numpy as np

def draw_annotations(image, annotations, categories_map):
    for ann in annotations:
        cat_id = ann.get('category_id')
        cat_name = categories_map.get(cat_id, "Unknown")
        
        # Draw Bounding Box
        bbox = ann.get('bbox')
        if bbox and len(bbox) == 4:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw Segmentation
        seg = ann.get('segmentation')
        if seg:
            for poly in seg:
                if len(poly) >= 4:
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

def review_video_annotations(coco_dir):
    coco_path = Path(coco_dir)
    # Recursively find all annotations.json
    # Expecting: [Category]/[video_name]/annotations.json
    # But recursively finding any annotations.json is safer
    ann_files = list(coco_path.rglob('annotations.json'))
    
    if not ann_files:
        print(f"No annotations.json found in {coco_dir}")
        return

    print(f"Found {len(ann_files)} annotation sets.")

    for ann_file in ann_files:
        print(f"\nReviewing: {ann_file}")
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])
        
        categories_map = {cat['id']: cat['name'] for cat in categories}
        
        # Group annotations by image_id
        img_to_anns = {}
        for ann in annotations:
            iid = ann['image_id']
            img_to_anns.setdefault(iid, []).append(ann)
            
        images.sort(key=lambda x: x.get('file_name', ''))
        
        idx = 0
        while idx < len(images):
            img_info = images[idx]
            # images folder is usually peer to annotations.json or in coco_dir/images
            # Based on D-05: results_coco_format/[Category]/[video_name]/annotations.json
            # And Task 2: images directory exists at coco_data/images
            # Wait, Task 2 says Path(os.path.join(args.coco_data, 'images')).mkdir(...)
            # So images are all in results_coco_format/images/
            
            # Find the root results_coco_format directory
            # If ann_file is results_coco_format/Category/Video/annotations.json
            # then images are in results_coco_format/images/
            results_root = ann_file.parent.parent.parent
            img_path = results_root / "images" / img_info['file_name']
            
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                idx += 1
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}")
                idx += 1
                continue
                
            draw_annotations(img, img_to_anns.get(img_info['id'], []), categories_map)
            
            winname = f"Review - {ann_file.parent.name}"
            cv2.imshow(winname, img)
            
            print(f"Frame {idx+1}/{len(images)}: {img_info['file_name']}", end='\r')
            
            key = cv2.waitKeyEx(0)
            
            # Left arrow (Linux/Mac/Windows waitKeyEx vary, but common ones are covered)
            if key in (ord('p'), 81, 2424832, 65361): # p or left arrow
                idx = max(0, idx - 1)
            # Right arrow
            elif key in (ord('n'), 83, 2555904, 65363, 32): # n or right arrow or space
                idx += 1
            elif key in (ord('q'), 27): # q or esc
                # D-05: Exit prompt with Cancel option
                question_img = img.copy()
                h, w, _ = question_img.shape
                test = "Stop reviewing this video? (y/n) [c] Cancel"
                (tw, th), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                
                # Draw a dark background for the prompt
                cv2.rectangle(question_img, (w//2 - tw//2 - 10, h//2 - th - 20),
                             (w//2 + tw//2 + 10, h//2 + 20), (0, 0, 0), -1)
                cv2.putText(question_img, test, (w//2 - tw//2, h//2), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(winname, question_img)
                
                exit_key = cv2.waitKey(0) & 0xFF
                if exit_key in (ord('y'), ord('Y')):
                    cv2.destroyWindow(winname)
                    break
                else:
                    # 'n' or 'c' or any other key returns to review
                    continue
            else:
                idx += 1
        
        print("\nFinished this video.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review annotations in COCO format")
    parser.add_argument("--coco_dir", type=str, default="results_coco_format", help="Root directory of COCO results")
    args = parser.parse_args()
    
    review_video_annotations(args.coco_dir)
