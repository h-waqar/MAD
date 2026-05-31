"""
video to frames utils
This module provides utility functions to extract frames from a video file.
This script contains the following actions:
1. Convert videos in a directory to frames and save them as images.
2. Rename the extracted frames to a sequential format.
3. given the maintenace action directory and a databse, extract all the saved annotation bbs, and save them as a coco format

"""
import glob
import os
import cv2
from pathlib import Path
import sqlite3
import json
from types import SimpleNamespace
import subprocess
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from collections import defaultdict
from utils import utils
import configs

start_point = None
end_point = None
is_drawing = False
image_copy = None
clean_state = None
winname = ""



def convert_video_to_frame_v1(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    # Build the ffmpeg command
    command = [
        "ffmpeg",
        "-loglevel", "error",  # Suppress all non-error messages
        "-i", video_path,  # Input video file
        "-q:v", "2",  # Set output quality; lower is better (range is roughly 2–31)
        "-start_number", "0",  # Start naming frames from 00000, 00001, ...
        f"{output_path}/%05d.jpg"  # Output filename pattern
    ]

    # Run the command
    subprocess.run(command, check=True)


def convert_video_to_frames_v0(video_path, output_dir, frame_number=0, resize_to=None):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str or Path): Path to the input video.
        output_dir (str or Path): Directory to save the extracted frames.
        resize_to (tuple): Optional (width, height) to resize frames.

    Returns:
        int: Number of frames extracted.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    saved_frame_idx = frame_number

    while True:
        success, frame = cap.read()
        if not success:
            break

        if count % 3 == 0:
            frame_filename = output_dir / f"{saved_frame_idx:06}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_frame_idx += 1
        count += 1

    cap.release()
    print(f"[INFO] Extracted {saved_frame_idx} frames from {video_path.name} to {output_dir}")
    return saved_frame_idx


def rename_files(root_dir):
     for tool_dir in os.listdir(root_dir):
        if '.DS_' in tool_dir:
            continue
        tool_path = os.path.join(root_dir, tool_dir)
        frame_number = 0

        frames = [f for f in os.listdir(tool_path) if f.endswith(('.jpg', '.png')) and '.DS_Sto' not in f]
        for frame in frames:
            os.rename(os.path.join(tool_path, frame), os.path.join(tool_path, f'{frame_number:06d}.jpg'))
            frame_number += 1
        
        print(f"[INFO] {tool_dir} has {frame_number} images")


def rename_images_to_numeric_order(dataset_dir):
    images_dir = Path(dataset_dir) / "images"
    ann_path = Path(dataset_dir) / "annotations.json"

    assert images_dir.exists(), f"No 'images/' folder found in {dataset_dir}"
    assert ann_path.exists(), f"No 'annotations.json' found in {dataset_dir}"

    # Load COCO annotations
    with open(ann_path, "r") as f:
        coco = json.load(f)

    # Process all images defined in the COCO 'images' field
    new_name_map = {}
    for idx, image_info in enumerate(sorted(coco["images"], key=lambda x: x["id"])):
        old_name = image_info["file_name"]
        old_path = images_dir / old_name
        if not old_path.exists():
            print(f"⚠️ Warning: File not found: {old_name}")
            continue

        new_name = f"{idx:05d}.jpg"
        new_path = images_dir / new_name

        # Rename the image file
        os.rename(old_path, new_path)

        # Update the annotation entry
        image_info["file_name"] = new_name
        new_name_map[old_name] = new_name

    # Save updated annotations
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Renamed and updated {len(new_name_map)} images.")


def video_dirs_to_frame(video_dir):
    """
    Convert all videos in a directory to frames.
    video_dir
    |- tool1
    |   |- video1.mp4

    converted to
    video_dir
    |- tool1
    |   |- img1.jpg
    |   |- img2.jpg
    """

    for tool_dir in os.listdir(video_dir):
        if '.DS_' in tool_dir:
            continue
        tool_path = os.path.join(video_dir, tool_dir)
        frame_number = 0

        video_files = [f for f in os.listdir(tool_path) if f.endswith(('.mp4', '.MP4')) and '.DS_Sto' not in f]
        for video_file in video_files:
        
            video_path = os.path.join(tool_path, video_file)
            frame_number = convert_video_to_frames_v0(video_path, tool_path, frame_number)

            # remove the video file after conversion
            os.remove(video_path)


def dir_of_videos_to_frames(video_dir):
    """
    convert a directory contains .mP4 videos to images
    """

    if not os.path.exists(video_dir):
        print(f" Err: {video_dir} does not exists.")
        return
    saved_dir = os.path.join(video_dir, "result")
    os.makedirs(saved_dir, exist_ok=True)
    videos = glob.glob(os.path.join(video_dir, "*.MP4"))

    for v in videos:
        v_name = os.path.split(v)[-1].split('.')[0]
        v_result = os.path.join(saved_dir, v_name)
        os.makedirs(v_result, exist_ok=True)
        convert_video_to_frames_v0(v, v_result)

def functionality_one_navigator(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        print("invalid or not found")
        return 

    entries = os.listdir(path)
    if not entries:
        print("empty folder")
        return
    video_extensions = [".MP4", ".mp4"]
    contains_video = False
    contains_folders_with_videos = False
    only_folders = True

    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            only_folders = False
            _, ext = os.path.splitext(entry)
            if ext.lower() in video_extensions:
                contains_video = True
        elif os.path.isdir(full_path):
            for sub in os.listdir(full_path):
                sub_path = os.path.join(full_path, sub)
                if os.path.isfile(sub_path):
                    _, ext = os.path.splitext(sub)
                    if ext.lower() in video_extensions:
                        contains_folders_with_videos = True

    if contains_video:
        dir_of_videos_to_frames(path)
    elif contains_folders_with_videos:
        video_dirs_to_frame(path)
    elif only_folders:
        return "folders_only"
    else:
        return "unknown structure"




def set_annotation_id_to_image_id(annotation_path, output_path=None):
    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    for ann in coco['annotations']:
        ann['image_id'] -= 1

    # If no output path is given, overwrite original
    if output_path is None:
        output_path = annotation_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Annotation IDs set to image_id. Saved to: {output_path}")


def summarize_coco_annotation(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    categories = data.get('categories', [])
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Count annotations per category
    category_counts = defaultdict(int)
    image_ids = set()
    for ann in data.get('annotations', []):
        cat_id = ann['category_id']
        category_counts[cat_id] += 1
        image_ids.add(ann['image_id'])

    print(f"🖼 Total Images: {num_images}")
    print(f"🔲 Total Bounding Boxes (Annotations): {num_annotations}")
    print(f"🔧 Total Unique Tools (Classes): {len(categories)}")
    print("\n📊 Bounding Box Count per Tool:")
    for cat_id, count in category_counts.items():
        print(f" - {category_id_to_name.get(cat_id, 'Unknown')} (ID {cat_id}): {count} boxes/segmentations")

    print(f"\n🧾 Unique Image IDs in Annotations: {len(image_ids)}")
    if num_images != len(image_ids):
        print("⚠️ Warning: Not all images have annotations!")

    # Prepare data for plotting
    tool_names = [category_id_to_name[cat_id] for cat_id in sorted(category_counts)]
    counts = [category_counts[cat_id] for cat_id in sorted(category_counts)]

    plt.figure(figsize=(12, 6))
    plt.bar(tool_names, counts, color='skyblue')
    plt.xlabel('Tool Classes')
    plt.ylabel('Total Bounding Boxes')
    plt.title('Bounding Box Count per Tool Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, is_drawing, image_copy, winname, clean_state, segment_by_points

    temp_image = image_copy.copy()  # Always start with a clean copy of the image

    # Draw crosshair lines for the mouse pointer
    cv2.line(temp_image, (x, 0), (x, temp_image.shape[0]), (255, 0, 0), 1)  # Vertical line (y-axis)
    cv2.line(temp_image, (0, y), (temp_image.shape[1], y), (255, 0, 0), 1)  # Horizontal line (x-axis)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing: Record the starting point
        image_copy = clean_state.copy()
        is_drawing = True
        start_point = [x, y]

    elif event == cv2.EVENT_RBUTTONDOWN:
        image_copy = clean_state.copy()
        is_drawing = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        # Show the updated image
        if is_drawing:
            # Update the rectangle dynamically as the mouse moves
            end_point = [x, y]
            cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(winname, temp_image)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        # Finish drawing: Record the end point and finalize the bounding box
        is_drawing = False
        end_point = [x, y]
        cv2.rectangle(image_copy, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(winname, image_copy)
        cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # print(f"Bounding Box: Start={start_point}, End={end_point}")


    # Show the updated image with crosshair
    cv2.imshow(winname, temp_image)
    cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def add_manually_bb(frame_image):
    global image_copy, clean_state, start_point, end_point, winname

    image_copy = frame_image.copy()  # Create a copy for resetting during drawing
    clean_state = frame_image.copy()
    cv2.imshow(winname, frame_image)
    cv2.setMouseCallback(winname, mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rescale_bbox(bbox, orig_size, new_size):
    """
    Rescale COCO bbox [x_min, y_min, width, height] from original image size to new image size.
    
    Parameters:
        bbox (list or tuple): [x_min, y_min, x_max, y_max]
        orig_size (tuple): (orig_w, orig_h)
        new_size (tuple): (new_w, new_h)
        
    Returns:
        list: [new_x_min, new_y_min, new_width, new_height]
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y

    width = new_x2 - new_x1
    height = new_y2 - new_y1

    return [
        new_x1,
        new_y1,
        width,
        height
    ]


def visualize_random_sample_from_coco(annotation_path, image_dir, tool=None, num_samples=10):
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Build image_id to file_name mapping
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    
    # Handle both numeric IDs and string names in categories
    category_mapping = {}
    if 'categories' in data:
        for cat in data['categories']:
            if isinstance(cat, dict) and 'id' in cat and 'name' in cat:
                category_mapping[cat['id']] = cat['name']
            elif isinstance(cat, str):
                # If category is just a string, use it as both ID and name
                category_mapping[cat] = cat
    
    # Define colors for different types (BGR format for OpenCV)
    colors = {
        'tool': (0, 255, 0),      # Green for tools
        'hand': (0, 0, 255),      # Red for hands
        'device': (255, 0, 0),    # Blue for devices
        'default': (0, 255, 255)  # Yellow for unknown
    }
    
    def get_category_info(ann):
        """Extract category name and type from annotation"""
        cat_id = ann.get('category_id', 'Unknown')
        
        # Handle string category IDs
        if isinstance(cat_id, str):
            category_name = cat_id
        else:
            category_name = category_mapping.get(cat_id, f'ID_{cat_id}')
        
        # Determine category type based on name
        name_lower = category_name.lower()
        if 'hand' in name_lower:
            cat_type = 'hand'
        elif any(device in name_lower for device in ['phone', 'tablet', 'computer', 'screen', 'monitor']):
            cat_type = 'device'
        else:
            cat_type = 'tool'
            
        return category_name, cat_type
    
    # Build annotations per image_id
    annotations_by_image = {}
    tool_images = {}
    for ann in data['annotations']:
        category_name, _ = get_category_info(ann)
        
        if tool and category_name != tool:
            continue
            
        image_id = ann['image_id']
        tool_images[image_id] = image_id_to_file[image_id]
        annotations_by_image.setdefault(image_id, []).append(ann)

    # Use filtered images if tool specified, otherwise sample from all
    target_images = tool_images if tool else image_id_to_file
    sampled_image_ids = random.sample(list(target_images.keys()), min(num_samples, len(target_images)))

    for image_id in sampled_image_ids:
        file_name = target_images[image_id]
        image_path = os.path.abspath(os.path.join(image_dir, file_name))
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        anns = annotations_by_image.get(image_id, [])
        for ann in anns:
            category_name, cat_type = get_category_info(ann)
            color = colors.get(cat_type, colors['default'])
            
            # Handle bounding boxes
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = map(int, map(np.ceil, ann['bbox']))
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Put category name with background for better visibility
                text_size = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                cv2.putText(image, category_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Handle segmentation if available
            if 'segmentation' in ann and ann['segmentation']:
                utils.safe_draw_polygons(image, ann['segmentation'], color=color)

        # Show image with category type counts
        ann_counts = {}
        for ann in anns:
            _, cat_type = get_category_info(ann)
            ann_counts[cat_type] = ann_counts.get(cat_type, 0) + 1
        
        count_text = " | ".join([f"{k}: {v}" for k, v in ann_counts.items()])
        window_name = f"Image ID: {image_id} - {file_name} ({count_text})"
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def show_saved_results(image, bbox):
    """
    Show the saved image with the bounding box.
    
    Parameters:
        image (numpy.ndarray): The image to display.
        bbox (list or tuple): Bounding box in COCO format [x_min, y_min, width, height].
    """
    x_min, y_min, width, height = map(int, bbox)
    x_max = x_min + width
    y_max = y_min + height

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    instr = "[ANY] Accept  [q] dont save"
    cv2.putText(image, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 3, cv2.LINE_AA)

    # Show the image
    winname = f'Result - {bbox}'
    cv2.imshow(winname, image)
    # -----------------------------------  Edditing bb
    key = cv2.waitKey(0) & 0xFF  # wait for *one* key
    if key in (ord('q'), 27):  # ESC or 'q' to quit
        cv2.destroyAllWindows()
        return False
        
    return True  # Indicate that the user accepted the bounding box



def check_if_video_is_done(v_path, images):

    image_names = [img['file_name'].split('_')[0] for img in images]
    video_name = os.path.basename(v_path).split('.')[0]

    return video_name in image_names


# ----------------------------------------------------------------------------------------------- Functionallity 3
def prepare_env():
    """
    Prepare the environment for the extracting of bbs from the maintenance action dataset.
    """
    print("[INFO] Preparing environment")

    args = SimpleNamespace(
        directory_prefix=None,
        working_dir=None,
        output_dir="annotation_results",
        image_dir=None,
        annotations=None,
        annotations_output=None,
        db_rows=None,
        tracker={},  # save the last working row id
    )

    print(' # prepare file: ', end='')
    directory_prefix = '/Users/saeednaamneh/Library/CloudStorage/GoogleDrive-yosef.naamneh@gmail.com/My Drive/AR-MAINTENANCE/HAR'
    db_path = os.path.join(directory_prefix, 'maintenance_dataset.db')
    table_name = "MaintenanceActions_metadata" 
    output_dir = "detection_dataset"
    working_dir = 'working_dir'
    os.makedirs(working_dir, exist_ok=True)
    # Parameters (change these as needed)
    output_dir = "TOOLs_dataset"
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    annotations_output = os.path.join(output_dir, "annotations.json")
    print(f"[DONE] image_dir: {image_dir}, annotations_output: {annotations_output}")

    print(' # read database: ', end='')
    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query your table
    # 'wrench', 'ratchet', 'adjustable spanner', 'hammer', 'tapemeasure', 'allen', 'plier', 'screwdriver', 'cutting knife', 'drill', 'electrical screwdriver'
    cursor.execute(f"""
    SELECT id, video_path, length_, resolution, tool, metadata
    FROM {table_name}
    """)
    rows = cursor.fetchall()
    rows.sort(key=lambda x: x[0])  # Sort by id
    print(f"[DONE] {len(rows)} rows found")
    
    print(' # prepare tracker: ', end='')
    args.directory_prefix = directory_prefix
    args.working_dir = working_dir
    args.output_dir = output_dir
    args.image_dir = image_dir
    args.annotations_output = annotations_output
    args.db_rows = rows

    if os.path.exists(annotations_output):
        with open(annotations_output, 'r') as f:
            args.annotations = json.load(f)
    else:
        args.annotations = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "SL", "supercategory": "tool"},
                {"id": 1, "name": "adjustable spanner", "supercategory": "tool"},
                {"id": 2, "name": "allen", "supercategory": "tool"},
                {"id": 3, "name": "drill", "supercategory": "tool"},
                {"id": 4, "name": "hammer", "supercategory": "tool"},
                {"id": 5, "name": "plier", "supercategory": "tool"},
                {"id": 6, "name": "ratchet", "supercategory": "tool"},
                {"id": 7, "name": "screwdriver", "supercategory": "tool"},
                {"id": 8, "name": "tapemeasure", "supercategory": "tool"},
                {"id": 9, "name": "wrench", "supercategory": "tool"},
            ],
        }
    if os.path.exists(os.path.join(output_dir, 'tracker.json')):
        with open(os.path.join(output_dir, 'tracker.json'), 'r') as f:
            args.tracker = json.load(f)
    else:
        args.tracker = {
            'edit': [],
            'error': [],
            'last_id': 0,
            "last_category_id": 11, 
             # Start from 10 to avoid conflicts with existing categories
            }

    args.category_mapping_id = {cat['name']: cat['id'] for cat in args.annotations['categories']}
    print(f"[DONE]")

    # Add any necessary setup code here
    print("[INFO] Environment prepared.")

    return args


def extractor(new_shape=(640, 360)):
    global start_point, end_point, is_drawing, image_copy, clean_state, winname

    args = prepare_env()
    args.tracker['edit'] = []
    args.tracker['error'] = []
    args.tracker['last_row'] = 0
    start_point = None
    end_point = None
    # COCO fields
    images, annotations = args.annotations['images'], args.annotations['annotations']
    categories = [cat['name'] for cat in args.annotations['categories']]
    img_id = args.tracker['last_id']
    ann_id = args.tracker['last_id']
    # for test only: take 5 random videos
    # random_rows = random.sample(args.db_rows, min(20, len(args.db_rows)))
    # print(len(random_rows), "random rows selected from the database")
    for row_idx, row in enumerate(args.db_rows):
        id_, video_path, length_, resolution, tool, meta_json = row
        if tool == "NULL":
            continue
        try:
            meta = json.loads(meta_json)
            tool_bbs = meta.get("tool_bbs", {})
            if not tool_bbs or isinstance(tool_bbs, list):
                args.tracker['edit'].append(id_)
                continue
        except Exception as e:
            print(f"Error parsing metadata for row {id_}: {e}")
            args.tracker['error'].append(id_)
            continue

         # Category mapping
        if tool not in categories:
            if 'electrical' in tool.lower():
                tool = 'drill'
            else:    
                new_category = {
                    "id": args.tracker['last_category_id'],
                    "name": tool,
                    "supercategory": "tool"
                }
                args.annotations['categories'].append(new_category)
                categories.append(tool)
                args.tracker['last_category_id'] += 1
        
        # Video settings
        w, h = map(int, resolution.split("x"))
        cap = cv2.VideoCapture(os.path.join(args.directory_prefix, video_path))
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            continue

        convert_video_to_frame_v1(os.path.join(args.directory_prefix, video_path), args.working_dir)
        # Create a directory for the tool if it doesn't exist
        for frame_str, bb in tool_bbs.items():
            if isinstance(bb[0], list):
                bb = bb[0]
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            take_frame = True
            start_point, end_point = None, None
            # resize and save the frame as .jpg in the target path
            frame_idx = int(frame_str)
            frame_orig_path = os.path.join(args.working_dir, f'{frame_idx:05d}.jpg')
            frame_name = f"{os.path.basename(video_path).split('.')[0]}_{frame_idx:05d}.jpg"
            resized_img = cv2.resize(cv2.imread(frame_orig_path), dsize=new_shape, interpolation=cv2.INTER_LINEAR)
            target_path = os.path.join(args.image_dir, frame_name)

            # Convert bb to COCO format [x, y, w, h]
            x_c, y_c, bw, bh = bb
            x_c *= w 
            y_c *= h
            bw *= w
            bh *= h

            bb = [x_c, y_c, bw, bh]
            # Convert to top-left (x1, y1) and bottom-right (x2, y2)
            x1 = int(bb[0] - bb[2] / 2)
            y1 = int(bb[1] - bb[3] / 2)
            x2 = int(bb[0] + bb[2] / 2)
            y2 = int(bb[1] + bb[3] / 2)
            new_w = abs(x2 - x1)
            new_h = abs(y2 - y1)

            # Read image
            img = cv2.imread(frame_orig_path)
            showing_img = img.copy()

            # Draw the rectangle
            cv2.rectangle(showing_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box, thickness 2
            
            # --------------------------------------------------  instructions
            instr = "[a/Enter] Accept   [e] Edit   [q] Quit"
            cv2.putText(showing_img, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 3, cv2.LINE_AA)


             # Show the image
            winname = f'v: {row_idx + 1}/{len(args.db_rows)} - frame- {frame_idx} - {tool}'
            cv2.imshow(winname, showing_img)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # -----------------------------------  Edditing bb
            key = cv2.waitKey(0) & 0xFF  # wait for *one* key
            if key in (ord('q'), 27):
                take_frame = False
                cv2.destroyAllWindows()
                break
            
            if key in (ord('e'),):  # EDIT BB
                cv2.destroyAllWindows()
                # winname = f'{img_info['file_name']} -- EDITTING'
                add_manually_bb(img)

                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    new_w, new_h = abs(x2 - x1), abs(y2 - y1)
                
                else:
                    print("No bounding box drawn. Skipping this frame.")
                    take_frame = False
                    
            
            if take_frame:
                cv2.imwrite(target_path, resized_img)
                bb = rescale_bbox([x1, y1, x2, y2], (w, h), new_shape)
                save_it = show_saved_results(resized_img, bb)

                if save_it:
                    print(f"Saving frame {frame_name} with bounding box: {bb}")

                    images.append({
                        "id": img_id,
                        "file_name": frame_name,
                        "width": new_shape[0],
                        "height": new_shape[1]
                    })
                    # Save the annotation
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": args.category_mapping_id[tool],  # Use the last added category
                        "bbox":bb,
                        "area": bb[-1] * bb[-2],
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
                    img_id += 1
                    args.tracker['last_id'] = img_id
                    coco_json = {
                            "images": images,
                            "annotations": annotations,
                            "categories": args.annotations['categories']
                        }
                    with open(args.annotations_output, "w") as f:
                        json.dump(coco_json, f, indent=4)
                    
                    with open(os.path.join(args.output_dir, 'tracker.json'), 'w') as f:
                        json.dump(args.tracker, f, indent=4)
        
            cv2.destroyAllWindows()


def visualize_hand_segments(image_path, annotation_path, color=(255, 51, 51) , alpha=0.4, line_thickness=2):
    """
    Draws hand segmentation polygons from a RoboFlow-style annotation onto the image.

    Parameters:
        image_path (str): Path to the image file (e.g., .jpg or .png)
        annotation_path (str): Path to the JSON file with polygon annotations
        color (tuple): RGB color for the polygon (default yellow)
        alpha (float): Transparency of the fill color
        line_thickness (int): Thickness of the polygon border
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotation
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    overlay = image.copy()

    # Draw each polygon
    for box in data.get("boxes", []):
        if box.get("label") == "hands" and box.get("type") == "polygon":
            points = np.array(box["points"], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=line_thickness)
            cv2.fillPoly(overlay, [points], color=color)

    # Blend with transparency
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    image_name = os.path.split(image_path)[-1].split('.')[0]
    cv2.imwrite(os.path.join("video_for_paper_mmtl/hand_annotations", f"{image_name}_maks.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    


def edit_class_names_interactive(annotation_path, image_dir, filter_class=None):
    """
    Interactive UI to edit class names for tools and devices in COCO annotations.
    Displays images with annotations and allows editing class names on the fly.
    
    When a class is changed for a frame, all subsequent frames from the same video
    (identified by the videoname_ prefix in filename) will be updated automatically.
    
    Args:
        annotation_path: Path to annotations.json file
        image_dir: Path to directory containing images
        filter_class: Optional class name or ID to filter - only show images containing this class
    """
    # Load data
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    # Store original counts for verification
    original_image_count = len(coco_data.get('images', []))
    original_annotation_count = len(coco_data.get('annotations', []))
    print(f"📂 Loaded: {original_image_count} images, {original_annotation_count} annotations")
    
    # Safe save function with backup and verification
    def safe_save_annotations(data, path, reason=""):
        """Save annotations with backup and verification to prevent data loss"""
        # Verify data has content
        if not data.get('images') and not data.get('annotations'):
            print(f"⚠️  WARNING: Refusing to save empty data! {reason}")
            return False
        
        if len(data.get('images', [])) == 0 and original_image_count > 0:
            print(f"⚠️  WARNING: All images would be deleted! Aborting save. {reason}")
            return False
        
        # Create backup before saving
        backup_path = path + '.backup'
        if os.path.exists(path):
            import shutil
            shutil.copy2(path, backup_path)
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"💾 Saved: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations")
            print(f"   {reason}" if reason else "")
            return True
        except Exception as e:
            print(f"❌ Error saving: {e}")
            # Restore from backup if save failed
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, path)
                print(f"🔄 Restored from backup")
            return False
    
    # Setup tracker
    tracker_path = os.path.join(os.path.dirname(annotation_path), 'class_editor_tracker.json')
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            tracker = json.load(f)
            last_processed_idx = tracker.get('last_image_index', -1)
    else:
        tracker = {'last_image_index': -1, 'processed_videos': [], 'last_video_index': -1}
        last_processed_idx = -1
    
    # Build mappings
    image_id_to_file = {img['id']: img for img in coco_data['images']}
    
    # Handle both numeric IDs and string names in categories (needed for filtering)
    category_mapping = {}
    category_id_to_info = {}
    if 'categories' in coco_data:
        for cat in coco_data['categories']:
            if isinstance(cat, dict) and 'id' in cat and 'name' in cat:
                category_mapping[cat['id']] = cat['name']
                category_id_to_info[cat['id']] = cat
            elif isinstance(cat, str):
                category_mapping[cat] = cat
    
    # Build annotations by image (needed for filtering)
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # If filter_class is specified, find matching category ID(s)
    filter_category_ids = set()
    if filter_class is not None:
        filter_class_str = str(filter_class).lower()
        for cat_id, cat_name in category_mapping.items():
            # Match by ID or by name (case-insensitive)
            if str(cat_id) == str(filter_class) or cat_name.lower() == filter_class_str:
                filter_category_ids.add(cat_id)
        
        if not filter_category_ids:
            print(f"\n⚠️  Warning: Class '{filter_class}' not found in categories!")
            print("Available classes:")
            for cat_id, cat_name in sorted(category_mapping.items(), key=lambda x: x[0]):
                print(f"  - {cat_name} (ID {cat_id})")
            return
        else:
            matching_names = [category_mapping[cid] for cid in filter_category_ids]
            print(f"\n🔍 Filtering by class: {matching_names} (IDs: {list(filter_category_ids)})")
    
    # Helper function to extract video name from filename (e.g., "002_00005.jpg" -> "002")
    def get_video_name(filename):
        """Extract video name from filename (part before first underscore)"""
        return filename.split('_')[0] if '_' in filename else filename.split('.')[0]
    
    # Group images by video name (with optional class filter)
    video_to_image_ids = defaultdict(list)
    for img_id, img_info in image_id_to_file.items():
        # If filter is active, only include images that have annotations with the filtered class
        if filter_category_ids:
            img_anns = annotations_by_image.get(img_id, [])
            has_filtered_class = any(ann.get('category_id') in filter_category_ids for ann in img_anns)
            if not has_filtered_class:
                continue
        
        video_name = get_video_name(img_info['file_name'])
        video_to_image_ids[video_name].append(img_id)
    
    # Get unique videos in order of appearance (only those with matching images)
    unique_videos = []
    seen_videos = set()
    for img_id in sorted(image_id_to_file.keys()):
        # Skip if image was filtered out
        video_name = get_video_name(image_id_to_file[img_id]['file_name'])
        if video_name not in video_to_image_ids:
            continue
        if img_id not in video_to_image_ids[video_name]:
            continue
        if video_name not in seen_videos:
            unique_videos.append(video_name)
            seen_videos.add(video_name)
    
    if filter_category_ids:
        total_filtered_images = sum(len(imgs) for imgs in video_to_image_ids.values())
        print(f"📋 Found {total_filtered_images} images across {len(unique_videos)} videos with class '{filter_class}'")
    
    # Build class lists from annotations.json categories (grouped by supercategory)
    # This ensures all existing classes (including custom ones) appear in the selection
    annotation_class_lists = {
        'TOOL': {},
        'DEVICE': {},
        'HAND': {}
    }
    
    for cat in coco_data.get('categories', []):
        if isinstance(cat, dict) and 'id' in cat and 'name' in cat:
            supercategory = cat.get('supercategory', 'tool').upper()
            if supercategory not in annotation_class_lists:
                supercategory = 'TOOL'  # Default fallback
            annotation_class_lists[supercategory][cat['id']] = cat['name']
    
    # If no classes found in annotations, fall back to config
    if not annotation_class_lists['TOOL'] and not annotation_class_lists['DEVICE']:
        print("    [!] No classes found in annotations.json, using config defaults")
        for obj_type, class_dict in configs.OBJECT_CLASSES.items():
            if obj_type in annotation_class_lists:
                annotation_class_lists[obj_type] = class_dict
    
    # Color definitions (BGR format)
    colors = {
        'tool': (0, 255, 0),      # Green
        'hand': (0, 0, 255),      # Red
        'device': (255, 0, 0),    # Blue
        'default': (0, 255, 255)  # Yellow
    }
    
    def get_category_type(cat_name):
        """Determine if category is tool, hand, or device"""
        name_lower = cat_name.lower()
        if 'hand' in name_lower:
            return 'hand'
        elif any(device in name_lower for device in ['phone', 'tablet', 'computer', 'screen', 'monitor', 'pc', 'laptop', 'washer', 'dryer', 'refrigerator', 'scaffold', 'vibratory', 'tv', 'fan', 'air intaker']):
            return 'device'
        else:
            return 'tool'
    
    def normalize_string_categories_to_ids():
        """Convert string category_ids to proper numeric IDs"""
        for ann in coco_data['annotations']:
            cat_id = ann.get('category_id')
            if isinstance(cat_id, str):
                # Try to find in existing categories
                cat_name_lower = cat_id.lower()
                found_id = None
                for cid, cname in category_mapping.items():
                    if cname.lower() == cat_name_lower:
                        found_id = cid
                        break
                
                if found_id is None:
                    # Create new category
                    existing_ids = [c['id'] for c in coco_data['categories'] if isinstance(c, dict)]
                    new_id = max(existing_ids, default=-1) + 1
                    cat_type = get_category_type(cat_id)
                    coco_data['categories'].append({
                        'id': new_id,
                        'name': cat_id,
                        'supercategory': cat_type
                    })
                    category_mapping[new_id] = cat_id
                    found_id = new_id
                
                ann['category_id'] = found_id
    
    # Normalize any string categories to IDs first
    normalize_string_categories_to_ids()
    
    # Process each image
    modified = False
    image_ids = list(image_id_to_file.keys())
    
    # Find starting video index based on tracker
    current_video_idx = 0
    processed_videos = set(tracker.get('processed_videos', []))
    last_video_idx = tracker.get('last_video_index', -1)
    
    # Start from the next unprocessed video
    if filter_class is not None:
        if last_video_idx >= 0 and last_video_idx < len(unique_videos) - 1:
            current_video_idx = last_video_idx + 1
        elif last_video_idx >= len(unique_videos) - 1:
            # All videos have been processed
            print("\n✅ All videos have already been processed!")
            print(f"   Processed videos: {len(processed_videos)}/{len(unique_videos)}")
            user_choice = input("\nRestart from beginning? (y/n): ")
            if user_choice.lower() != 'y':
                return
            current_video_idx = 0
            tracker = {'last_image_index': -1, 'processed_videos': [], 'last_video_index': -1}
            processed_videos = set()
    
    print("\n" + "="*60)
    print("Interactive Class Name Editor - Video Sequence Mode")
    print("="*60)
    print("\nControls:")
    print("  [Enter]   - Accept and move to next video")
    print("  [->]       - Next frame (within current video)")
    print("  [<-]       - Previous frame (within current video)")
    print("  [n]       - Rename: Show class selection panel (applies to ALL frames in video)")
    print("  [d]       - Delete: Remove current frame and its annotations from dataset")
    print(f"\nProgress: {len(processed_videos)} videos already processed")
    print(f"Starting from video {current_video_idx + 1}/{len(unique_videos)}")
    print(f"Total videos: {len(unique_videos)}, Total frames: {len(image_ids)}")
    print("="*60 + "\n")
    
    while current_video_idx < len(unique_videos):
        current_video = unique_videos[current_video_idx]
        
        # Skip if this video was already processed
        if current_video in processed_videos:
            print(f"⏭️  Skipping already processed video: {current_video}")
            current_video_idx += 1
            continue
        
        # Get frames from this video for display
        video_image_ids = video_to_image_ids[current_video]
        if not video_image_ids:
            current_video_idx += 1
            continue
        
        # Start with first frame of the video
        current_frame_idx = 0
        
        # Inner loop for navigating frames within a video
        while current_frame_idx < len(video_image_ids):
            image_id = video_image_ids[current_frame_idx]
            img_info = image_id_to_file[image_id]
            file_name = img_info['file_name']
            image_path = os.path.join(image_dir, file_name)
        
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Image not found: {image_path}")
                current_frame_idx += 1
                continue
            
            # Get original dimensions
            orig_h, orig_w = image.shape[:2]
            
            # Resize image to be bigger (1280 width or maintain aspect ratio)
            target_width = 1280
            scale = target_width / orig_w
            new_h = int(orig_h * scale)
            image = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Get annotations for this image
            anns = annotations_by_image.get(image_id, [])
            if not anns:
                current_frame_idx += 1
                continue
            
            # Draw annotations - ALL AT ONCE
            display_image = image.copy()
            ann_info_list = []
            
            # IMPORTANT: Clear any previous data to avoid overlay issues
            bbox_info_list = []
            for idx, ann in enumerate(anns):
                cat_id = ann.get('category_id', 'Unknown')
                
                # Get category name
                if isinstance(cat_id, str):
                    cat_name = cat_id
                else:
                    cat_name = category_mapping.get(cat_id, f'ID_{cat_id}')
                
                cat_type = get_category_type(cat_name)
                color = colors.get(cat_type, colors['default'])
                
                # Store info for editing
                ann_info_list.append({
                    'idx': idx,
                    'ann': ann,
                    'cat_name': cat_name,
                    'cat_type': cat_type,
                    'cat_id': cat_id
                })
                
                # Draw bounding box if available
                if 'bbox' in ann and ann['bbox']:
                    # Scale bbox to resized image
                    x, y, w_box, h_box = ann['bbox']
                    x = int(x * scale)
                    y = int(y * scale)
                    w_box = int(w_box * scale)
                    h_box = int(h_box * scale)
                    
                    # Draw bbox
                    cv2.rectangle(display_image, (x, y), (x + w_box, y + h_box), color, 2)
                    
                    # Prepare label
                    actual_cat_id = ann.get('category_id', idx)
                    label = f"[{actual_cat_id}] {cat_name}"
                    
                    bbox_info_list.append({
                        'x': x, 'y': y, 'w': w_box, 'h': h_box,
                        'label': label, 'color': color
                    })
                
                # Draw segmentation if available - SCALE THE SEGMENTATION
                if 'segmentation' in ann and ann['segmentation']:
                    # Scale segmentation polygons
                    scaled_segs = utils.rescale_polygon(ann['segmentation'], (orig_w, orig_h), (target_width, new_h))
                    utils.safe_draw_polygons(display_image, scaled_segs, color=color, alpha=0.3)
            
            # Calculate info panel height (will be used for label positioning)
            info_font = cv2.FONT_HERSHEY_SIMPLEX
            info_font_scale = 0.7
            info_thickness = 2
            line_height = 30
            panel_height = 2 * line_height + 20  # 2 lines + padding
            
            # Second pass: Draw labels with smart positioning to avoid overlaps
            label_font_scale = 0.6
            label_thickness = 2
            occupied_regions = []  # Track where labels are placed (cleared for each image)
            
            for bbox_info in bbox_info_list:
                x, y, w_box, h_box = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                label = bbox_info['label']
                color = bbox_info['color']
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)[0]
                text_w, text_h = text_size[0], text_size[1]
                
                # Determine best label position
                label_positions = []
                
                # Position 1: Above bbox (default)
                if y - text_h - 15 >= panel_height:  # Not overlapping with top panel
                    label_positions.append({
                        'x': x,
                        'y': y - 5,
                        'bg_y1': y - text_h - 10,
                        'bg_y2': y,
                        'priority': 1
                    })
                
                # Position 2: Inside top of bbox
                if y + text_h + 10 < y + h_box:
                    label_positions.append({
                        'x': x + 5,
                        'y': y + text_h + 5,
                        'bg_y1': y + 2,
                        'bg_y2': y + text_h + 10,
                        'priority': 2
                    })
                
                # Position 3: Inside bottom of bbox (if big bbox going to top)
                if y < panel_height + 10 and y + h_box - text_h - 10 > y:  # Bbox extends into/near panel area
                    label_positions.append({
                        'x': x + 5,
                        'y': y + h_box - 5,
                        'bg_y1': y + h_box - text_h - 10,
                        'bg_y2': y + h_box - 2,
                        'priority': 1  # Higher priority for bboxes near top
                    })
                
                # Position 4: Inside bottom of bbox
                if y + h_box - text_h - 10 > y:
                    label_positions.append({
                        'x': x + 5,
                        'y': y + h_box - 5,
                        'bg_y1': y + h_box - text_h - 10,
                        'bg_y2': y + h_box - 2,
                        'priority': 3
                    })
                
                # Check for overlaps and pick best position
                best_pos = None
                for pos in sorted(label_positions, key=lambda p: p['priority']):
                    label_region = (pos['x'], pos['bg_y1'], pos['x'] + text_w + 5, pos['bg_y2'])
                    
                    # Check if overlaps with occupied regions
                    overlaps = False
                    for occ in occupied_regions:
                        if not (label_region[2] < occ[0] or label_region[0] > occ[2] or
                               label_region[3] < occ[1] or label_region[1] > occ[3]):
                            overlaps = True
                            break
                    
                    if not overlaps and pos['bg_y1'] >= panel_height:  # Not overlapping and below panel
                        best_pos = pos
                        occupied_regions.append(label_region)
                        break
                
                # Use best position or fallback
                if best_pos is None and label_positions:
                    best_pos = label_positions[0]  # Fallback to first option
                    label_region = (best_pos['x'], best_pos['bg_y1'], 
                                  best_pos['x'] + text_w + 5, best_pos['bg_y2'])
                    occupied_regions.append(label_region)
                
                if best_pos:
                    # Draw background rectangle
                    cv2.rectangle(display_image, 
                                (best_pos['x'], best_pos['bg_y1']), 
                                (best_pos['x'] + text_w + 5, best_pos['bg_y2']), 
                                color, -1)
                    
                    # Draw text
                    cv2.putText(display_image, label, (best_pos['x'] + 2, best_pos['y']), 
                              cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 
                              (255, 255, 255), label_thickness, cv2.LINE_AA)
            
            # Add info panel at top
            info_lines = [
                f"Video {current_video_idx + 1}/{len(unique_videos)}: {current_video} - Frame {current_frame_idx + 1}/{len(video_image_ids)}: {file_name}",
                f"Objects: {len(anns)} | [Enter] Next Video  [->/<-] Nav Frame  [n] Change Class  [d] Delete Frame  [q] Quit",
            ]
            
            # Draw semi-transparent background for text with proper sizing
            panel_height = len(info_lines) * line_height + 20
            overlay = display_image.copy()
            cv2.rectangle(overlay, (0, 0), (display_image.shape[1], panel_height), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
            
            y_pos = 25
            for line in info_lines:
                cv2.putText(display_image, line, (10, y_pos), 
                           info_font, info_font_scale, (255, 255, 255), info_thickness, cv2.LINE_AA)
                y_pos += line_height
            
            # Display
            window_name = "Class Editor - Video Sequence Mode"
            cv2.imshow(window_name, display_image)
            cv2.resizeWindow(window_name, target_width, new_h)
            key = cv2.waitKey(0) & 0xFF
            
            # Debug: Print key code
            if key not in (2, 3, 13, 32, ord('n'), ord('d'), ord('q'), 27):
                print(f"    [DEBUG] Key pressed: {key}")
            
            if key == 13:  # Enter - Accept and move to next video
                # Mark as processed even if not edited
                if current_video not in processed_videos:
                    processed_videos.add(current_video)
                    if 'processed_videos' not in tracker:
                        tracker['processed_videos'] = []
                    if current_video not in tracker['processed_videos']:
                        tracker['processed_videos'].append(current_video)
                
                current_video_idx += 1
                # Update tracker
                tracker['last_video_index'] = current_video_idx - 1
                with open(tracker_path, 'w') as f:
                    json.dump(tracker, f, indent=4)
                modified = True
                break  # Exit frame loop, move to next video
                
            elif key == 3:  # Right Arrow (special key code 3 on macOS) - Next frame
                current_frame_idx += 1
                if current_frame_idx >= len(video_image_ids):
                    current_frame_idx = len(video_image_ids) - 1
                
            elif key == 2:  # Left Arrow (special key code 2 on macOS) - Previous frame
                current_frame_idx = max(current_frame_idx - 1, 0)
            
            elif key == ord('d'):  # Delete current frame and its annotations
                # Show confirmation dialog
                confirm_image = display_image.copy()
                confirm_text = "DELETE THIS FRAME? Press [y] to confirm, [n] to cancel"
                text_size = cv2.getTextSize(confirm_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = (confirm_image.shape[1] - text_size[0]) // 2
                text_y = confirm_image.shape[0] // 2
                
                # Draw red background for confirmation
                cv2.rectangle(confirm_image, 
                            (text_x - 20, text_y - text_size[1] - 20),
                            (text_x + text_size[0] + 20, text_y + 20),
                            (0, 0, 180), -1)
                cv2.putText(confirm_image, confirm_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow(window_name, confirm_image)
                confirm_key = cv2.waitKey(0) & 0xFF
                
                if confirm_key == ord('y'):
                    # Delete the image file from disk
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"🗑️  Deleted image file: {image_path}")
                    else:
                        print(f"⚠️  Image file not found (already deleted?): {image_path}")
                    
                    # Remove image from coco_data['images']
                    coco_data['images'] = [img for img in coco_data['images'] if img['id'] != image_id]
                    
                    # Remove all annotations for this image
                    deleted_ann_count = len([ann for ann in coco_data['annotations'] if ann['image_id'] == image_id])
                    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] != image_id]
                    
                    # Update internal data structures
                    if image_id in image_id_to_file:
                        del image_id_to_file[image_id]
                    if image_id in annotations_by_image:
                        del annotations_by_image[image_id]
                    
                    # Remove from video_image_ids list
                    video_image_ids.remove(image_id)
                    
                    print(f"🗑️  Removed {deleted_ann_count} annotations for image ID {image_id}")
                    print(f"📊 Remaining: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
                    
                    modified = True
                    
                    # Check if this video still has frames
                    if len(video_image_ids) == 0:
                        print(f"📁 No more frames in video {current_video}. Moving to next video.")
                        # Mark as processed and move to next video
                        if current_video not in processed_videos:
                            processed_videos.add(current_video)
                            if 'processed_videos' not in tracker:
                                tracker['processed_videos'] = []
                            if current_video not in tracker['processed_videos']:
                                tracker['processed_videos'].append(current_video)
                        current_video_idx += 1
                        tracker['last_video_index'] = current_video_idx - 1
                        with open(tracker_path, 'w') as f:
                            json.dump(tracker, f, indent=4)
                        break  # Exit frame loop
                    
                    # Adjust frame index if needed
                    if current_frame_idx >= len(video_image_ids):
                        current_frame_idx = max(0, len(video_image_ids) - 1)
                    
                    # Continue to next iteration to show updated frame
                    continue
                else:
                    print("   [!] Delete cancelled")
                    continue
                
            elif key == ord('n'):  # Show class selection panel
                editable_objs = [info for info in ann_info_list if info['cat_type'] in ['tool', 'device']]
                
                if not editable_objs:
                    print("   [!] No tool or device objects to edit on this image")
                    continue
            
                # Start with first editable object type
                init_class = editable_objs[0]['cat_type'].upper()
                choose_category = 0
                
                # Create dimmed background to highlight editable objects
                dimmed_image = display_image.copy()
                # Darken the entire image
                dimmed_image = cv2.addWeighted(dimmed_image, 0.3, np.zeros_like(dimmed_image), 0.7, 0)
                
                # Restore brightness only inside the bounding boxes of editable objects
                for obj in editable_objs:
                    ann_obj = obj['ann']
                    if 'bbox' in ann_obj and ann_obj['bbox']:
                        bx, by, bw, bh = ann_obj['bbox']
                        bx, by = int(bx * scale), int(by * scale)
                        bw, bh = int(bw * scale), int(bh * scale)
                        # Ensure bounds are within image
                        bx1, by1 = max(0, bx), max(0, by)
                        bx2, by2 = min(display_image.shape[1], bx + bw), min(display_image.shape[0], by + bh)
                        # Copy the original (bright) region back
                        dimmed_image[by1:by2, bx1:bx2] = display_image[by1:by2, bx1:bx2]
                        # Add a highlight border
                        cv2.rectangle(dimmed_image, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
                
                # Use dimmed_image as base for class selection UI
                display_image_for_class_selection = dimmed_image.copy()
                
                while True:
                    # Build class list from annotations.json categories
                    img_h, img_w = image.shape[:2]
                    mapping = annotation_class_lists[init_class]  # Use annotation-based list
                    color = configs.OBJECT_COLORS[init_class]
                    
                    # Build full list of class options - category IDs are already correct
                    class_lines = []
                    cat_id_to_idx = {}  # Map category_id to list index for selection
                    
                    for idx, (cat_id, cname) in enumerate(sorted(mapping.items())):
                        class_lines.append(f"[{cat_id}] {cname}")
                        cat_id_to_idx[cat_id] = idx
                    
                    # Add custom label option
                    class_lines.append("[c] Custom Label (type new name)")
                    
                    # Calculate optimal layout
                    margin_x = max(10, int(0.02 * img_w))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fscale = max(0.6, min(0.9, 0.75 * (img_w / 1280.0)))  # Slightly bigger font
                    thick = 2
                    line_gap = 1.4  # Slightly more spacing
                    
                    # Measure dimensions
                    line_h = 0
                    max_line_w = 0
                    for ln in class_lines:
                        (tw, th), _ = cv2.getTextSize(ln, font, fscale, thick)
                        max_line_w = max(max_line_w, tw)
                        line_h = max(line_h, int(th * 1.2))
                    
                    # Calculate panel dimensions
                    pad_x, pad_t = 12, 10
                    start_y = panel_height + 10
                    available_h = img_h - start_y - 20
                    
                    title = f"Category (press [n] to change):   {init_class}"
                    (title_w, title_h), _ = cv2.getTextSize(title, font, fscale + 0.1, thick)
                    
                    panel_w = min(img_w - 2*margin_x, max(max_line_w, title_w) + 2*pad_x)
                    total_content_h = pad_t + int((len(class_lines) + 1) * line_h * line_gap) + pad_t
                    panel_h = min(total_content_h, available_h)
                    
                    # Draw the panel
                    image_with_panel = display_image_for_class_selection.copy()
                    utils.draw_translucent_panel(image_with_panel, margin_x, start_y, panel_w, panel_h, 
                                                color=configs.COLORS['classes_background'], alpha=0.4)
                    
                    # Draw title
                    y = start_y + pad_t + line_h
                    cv2.putText(image_with_panel, title, (margin_x + 5, int(y)),
                              font, fscale + 0.1, color, thick, cv2.LINE_AA)
                    y += int(line_h * line_gap * 1.2)  # Extra space after title
                    
                    # Draw class options
                    x_left = margin_x + pad_x
                    for i, ln in enumerate(class_lines):
                        if y + line_h > start_y + panel_h - pad_t:
                            break  # Stop if exceeding bounds
                        cv2.putText(image_with_panel, ln, (x_left, int(y)),
                                  font, fscale, (200, 200, 200), thick - 1, cv2.LINE_AA)
                        y += int(line_h * line_gap)
                    
                    # Show and get user input
                    metrics = {
                        "x_left": margin_x + pad_x,
                        "y_title": start_y + pad_t,
                        "line_h": line_h,
                        "font_scale": fscale,
                        "thickness": thick
                    }
                    
                    # Custom input handling for 'c' option
                    font_input = cv2.FONT_HERSHEY_SIMPLEX
                    typed = ""
                    
                    # Calculate prompt position - below the panel with spacing
                    prompt_y = start_y + panel_h + int(line_h * 1.5)
                    if prompt_y + line_h + 10 > img_h:
                        # If too low, place at top
                        prompt_y = int(line_h * 2)
                    
                    while True:
                        vis = image_with_panel.copy()
                        
                        # Draw semi-transparent background for prompt
                        prompt_text = f"Enter class id or 'c' for custom: {typed or '_'}"
                        prompt_size = cv2.getTextSize(prompt_text, font_input, fscale, thick)[0]
                        prompt_bg_x1 = margin_x - 5
                        prompt_bg_y1 = prompt_y - line_h - 5
                        prompt_bg_x2 = margin_x + prompt_size[0] + 10
                        prompt_bg_y2 = prompt_y + 10
                        
                        overlay = vis.copy()
                        cv2.rectangle(overlay, (prompt_bg_x1, prompt_bg_y1), 
                                    (prompt_bg_x2, prompt_bg_y2), (40, 40, 40), -1)
                        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                        
                        cv2.putText(vis, prompt_text, (margin_x, prompt_y),
                                  font_input, fscale, (0, 255, 255), thick, cv2.LINE_AA)
                        cv2.imshow(window_name, vis)
                        
                        key_input = cv2.waitKey(0) & 0xFF
                        
                        if key_input in (13, 32):  # Enter/Space - submit
                            if typed.lower() == 'c':
                                user_input = -999  # Custom label marker
                            elif typed.isdigit():
                                # User typed category ID directly
                                typed_id = int(typed)
                                # Check if this category ID exists in our mapping
                                if typed_id in mapping:
                                    user_input = typed_id  # Use category ID directly
                                else:
                                    print(f"     [!] Invalid ID {typed_id}. Not found in class list.")
                                    typed = ""
                                    continue
                            else:
                                typed = ""
                                continue
                            break
                        elif key_input in (27, ord('q')):  # Esc/q - cancel
                            user_input = -1
                            break
                        elif key_input in (ord('n'), ord('N')):  # Change category
                            user_input = -10
                            break
                        elif key_input in (8, 127):  # Backspace
                            typed = typed[:-1]
                        elif ord('0') <= key_input <= ord('9') or key_input in (ord('c'), ord('C')):
                            typed += chr(key_input)
                    
                    if user_input == -10:  # User pressed 'n' to change category
                        choose_category += 1
                        available_types = ['TOOL', 'DEVICE']
                        init_class = available_types[choose_category % len(available_types)]
                        continue
                        
                    if user_input == -1:  # User pressed Esc or 'q'
                        print("     [!] No category selected. Exiting category selection.")
                        break
                    
                    # Handle custom label
                    new_class_name = None
                    new_cat_id = None
                    
                    if user_input == -999:  # Custom label
                        custom_label = input("\n    Enter custom label name: ").strip()
                        if custom_label:
                            new_class_name = custom_label
                            cat_type = init_class.lower()
                            
                            # Create actual annotation category with proper ID
                            existing_ids = [cat['id'] for cat in coco_data['categories'] if isinstance(cat, dict)]
                            new_cat_id = max(existing_ids, default=-1) + 1
                            
                            coco_data['categories'].append({
                                'id': new_cat_id,
                                'name': custom_label,
                                'supercategory': cat_type
                            })
                            category_mapping[new_cat_id] = custom_label
                            
                            # Add to annotation class list for immediate use
                            annotation_class_lists[init_class][new_cat_id] = custom_label
                            mapping[new_cat_id] = custom_label
                            
                            print(f"    [*] Added new {init_class} class: '{custom_label}' with ID {new_cat_id}")
                            print(f"    [*] This class will now appear in the list as [{new_cat_id}] {custom_label}")
                            
                    elif user_input >= 0:  # Valid category ID
                        if user_input in mapping:
                            new_class_name = mapping[user_input]
                            new_cat_id = user_input
                        else:
                            print("     [!] Invalid selection.")
                            continue
                    else:
                        print("     [!] Invalid selection.")
                        continue
                    
                    if new_class_name and new_cat_id is not None:
                        cat_type = init_class.lower()
                        
                        # new_cat_id is already set from above (either selected or newly created)
                        # Update ALL objects of the same type IN THE CURRENT FRAME
                        updated_count = 0
                        for info in ann_info_list:
                            if info['cat_type'] == cat_type:
                                info['ann']['category_id'] = new_cat_id
                                info['cat_name'] = new_class_name
                                updated_count += 1
                        
                        # Now update ALL frames from the same video
                        print(f"\n{'='*60}")
                        print(f"Updating all frames from video: {current_video}")
                        print(f"{'='*60}")
                        
                        video_update_log = []
                        total_frames_updated = 0
                        total_objects_updated = 0
                        
                        for vid_img_id in video_image_ids:
                            vid_anns = annotations_by_image.get(vid_img_id, [])
                            frame_objects_updated = 0
                            frame_changes = []
                            
                            for vid_ann in vid_anns:
                                vid_cat_id = vid_ann.get('category_id')
                                
                                # Get category name
                                if isinstance(vid_cat_id, str):
                                    vid_cat_name = vid_cat_id
                                else:
                                    vid_cat_name = category_mapping.get(vid_cat_id, f'ID_{vid_cat_id}')
                                
                                vid_cat_type = get_category_type(vid_cat_name)
                                
                                # Update if it matches the category type we're changing
                                if vid_cat_type == cat_type:
                                    old_cat_id = vid_ann['category_id']
                                    old_cat_name = category_mapping.get(old_cat_id, f'ID_{old_cat_id}') if not isinstance(old_cat_id, str) else old_cat_id
                                    
                                    vid_ann['category_id'] = new_cat_id
                                    frame_objects_updated += 1
                                    total_objects_updated += 1
                                    
                                    frame_changes.append({
                                        'old_id': old_cat_id,
                                        'old_name': old_cat_name,
                                        'new_id': new_cat_id,
                                        'new_name': new_class_name
                                    })
                            
                            if frame_objects_updated > 0:
                                total_frames_updated += 1
                                vid_file_name = image_id_to_file[vid_img_id]['file_name']
                                video_update_log.append({
                                    'frame': vid_file_name,
                                    'objects_updated': frame_objects_updated,
                                    'changes': frame_changes
                                })
                        
                        # Print detailed update log
                        print(f"\n✅ Updated {total_objects_updated} objects across {total_frames_updated} frames")
                        print(f"\nDetailed changes:")
                        for log_entry in video_update_log:
                            changes_str = ", ".join([f"[{c['old_id']}]{c['old_name']} -> [{c['new_id']}]{c['new_name']}" 
                                                    for c in log_entry['changes']])
                            print(f"  📄 {log_entry['frame']}: {log_entry['objects_updated']} object(s) - {changes_str}")
                        print(f"{'='*60}\n")
                        
                        # Redraw display with updated labels using smart positioning
                        display_image = image.copy()
                        
                        # IMPORTANT: Clear previous data to avoid overlay
                        redraw_bbox_info = []
                        for idx, ann in enumerate(anns):
                            info = ann_info_list[idx]
                            cat_type_local = info['cat_type']
                            color_local = colors.get(cat_type_local, colors['default'])
                            cat_name_display = info['cat_name']
                            
                            if 'bbox' in ann and ann['bbox']:
                                x, y, w_box, h_box = ann['bbox']
                                x, y = int(x * scale), int(y * scale)
                                w_box, h_box = int(w_box * scale), int(h_box * scale)
                                
                                cv2.rectangle(display_image, (x, y), (x + w_box, y + h_box), color_local, 2)
                                
                                display_cat_id = ann.get('category_id', 0)
                                label = f"[{display_cat_id}] {cat_name_display}"
                                
                                redraw_bbox_info.append({
                                    'x': x, 'y': y, 'w': w_box, 'h': h_box,
                                    'label': label, 'color': color_local
                                })
                            
                            if 'segmentation' in ann and ann['segmentation']:
                                scaled_segs = utils.rescale_polygon(ann['segmentation'], (orig_w, orig_h), (target_width, new_h))
                                utils.safe_draw_polygons(display_image, scaled_segs, color=color_local, alpha=0.3)
                        
                        # Draw labels with smart positioning
                        occupied_regions_redraw = []
                        for bbox_info in redraw_bbox_info:
                            x, y, w_box, h_box = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                            label = bbox_info['label']
                            color = bbox_info['color']
                            
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            text_w, text_h = text_size[0], text_size[1]
                            
                            label_positions = []
                            if y - text_h - 15 >= panel_height:
                                label_positions.append({'x': x, 'y': y - 5, 'bg_y1': y - text_h - 10, 'bg_y2': y, 'priority': 1})
                            if y + text_h + 10 < y + h_box:
                                label_positions.append({'x': x + 5, 'y': y + text_h + 5, 'bg_y1': y + 2, 'bg_y2': y + text_h + 10, 'priority': 2})
                            if y < panel_height + 10 and y + h_box - text_h - 10 > y:
                                label_positions.append({'x': x + 5, 'y': y + h_box - 5, 'bg_y1': y + h_box - text_h - 10, 'bg_y2': y + h_box - 2, 'priority': 1})
                            if y + h_box - text_h - 10 > y:
                                label_positions.append({'x': x + 5, 'y': y + h_box - 5, 'bg_y1': y + h_box - text_h - 10, 'bg_y2': y + h_box - 2, 'priority': 3})
                            
                            best_pos = None
                            for pos in sorted(label_positions, key=lambda p: p['priority']):
                                label_region = (pos['x'], pos['bg_y1'], pos['x'] + text_w + 5, pos['bg_y2'])
                                overlaps = False
                                for occ in occupied_regions_redraw:
                                    if not (label_region[2] < occ[0] or label_region[0] > occ[2] or
                                           label_region[3] < occ[1] or label_region[1] > occ[3]):
                                        overlaps = True
                                        break
                                if not overlaps and pos['bg_y1'] >= panel_height:
                                    best_pos = pos
                                    occupied_regions_redraw.append(label_region)
                                    break
                            
                            if best_pos is None and label_positions:
                                best_pos = label_positions[0]
                                occupied_regions_redraw.append((best_pos['x'], best_pos['bg_y1'], best_pos['x'] + text_w + 5, best_pos['bg_y2']))
                            
                            if best_pos:
                                cv2.rectangle(display_image, (best_pos['x'], best_pos['bg_y1']), 
                                            (best_pos['x'] + text_w + 5, best_pos['bg_y2']), color, -1)
                                cv2.putText(display_image, label, (best_pos['x'] + 2, best_pos['y']), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Redraw info panel with same styling
                        overlay = display_image.copy()
                        cv2.rectangle(overlay, (0, 0), (display_image.shape[1], panel_height), (40, 40, 40), -1)
                        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
                        y_pos = 25
                        for line in info_lines:
                            cv2.putText(display_image, line, (10, y_pos), 
                                       info_font, info_font_scale, (255, 255, 255), info_thickness, cv2.LINE_AA)
                            y_pos += line_height
                        
                        modified = True
                        print(f"    [*] Local frame: Updated {updated_count} {cat_type}(s) to {new_class_name}")
                        
                        # Mark as processed (but stay on current video)
                        processed_videos.add(current_video)
                        
                        # Update tracker
                        if 'processed_videos' not in tracker:
                            tracker['processed_videos'] = []
                        if current_video not in tracker['processed_videos']:
                            tracker['processed_videos'].append(current_video)
                        tracker['last_video_index'] = current_video_idx
                        with open(tracker_path, 'w') as f:
                            json.dump(tracker, f, indent=4)
                        
                        # Exit class selection loop only (stay on current video/frame)
                        break
                
                # After breaking from class selection, just continue showing current frame
                # (Don't exit frame loop - allow user to navigate or edit more)
            
            elif key in (27, ord('q')):  # ESC or q - Quit
                cv2.destroyAllWindows()
                # Save any pending changes before exiting
                if modified:
                    safe_save_annotations(coco_data, annotation_path, "Saved on quit")
                    # Save tracker state
                    tracker['last_video_index'] = current_video_idx - 1
                    if 'processed_videos' not in tracker:
                        tracker['processed_videos'] = []
                    with open(tracker_path, 'w') as f:
                        json.dump(tracker, f, indent=4)
                    print(f"✅ Tracker saved: Processed up to video {tracker['last_video_index'] + 1}/{len(unique_videos)}")
                return  # Exit function completely
    
    cv2.destroyAllWindows()
    
    # Auto-save changes and tracker
    if modified:
        safe_save_annotations(coco_data, annotation_path, "Saved at end of processing")
        
        # Save final tracker state
        tracker['last_video_index'] = current_video_idx - 1
        if 'processed_videos' not in tracker:
            tracker['processed_videos'] = []
        with open(tracker_path, 'w') as f:
            json.dump(tracker, f, indent=4)
        print(f"✅ Tracker saved: Processed up to video {tracker['last_video_index'] + 1}/{len(unique_videos)}")
        print(f"✅ Total processed videos: {len(tracker.get('processed_videos', []))}")
    
    print("\n✨ Class editing complete!")


def introduction():
    print("This script contains multiple functionalities:")
    print("1. Convert videos in a directory to frames and save them as images.")
    print("2. Rename the images files (also update the annotations.json).")
    print("3. Extract bbs as coco formart from The Maintenance Action Dataset")
    print("4. Visualze random samples from a coco dataset")
    print("5. Summarize a COCO annotation file")
    print("6. Edit class names for tools and devices interactively")
    print('')
    input_choice = input("Choose [1/2/3/4/5/6]: ")
    if input_choice == '1':
        video_dir = utils.input_with_path_completion("Enter the path to the video directory: ")
        functionality_one_navigator(video_dir)
    elif input_choice == '2':
        video_dir = utils.input_with_path_completion("Enter the path to dir containing images and annotations.json: ")
        rename_images_to_numeric_order(video_dir)
    elif input_choice == '3':
        extractor()
    elif input_choice == '4':
        root_file = utils.input_with_path_completion("Enter the path to the root file [should contain images, annotations.json]: ")
        image_dir = os.path.join(root_file, "images")
        annotation_path = os.path.join(root_file, "annotations.json")
        tool = None if input("Enter the tool name to visualize (or leave empty for random sample): ") == "" else input("Enter the tool name to visualize (or leave empty for random sample): ") 
        visualize_random_sample_from_coco(annotation_path, image_dir, tool=tool)
    elif input_choice == '5':
        json_path = utils.input_with_path_completion("Enter the path to the COCO annotation file (.json): ")
        summarize_coco_annotation(json_path)
    elif input_choice == '6':
        root_file = utils.input_with_path_completion("Enter the path to the root folder [should contain images/ and annotations.json]: ")
        image_dir = os.path.join(root_file, "images")
        annotation_path = os.path.join(root_file, "annotations1.json")
        filter_class_input = input("Enter class name or ID to filter (leave empty to show all): ").strip()
        filter_class = filter_class_input if filter_class_input else None
        edit_class_names_interactive(annotation_path, image_dir, filter_class=filter_class)
    else:
        print("Invalid choice. Please try again.")



if __name__ == "__main__":
    introduction()
    