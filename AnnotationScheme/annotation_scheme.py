"""
This scheme is responsible for the annotation scheme mentioned in our Maintenance Datasets paper
it works as follows:
 - get video input
 - convert the input to frames
     - First, use YOLOv8 to predict the bb, if detected, goto SAM2
     - IF not, the first frame will pop up to choose the initial bb manually
 - SAM2 is then applied to continue to detect the bb based on the initial bb we choose
 - repeat every n frames

 
 UPDATES:
  13/07/2025 -- added annotations for hand segmentations by asking user in real-time to choose points for hands or tool (or both)
  23/07/2025 -- if we save the coco images as vidname_numeric.jpg, then we can track for every video the done *frames* rather than
                the video name...
  13/08/2025 -- The scheme now support multi object segmentations (not only hands and tool) to change, goto config.py and change the object to annotate

# TODO:
  1. Add in the intro menu the option to annotate the unannotated frames of each video - default start with the first unannotated VIDEO
  2. add automate hand segmentations -- intergrate egoHOS

SEARCH WORDS IN THIS FILE FOR EDITTING/FIXING:
1. MOVE
2. EDIT
3. REMOVE
"""
from PIL import Image
from pathlib import Path
import shutil
import re
from datetime import datetime
import os
import sys
sys.path.append(os.getcwd())
from AnnotationScheme.utils.sam2_loader import SAM2_ROOT  # noqa: F401
from segmentanything.sam2.build_sam import build_sam2
from segmentanything.sam2.build_sam import (build_sam2_video_predictor)
from segmentanything.sam2.sam2_image_predictor import SAM2ImagePredictor
from AnnotationScheme.utils import utils
from AnnotationScheme.utils.id_manager import IdManager
from AnnotationScheme import configs
import json
import torch
from types import SimpleNamespace
import cv2
import numpy as np
import argparse
from collections import defaultdict
from bitarray import bitarray
import atexit
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# Initialize IdManager
id_manager = IdManager()

# Initialize HandDetector (lazy load to avoid unnecessary startup delay)
hand_detector = None

def get_hand_detector():
    global hand_detector
    if hand_detector is None:
        from AnnotationScheme.utils.hand_detector import HandDetector
        print(" [!] Loading HandDetector (YOLO11-pose)...")
        hand_detector = HandDetector()
    return hand_detector

# ---------------------------------------------------------- Admin
parser = argparse.ArgumentParser(
        description="Semi-automatic video annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--weights", type=str,
                        default='l', help="choose sam2 weights [tiny (t), small (s), base_plus (b), large (l)] default- t")
parser.add_argument("--new_shape", nargs=2, type=int, metavar=("W", "H"),
                        default=(640, 360), help="Resize frames before export")

parser.add_argument("--repeat", type=int,
                        default=100, help="Resize frames before export")
parser.add_argument("--pass_annotated", action='store_true',
                        help="If set, the videos that are annotated (even partially) will be skipped")

args_parser, _ = parser.parse_known_args()

# INITIALIZE MODELS AND GLOBAL VARIABLES
sam_video_predictor = build_sam2_video_predictor(configs.config_weights_mapping[args_parser.weights]['configs'],
                                                 configs.config_weights_mapping[args_parser.weights]['weights'], device=device)
sam_predictor = SAM2ImagePredictor(build_sam2(configs.config_weights_mapping[args_parser.weights]['configs'],
                                              configs.config_weights_mapping[args_parser.weights]['weights'], device=device))
yolo_failed = False
# right click for points to segment, left point for excluded segmentation
segment_by_points = True
active_prompt = configs.OBJECT_WITH_BB[0]
object_segmentations = {}
object_annotation_info = {
        'include_points': [],
        'exclude_points': [],
        'history': [],
        'start_point': None,
        'end_point': None
    }
for obj in configs.OBJECT_TO_ANNOTATE.keys():
    object_segmentations[obj] = []
    
# these variables for manually annotator GUI
next_video = False
is_drawing = False
image_copy = None   # To reset the image during dynamic drawing
clean_state = None  # the clean image state for the drawing
previous_state = None  # to reset the image during dynamic drawing
winname = f'{-1}'   # before starting...
tracker_path = 'tracker.json' # track the database paths (add for each path if manually or yolo)
cursor_pos = None            # (x, y) or None
next_category = False


INSTANCE_BLOCK = 1000
INSTANCE_NEXT  = {cls: configs.OBJECT_TO_ANNOTATE[cls] * INSTANCE_BLOCK
                  for cls in configs.OBJECT_TO_ANNOTATE}
# Registry: SAM2 obj_id (instance) -> {"class": <semantic class>, "class_name": None or str}
INST2CLASS     = {}
#########################################################
#                     Helpers                           #
#########################################################

# Caching for check_key performance
_KEY_CACHE = {}

def get_keycodes(key_config):
    """Map shortcut name or list of names to keycodes."""
    config_tuple = tuple(key_config) if isinstance(key_config, list) else (key_config,)
    if config_tuple in _KEY_CACHE:
        return _KEY_CACHE[config_tuple]

    if isinstance(key_config, str):
        key_config = [key_config]

    codes = []
    for k in key_config:
        k = k.lower()
        if k == "enter": codes.append(13)
        elif k == "space": codes.append(32)
        elif k == "esc": codes.append(27)
        elif k == "left": codes.extend([81, 2424832, 65361])
        elif k == "right": codes.extend([83, 2555904, 65363])
        elif len(k) == 1: codes.append(ord(k))

    _KEY_CACHE[config_tuple] = codes
    return codes

def _get_display_key(action_name):
    """Return a pretty string for the first shortcut of an action."""
    shortcut = configs.SHORTCUTS.get(action_name, "")
    if isinstance(shortcut, list):
        shortcut = shortcut[0]
    shortcut = str(shortcut).lower()
    if shortcut == "enter": return "Enter"
    if shortcut == "space": return "Space"
    if shortcut == "esc":   return "Esc"
    if shortcut == "f1":    return "F1"
    if shortcut == "left":  return "Left"
    if shortcut == "right": return "Right"
    return shortcut.upper()

def check_key(pressed_code, action_name):
    """Check if the pressed code matches the shortcut for action_name."""
    if pressed_code is None:
        return False
    target_codes = get_keycodes(configs.SHORTCUTS.get(action_name, ""))
    return (pressed_code & 0xFF) in target_codes or pressed_code in target_codes
def _draw_saved_annotations(canvas):
    """Draw every committed annotation from object_segmentations onto canvas."""
    for obj, anns in object_segmentations.items():
        col = configs.OBJECT_COLORS[obj]
        for ann in anns:
            sp, ep = ann.get('start_point'), ann.get('end_point')
            if sp and ep:
                cv2.rectangle(canvas, tuple(map(int, sp)), tuple(map(int, ep)), col, 2)
            for pt in ann.get('include_points', []):
                cv2.circle(canvas, tuple(map(int, pt)), 5, col, -1)
            for pt in ann.get('exclude_points', []):
                cv2.circle(canvas, tuple(map(int, pt)), 5, configs.COLORS['exclude'], -1)


def mouse_callback(event, x, y, flags, param):
    global cursor_pos, is_drawing, segment_by_points, object_annotation_info
    cursor_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if segment_by_points:
            object_annotation_info['include_points'].append([x, y])
            object_annotation_info['history'].append(('inc', [x, y]))
        else:
            object_annotation_info['start_point'] = [x, y]
            object_annotation_info['end_point']   = [x, y]
            is_drawing = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        if segment_by_points:
            object_annotation_info['exclude_points'].append([x, y])
            object_annotation_info['history'].append(('exc', [x, y]))
        else:
            object_annotation_info['start_point'] = [x, y]
            object_annotation_info['end_point']   = [x, y]
            is_drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing and not segment_by_points:
        object_annotation_info['end_point'] = [x, y]

    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        if not segment_by_points and is_drawing:
            object_annotation_info['end_point'] = [x, y]
        is_drawing = False


def add_manually_bb(args, frame_image, frame_idx=0, total_frames=0, frame_tracker_text=''):
    global winname, segment_by_points, active_prompt, cursor_pos
    global object_annotation_info, object_segmentations

    object_idx = 0
    object_colors = configs.OBJECT_COLORS.copy()
    active_prompt = configs.OBJECT_WITH_BB[object_idx]  # default object
    base = frame_image.copy()
    cursor_pos = None
    is_running = True
    undo_stack = []

    def _finalize_current_prompt():
        global object_annotation_info
        if utils.annotation_frame_info(object_annotation_info):
            payload = dict(object_annotation_info)
            object_segmentations[active_prompt].append(payload)
            
            # TRACK FINALIZATION
            undo_stack.append({
                'type': 'finalized', 
                'prompt': active_prompt, 
                'mode': segment_by_points
            })

            # reset the working buffer
            object_annotation_info = {
                'include_points': [], 'exclude_points': [],
                'history': [], # D-03: Fixed history reset bug
                'start_point': None, 'end_point': None
            }

    def _revert_last_action():
        nonlocal undo_stack, object_idx
        global active_prompt, segment_by_points, object_annotation_info
        
        if not undo_stack:
            print(" [Undo] Nothing to undo")
            return
            
        last_action = undo_stack.pop()
        if last_action['type'] == 'finalized':
            active_prompt = last_action['prompt']
            segment_by_points = last_action['mode']
            object_annotation_info = object_segmentations[active_prompt].pop()
            # Update object_idx to match active_prompt
            object_idx = list(configs.OBJECT_TO_ANNOTATE.keys()).index(active_prompt)
            print(f" [Undo] Restored finalized annotation for {active_prompt}")
        
        elif last_action['type'] == 'switch_object':
            active_prompt = last_action['old_prompt']
            object_idx = last_action['old_idx']
            print(f" [Undo] Reverted object switch to {active_prompt}")
            
        elif last_action['type'] == 'toggle_mode':
            segment_by_points = last_action['old_mode']
            print(f" [Undo] Reverted mode switch to {'Points' if segment_by_points else 'BB'}")
        
        elif last_action['type'] == 'hand_automate':
            count = last_action['count']
            for _ in range(count):
                if object_segmentations['HAND']:
                    object_segmentations['HAND'].pop()
            print(f" [Undo] Removed {count} automated hand prompts")

    utils.place_window(winname)
    cv2.imshow(winname, base)
    cv2.setMouseCallback(winname, mouse_callback, object_colors)

    while is_running:
        frame_vis = base.copy()
        _draw_saved_annotations(frame_vis)  # all previously confirmed items

        # 1) Top banner with dynamic help
        u_key = _get_display_key("undo")
        m_key = _get_display_key("switch_object")
        d_key = _get_display_key("toggle_mode")
        acc_key = _get_display_key("accept")
        edit_items = [
            f"[{u_key}] Undo", f"[{m_key}] Next Obj", 
            f"[{d_key}] Toggle Mode", f"[{acc_key}] Done"
        ]
        utils.draw_menu_banner(frame_vis, items=edit_items, frame_idx=frame_idx, total_frames=total_frames, app_mode=args.mode)

        # 2) overlay current in-progress (not yet finalized)
        col = object_colors[active_prompt]
        sp, ep = object_annotation_info['start_point'], object_annotation_info['end_point']
        if sp and ep and not segment_by_points:
            cv2.rectangle(frame_vis, tuple(map(int, sp)), tuple(map(int, ep)), col, 2)
        for pt in object_annotation_info['include_points']:
            cv2.circle(frame_vis, tuple(map(int, pt)), 5, col, -1)
        for pt in object_annotation_info['exclude_points']:
            cv2.circle(frame_vis, tuple(map(int, pt)), 5, configs.COLORS['exclude'], -1)

        # 3) crosshair
        if cursor_pos is not None:
            x, y = cursor_pos
            cv2.line(frame_vis, (x, 0), (x, frame_vis.shape[0]), configs.COLORS['cross_line'], 1)
            cv2.line(frame_vis, (0, y), (frame_vis.shape[1], y), configs.COLORS['cross_line'], 1)

        cv2.imshow(winname, frame_vis)
        key = cv2.waitKeyEx(10)
        if key == -1:
            continue

        pure_key = key & 0xFF
        
        # D-09: Undo
        if check_key(key, "undo"):
            if segment_by_points:
                if object_annotation_info.get('history'):
                    h_type, _ = object_annotation_info['history'].pop()
                    if h_type == 'inc' and object_annotation_info['include_points']:
                        object_annotation_info['include_points'].pop()
                    elif h_type == 'exc' and object_annotation_info['exclude_points']:
                        object_annotation_info['exclude_points'].pop()
                    print(f" [Undo] Removed last {h_type} point")
                else:
                    _revert_last_action()
            else:
                if object_annotation_info['start_point'] or object_annotation_info['end_point']:
                    object_annotation_info['start_point'] = None
                    object_annotation_info['end_point'] = None
                    print(" [Undo] Reset bounding box")
                else:
                    _revert_last_action()
            continue

        # D-10: Frame Navigation
        if check_key(key, "left"):
             _finalize_current_prompt()
             return 65361
        if check_key(key, "right"):
             _finalize_current_prompt()
             return 65363
        
        if check_key(key, "segment_back") or check_key(key, "segment_forward") or check_key(key, "jump_next"):
             _finalize_current_prompt()
             return (key & 0xFF) if (key & 0xFF) != 0 else key

        if check_key(key, "previous_frame"):
             _finalize_current_prompt()
             return ord('p')

        # D-13: Hand Automation
        if check_key(key, "hand_automate"):
            active_prompt = "HAND"
            detector = get_hand_detector()
            prompts = detector.detect_hands(base)
            if prompts:
                for pts, lbls in prompts:
                    payload = {
                        'include_points': pts.tolist(),
                        'exclude_points': [],
                        'history': [('inc', p) for p in pts.tolist()],
                        'start_point': None, 'end_point': None
                    }
                    object_segmentations['HAND'].append(payload)
                undo_stack.append({'type': 'hand_automate', 'count': len(prompts)})
                print(f" [!] Automated {len(prompts)} hand prompts.")
            else:
                print(" [!] No hands detected.")
            continue

        if check_key(key, "help"):
            h_key = _get_display_key("help")
            hand_key = _get_display_key("hand_automate")
            m_key = _get_display_key("switch_object")
            d_key = _get_display_key("toggle_mode")
            u_key = _get_display_key("undo")
            acc_key = _get_display_key("accept")
            p_key = _get_display_key("previous_frame")
            q_key = _get_display_key("quit")
            
            help_text = ['Edit Mode Help:',
                         ' [L-Click] Add Point / Box Start',
                         ' [R-Click] Exclude Point / Box End',
                         f' [{u_key}] Undo',
                         f' [{m_key}] Next Object',
                         f' [{d_key}] Toggle Points/BB',
                         f' [{hand_key}] Automate Hands',
                         f' [{p_key}] Previous Frame',
                         ' [Arrows] Navigate 1 frame',
                         f' [{acc_key}] Done / Save',
                         f' [{q_key}] Cancel / Exit Help'
            ]
            utils.draw_menu_panel(frame_vis, help_text, start_xy=(10, 100),
                                  bg_color=configs.COLORS['panel_color'],
                                  text_color=col,
                                  line_gap=configs.LINE_GAP)
            cv2.imshow(winname, frame_vis)
            cv2.waitKey(0)
            continue

        if check_key(key, "quit"):
             _finalize_current_prompt()
             return ord('q')

        if check_key(key, "accept"):  # Enter / Space → finish editing this frame
            _finalize_current_prompt()
            return key

        elif check_key(key, "switch_object"):  # switch object
            old_prompt = active_prompt
            old_idx = object_idx
            _finalize_current_prompt()
            object_idx = (object_idx + 1) % len(configs.OBJECT_TO_ANNOTATE)
            active_prompt = list(configs.OBJECT_TO_ANNOTATE.keys())[object_idx]
            undo_stack.append({'type': 'switch_object', 'old_prompt': old_prompt, 'old_idx': old_idx})

        elif check_key(key, "toggle_mode"):  # toggle drawing mode
            old_mode = segment_by_points
            _finalize_current_prompt()
            segment_by_points = not segment_by_points
            undo_stack.append({'type': 'toggle_mode', 'old_mode': old_mode})

        elif check_key(key, "quit") or check_key(key, "cancel"):  # cancel/quit editing
            reset_globals()
            return key


def reset_globals():
    global next_video, object_segmentations, segment_by_points, next_category, object_annotation_info
    next_video = False
    segment_by_points = True
    next_category = False
    object_annotation_info = {
        'include_points': [],
        'exclude_points': [],
        'history': [],
        'start_point': None,
        'end_point': None
    }

    for obj_id in object_segmentations.keys():
        object_segmentations[obj_id] = []
       



def run_preview(args, preview_path, annotation_results, video_coco_dir, new_shape=(640, 360), milli=0, save_vis=False):
    """
    Run the result of the bounding boxes with a choice to edit
    :param preview_path: path to MP4 bb result
    :param annotation_results: dictionary containing bounding boxes and segmentations for each object in configs
    :param video_coco_dir: Path to save per-video results
    :return:
    """
    global segment_by_points, winname
    if save_vis:
        vis_dir = os.path.join(video_coco_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    def extend_annotations(v_stack):
        for frame_info in v_stack:
            if frame_info['to_save_image'] is not None:
                args.annotations['images'].append(frame_info['img_info'])
                args.annotations['annotations'].extend(frame_info['annotations'])
                # D-05: Save to per-video images folder
                img_dest = os.path.join(video_coco_dir, 'images', frame_info['img_info']['file_name'])
                cv2.imwrite(img_dest, frame_info['to_save_image'])

    def _commit_frame(idx, clean_img=None, vis_img=None):
        nonlocal img_id, ann_id
        
        # Check if frame already in stack
        f_name = f"{clean_vid_name}_{idx:05d}.jpg"
        for i, existing in enumerate(video_stack):
            if existing['frame_name'] == f_name:
                video_stack.pop(i)
                break

        if clean_img is None:
            f_path = os.path.join(preview_path, 'frames', f"{idx:05d}.jpg")
            clean_img = cv2.imread(f_path)
        
        if clean_img is None: return False
        
        f_h, f_w = clean_img.shape[:2]
        f_name = f"{clean_vid_name}_{idx:05d}.jpg"
        
        f_info = {
            'frame_name': f_name,
            'annotations': [],
            'to_save_image': None,
            'img_info': None
        }

        im_saved = False
        
        # We might need to generate vis_img if it's not provided and save_vis is true
        if save_vis and vis_img is None:
            vis_img = clean_img.copy()
            # Draw existing annotations for visualization
            for obj_name in annotation_results.keys():
                if f'{idx}' in annotation_results[obj_name]['bb']:
                    class_names = annotation_results[obj_name]['bb'].get('class_names', [])
                    bbs = annotation_results[obj_name]['bb'][f'{idx}']
                    if f'{idx}' in annotation_results[obj_name]['seg'] and annotation_results[obj_name]['seg'][f'{idx}']:
                        utils.safe_draw_polygons(vis_img, annotation_results[obj_name]['seg'][f'{idx}'], color=configs.OBJECT_COLORS[obj_name], alpha=0.4)
                    if bbs and bbs != []:
                        for bb_idx, bb in enumerate(bbs):
                            class_name = class_names[bb_idx] if bb_idx < len(class_names) and class_names[bb_idx] else 'unknown'
                            vis_img = utils.draw_cocoBB_from_dict(vis_img.copy(), [bb], class_name,
                                                                        color=configs.OBJECT_COLORS[obj_name],
                                                                        orig_width=f_w, orig_height=f_h,
                                                                        target_size=(f_w, f_h))

        for obj_name in annotation_results.keys():
            # classes with bounding boxes
            if obj_name in configs.OBJECT_WITH_BB:
                bbs = annotation_results[obj_name]['bb'].get(f'{idx}', [])
                segs = annotation_results[obj_name]['seg'].get(f'{idx}', [])
                if not bbs or not segs:
                    continue

                class_names = annotation_results[obj_name]['bb'].get('class_names', [])
                if not class_names or len(class_names) < len(bbs):
                    continue

                for bb_idx, (bb, seg) in enumerate(zip(bbs, segs)):
                    class_name = class_names[bb_idx] if bb_idx < len(class_names) else None
                    if class_name is None:
                        continue
                        
                    cat_id = args.category_mapping_name_to_id[class_name.lower()]
                    _bb = utils.rescale_bbox_x1y1wh(bb, (f_w, f_h), new_shape)
                    obj_ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": _bb,
                        "area": _bb[-1] * _bb[-2],
                        "iscrowd": 0,
                        "segmentation": utils.rescale_polygon(seg, (f_w, f_h), new_shape),
                    }
                    f_info['annotations'].append(obj_ann)
                    ann_id += 1
                    args._progress_state[args.vid_name][idx] = True
                    im_saved = True

            # classes without bounding boxes
            else:
                segs = annotation_results[obj_name]['seg'].get(f'{idx}', [])
                if not segs:
                    continue
                for seg in segs:
                    hand_ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 10,  # TODO: map properly if needed
                        "bbox": [],
                        "area": 0,
                        "iscrowd": 0,
                        "segmentation": utils.rescale_polygon(seg, (f_w, f_h), new_shape),
                    }
                    f_info['annotations'].append(hand_ann)
                    ann_id += 1
                    args._progress_state[args.vid_name][idx] = True
                    im_saved = True

        if im_saved:
            f_info['to_save_image'] = cv2.resize(clean_img, dsize=new_shape, interpolation=cv2.INTER_LINEAR)
            f_info['img_info'] = {
                "id": img_id,
                "file_name": f_name,
                "width": new_shape[0],
                "height": new_shape[1]
            }
            video_stack.append(f_info)
            img_id += 1
            if save_vis and vis_img is not None:
                cv2.imwrite(os.path.join(video_coco_dir, 'vis', f_name), vis_img)
            return True
        return False

    img_id = args.annotations['images'][-1]['id'] + 1 if args.annotations['images'] else 0
    ann_id = args.annotations['annotations'][-1]['id'] + 1 if args.annotations['annotations'] else 0
    frame_idx = 0
    video_dir_name = os.path.split(preview_path)[-1]
    # D-07: Strip v{index}_ prefix for clean COCO export
    clean_vid_name = re.sub(r'^v\d+_', '', video_dir_name)

    total_frames = len(os.listdir(os.path.join(preview_path, 'frames')))
    winname = f'Preview - {video_dir_name}'

    video_stack = []

    # D-05: Auto-accept all propagated frames if enabled
    if args.auto_accept:
        print(f" [*] Auto-committing {len(success_indices)} propagated segments...")
        for segment in success_indices:
            for f in segment:
                _commit_frame(f)
    else:
        acc_key = _get_display_key("accept")
        print(f" [*] Manual Accept mode active. Waiting for [{acc_key}] to save frames.")

    while True:
        if frame_idx >= total_frames:
            # D-05: Final discard prompt with cancel
            question_img = orig_frame_img.copy() if 'orig_frame_img' in locals() else np.zeros((new_shape[1], new_shape[0], 3), dtype=np.uint8)
            h, w, _ = question_img.shape
            cancel_key = _get_display_key("cancel")
            test = f"Save & Finish? (y) / Discard? (d) / [{cancel_key}] Cancel"
            (tw, th), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_DUPLEX, 1.9, 3)
            cv2.putText(question_img, test, (w//2 - tw//2, h//6), cv2.FONT_HERSHEY_DUPLEX,
                        1.9, (0, 0, 255), 3, cv2.LINE_AA)

            utils.place_window(winname, winnsize=configs.winnsize)
            cv2.imshow(winname, question_img)

            while True:
                key = cv2.waitKey(0)
                if check_key(key, "cancel"):
                    print(f" [cancel] Returning to review (last frame)")
                    frame_idx = total_frames - 1
                    # pop the last frame from stack since we are going back to it
                    if video_stack:
                        video_stack.pop()
                        img_id -= 1
                    break 
                elif (key & 0xFF) in (ord('d'), ord('D')): # Discard
                    args._progress_state[args.vid_name].setall(False)
                    print(f' [@] Deleted the annotations')
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    return None 
                elif (key & 0xFF) in (ord('y'), ord('Y')): # Save
                    extend_annotations(video_stack)
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    return len(video_stack)
            
            if frame_idx < total_frames:
                continue
            else:
                return len(video_stack)

        orig_frame_path = os.path.join(preview_path, 'frames', f'{frame_idx:05d}.jpg')
        orig_frame_img = cv2.imread(orig_frame_path)
        if orig_frame_img is None:
            frame_idx += 1
            continue
            
        clean_frame_img = orig_frame_img.copy()
        to_save_image = orig_frame_img.copy()
        orig_h, orig_w, _ = orig_frame_img.shape
        frame_name = f"{clean_vid_name}_{frame_idx:05d}.jpg"

        frame_info = {'frame_name': frame_name,
                      'annotations': [],
                      'to_save_image': None,
                      'img_info': None}

        # -------------------------------------------------- check first if the frame is already annotated
        total_obj_annotated = 0

        for tmp_idx, (obj, _) in enumerate(configs.OBJECT_TO_ANNOTATE.items()):
            if f'{frame_idx}' in annotation_results[obj]['bb']:
                # Get class names for each BB instance (stored once for all frames)
                class_names = annotation_results[obj]['bb'].get('class_names', [])
                bbs = annotation_results[obj]['bb'][f'{frame_idx}']
                
                # Draw segmentations if available
                if f'{frame_idx}' in annotation_results[obj]['seg'] and annotation_results[obj]['seg'][f'{frame_idx}']:
                    utils.safe_draw_polygons(orig_frame_img, annotation_results[obj]['seg'][f'{frame_idx}'], color=configs.OBJECT_COLORS[obj], alpha=0.4)
                    total_obj_annotated += 1
                
                # Draw each BB with its corresponding class name
                if bbs and bbs != []:
                    for bb_idx, bb in enumerate(bbs):
                        class_name = class_names[bb_idx] if bb_idx < len(class_names) and class_names[bb_idx] else 'unknown'
                        orig_frame_img = utils.draw_cocoBB_from_dict(orig_frame_img.copy(), [bb],
                                                                    class_name,
                                                                    color=configs.OBJECT_COLORS[obj],
                                                                    orig_width=orig_w, orig_height=orig_h,
                                                                    target_size=(orig_w, orig_h)
                                                                    )
                    total_obj_annotated += 1   
            
            
        if total_obj_annotated == 0:
            # if no annotations, then skip this frame
            args._progress_state[args.vid_name][frame_idx] = False
            # print(f"   [*] Skipped frame {frame_idx} as no annotations found")
            frame_idx += 1
            continue
        
        # --------------------------------------------------  instructions
        vis_image = orig_frame_img.copy()
        
        # D-04: Dynamic banner with custom shortcuts
        acc_key = _get_display_key("accept")
        edit_key = _get_display_key("edit")
        cat_key = _get_display_key("change_category")
        v_key = _get_display_key("preplay")
        q_key = _get_display_key("quit")
        x_key = _get_display_key("exit")
        h_key = _get_display_key("help")
        
        preview_items = [
            f"[{acc_key}] Accept", f"[{edit_key}] Edit", 
            f"[{cat_key}] Class", # f"[{v_key}] Pre-play",
            f"[{q_key}] Quit BB", f"[{x_key}] Exit", f"[{h_key}] Help"
        ]
        y0, y1 = utils.draw_menu_banner(orig_frame_img, items=preview_items, frame_idx=frame_idx, total_frames=total_frames, app_mode=args.mode)
        utils.place_window(winname, winnsize=configs.winnsize)  # Place the window at the top left corner
        cv2.imshow(winname, orig_frame_img)

        key = cv2.waitKeyEx(0)
        pure_key = key & 0xFF
        
        # D-08: Pre-play source video
        if check_key(key, "preplay"):
            v_path = args.video_path
            if v_path and os.path.exists(v_path):
                print(f" [v] Pre-playing source video: {v_path}")
                cap = cv2.VideoCapture(v_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    cv2.imshow(winname, frame)
                    if cv2.waitKey(30) & 0xFF == 27: # ESC to stop
                        break
                cap.release()
                cv2.imshow(winname, orig_frame_img)
            continue

        # D-10: Frame Navigation (Arrows)
        if check_key(key, "left"):
            if frame_idx > 0:
                frame_idx -= 1
                if video_stack:
                    video_stack.pop()
            continue
        elif check_key(key, "right"):
            frame_idx += 1
            continue
        elif check_key(key, "segment_back"):
            jump = min(frame_idx, args.repeat)
            frame_idx -= jump
            for _ in range(jump):
                if video_stack: video_stack.pop()
            continue
        elif check_key(key, "segment_forward"):
            frame_idx = min(total_frames - 1, frame_idx + args.repeat)
            continue

        if check_key(key, "quit"):         # --------- QUIT video -- all annotations saved
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            extend_annotations(video_stack)
            return len(video_stack)
        
        elif check_key(key, "exit"):          # --------- EXIT the program NO ANNOTATIONS SAVED
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            extend_annotations(video_stack)
            sys.exit()
        
        elif check_key(key, "previous_frame"):          # --------- Previous frame
            frame_idx = max(0, frame_idx - 1)
            if len(video_stack):
                video_stack.pop()
            continue

        elif check_key(key, "skip"): # --------- Skip current frame
            args._progress_state[args.vid_name][frame_idx] = False
            frame_idx += 1
            continue

        elif check_key(key, "edit"):          # --------- EDIT
            reset_globals()
            frame_tracker_text = f'Frame Tracker: {frame_idx}/{total_frames-1}'
            ret_key = add_manually_bb(args, clean_frame_img, frame_idx=frame_idx, total_frames=total_frames, frame_tracker_text=frame_tracker_text)
            
            if check_key(ret_key, "left"):
                if frame_idx > 0:
                    frame_idx -= 1
                    if video_stack:
                        video_stack.pop()
                continue
            elif check_key(ret_key, "right"):
                frame_idx += 1
                continue
            elif check_key(ret_key, "segment_back"):
                jump = min(frame_idx, args.repeat)
                frame_idx -= jump
                for _ in range(jump):
                    if video_stack: video_stack.pop()
                continue
            elif check_key(ret_key, "segment_forward"):
                frame_idx = min(total_frames - 1, frame_idx + args.repeat)
                continue
            elif check_key(ret_key, "previous_frame"):
                frame_idx = max(0, frame_idx - 1)
                if len(video_stack):
                    video_stack.pop()
                continue
            elif check_key(ret_key, "jump_next"):
                next_un = args._progress_state[args.vid_name].find(0, frame_idx + 1)
                if next_un != -1:
                    frame_idx = next_un
                    print(f" [j] Jumping to next unannotated frame: {frame_idx}")
                else:
                    print(" [j] No more unannotated frames found.")
                continue
            elif check_key(ret_key, "quit"):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                extend_annotations(video_stack)
                return len(video_stack)

            if utils.no_point_selected_by_user(object_segmentations):
                cv2.destroyAllWindows()
                continue
            
            # D-12: Get range for re-editing
            try:
                print(f"\n[SAM Targeted Re-editing] Starting frame: {frame_idx}")
                ending_frame_input = input(f"Enter ending frame index (current to {total_frames-1}): ").strip()
                ending_frame = int(ending_frame_input) if ending_frame_input else total_frames - 1
                if ending_frame < frame_idx:
                    ending_frame = frame_idx
                if ending_frame >= total_frames:
                    ending_frame = total_frames - 1
            except ValueError:
                ending_frame = total_frames - 1
            
            # Classes that were actually edited in this pass
            def _was_edited(cls_name):
                lst = object_segmentations.get(cls_name, [])
                return any(utils.annotation_frame_info(a) for a in lst)

            edited_classes = {cls for cls in configs.OBJECT_TO_ANNOTATE.keys() if _was_edited(cls)}
            
            # Run targeted propagation
            max_frames = ending_frame - frame_idx + 1
            
            # D-12: Optimized re-editing - only load relevant frames to avoid CUDA OOM
            temp_editing_dir = os.path.join(preview_path, 'temp_editing')
            os.makedirs(temp_editing_dir, exist_ok=True)
            utils.reset_working_dir(temp_editing_dir)
            
            print(f"   > Preparing {max_frames} frames for re-editing...")
            for f in range(frame_idx, ending_frame + 1):
                f_name = f"{f:05d}.jpg"
                src = os.path.join(preview_path, 'frames', f_name)
                if os.path.exists(src):
                    shutil.copy(src, temp_editing_dir)
            
            # video_segments: {relative_frame_idx: {inst_id: mask}}
            # We use detected_frame_idx=0 because the first frame in temp_editing_dir is our start frame
            video_segments_rel, inst2class = second_stage(sam_video_predictor, temp_editing_dir, 0, max_frame_num_to_track=max_frames)
            
            # Map relative indices back to absolute video indices
            video_segments = { (k + frame_idx): v for k, v in video_segments_rel.items() }
            
            # Cleanup temp directory
            utils.reset_working_dir(temp_editing_dir, delete_=True)
            
            # Merge results back into annotation_results
            cleared_frame_class = set()
            for target_f, inst_masks in video_segments.items():
                for inst_id, mask in inst_masks.items():
                    cls = inst2class[inst_id]['class']
                    if cls not in edited_classes: continue
                    
                    # Convert mask to polygon and BB
                    mask_uint8 = (mask[0] * 255).astype(np.uint8) if len(mask.shape) == 3 else (mask * 255).astype(np.uint8)
                    cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polys = [c.squeeze(1).tolist() for c in cnts if cv2.contourArea(c) >= 10]
                    
                    if not polys: continue
                    
                    # Ensure keys exist
                    annotation_results[cls]['bb'].setdefault(f'{target_f}', [])
                    annotation_results[cls]['seg'].setdefault(f'{target_f}', [])
                    
                    # Clear existing for this class on this frame if we haven't yet in this merge session
                    if (target_f, cls) not in cleared_frame_class:
                        annotation_results[cls]['bb'][f'{target_f}'] = []
                        annotation_results[cls]['seg'][f'{target_f}'] = []
                        cleared_frame_class.add((target_f, cls))

                    annotation_results[cls]['seg'][f'{target_f}'].append(polys)
                    
                    if cls in configs.OBJECT_WITH_BB:
                        ys, xs = np.where(mask_uint8)
                        if xs.size:
                            bb = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())]
                            annotation_results[cls]['bb'][f'{target_f}'].append(bb)
                    
                    args._progress_state[args.vid_name][target_f] = True

            # D-12: Auto-accept propagated segment and resume after it if enabled
            if args.auto_accept:
                print(f" [✓] Auto-accepting segment: {frame_idx} to {ending_frame}")
                for f in range(frame_idx, ending_frame + 1):
                    _commit_frame(f)
                frame_idx = ending_frame + 1
            else:
                print(f" [!] Segment propagated. Manual review mode: please accept [a] or skip [s] each frame.")
                # We stay at frame_idx to allow manual review
            
            cv2.destroyAllWindows()
            continue

        elif check_key(key, "change_category"):             # ---------  Change category in-GUI
            def create_highlighted_image(base_img, target_class, highlight_bb_idx=None):
                """Create spotlight effect - darken background, keep bounding box areas bright
                
                Args:
                    base_img: Base image to process
                    target_class: The class type (TOOL/DEVICE) to highlight
                    highlight_bb_idx: If provided, only highlight this specific BB index, otherwise highlight all
                """
                darkened = base_img.copy()
                overlay = np.zeros_like(darkened, dtype=np.uint8)
                cv2.addWeighted(darkened, 0.3, overlay, 0.7, 0, darkened)
                
                # Restore bright areas for target class bounding boxes
                if f'{frame_idx}' in annotation_results[target_class]['bb']:
                    bbs = annotation_results[target_class]['bb'][f'{frame_idx}']
                    if bbs and bbs != []:
                        for idx, bb in enumerate(bbs):
                            if isinstance(bb, list) and len(bb) == 4:
                                x, y, w, h = map(int, map(np.ceil, bb))
                                # Only highlight the specific BB if index is provided
                                if highlight_bb_idx is None or idx == highlight_bb_idx:
                                    darkened[y:y+h, x:x+w] = base_img[y:y+h, x:x+w]
                                    cv2.rectangle(darkened, (x, y), (x + w, y + h), (0, 255, 255), 3)
                                    # Draw index number on the BB
                                    cv2.putText(darkened, f"#{idx+1}", (x+5, y+25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                else:
                                    # Slightly highlight other BBs of same class
                                    dim_color = tuple(int(c * 0.6) for c in (0, 255, 255))
                                    cv2.rectangle(darkened, (x, y), (x + w, y + h), dim_color, 1)
                                    cv2.putText(darkened, f"#{idx+1}", (x+5, y+25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, dim_color, 1)
                
                # Draw other objects with dimmed colors
                for obj_type in configs.OBJECT_TO_ANNOTATE.keys():
                    if obj_type != target_class and f'{frame_idx}' in annotation_results[obj_type]['bb']:
                        bbs = annotation_results[obj_type]['bb'].get(f'{frame_idx}', [])
                        for idx, bb in enumerate(bbs):
                            if isinstance(bb, list) and len(bb) == 4:
                                x, y, w, h = map(int, map(np.ceil, bb))
                                dim_color = tuple(int(c * 0.5) for c in configs.OBJECT_COLORS[obj_type])
                                cv2.rectangle(darkened, (x, y), (x + w, y + h), dim_color, 1)
                                cv2.putText(darkened, f"#{idx+1}", (x+5, y+25), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, dim_color, 1)
                    
                    if f'{frame_idx}' in annotation_results[obj_type]['seg']:
                        segs = annotation_results[obj_type]['seg'][f'{frame_idx}']
                        if segs:
                            dim_color = tuple(int(c * 0.4) for c in configs.OBJECT_COLORS[obj_type])
                            utils.safe_draw_polygons(darkened, segs, color=dim_color, alpha=0.2)
                
                return darkened

            # Initialize class name tracking - ONCE per object type (not per frame)
            # Find the first frame with BBs to determine instance count
            for obj_type in configs.OBJECT_WITH_BB:
                if 'class_names' not in annotation_results[obj_type]['bb']:
                    # Find first frame with BBs to get instance count
                    num_instances = 0
                    for fid in annotation_results[obj_type]['bb'].keys():
                        if fid != 'class_names' and isinstance(annotation_results[obj_type]['bb'][fid], list):
                            bbs = annotation_results[obj_type]['bb'][fid]
                            if bbs and bbs != []:
                                num_instances = max(num_instances, len(bbs))
                    
                    if num_instances > 0:
                        # Initialize class_names as a list (one per instance, applies to ALL frames)
                        annotation_results[obj_type]['bb']['class_names'] = [None] * num_instances

            image_with_object = orig_frame_img.copy()
            
            # Iterate through each object type that has bounding boxes
            for obj_type in configs.OBJECT_WITH_BB:
                if f'{frame_idx}' not in annotation_results[obj_type]['bb']:
                    continue
                    
                bbs = annotation_results[obj_type]['bb'].get(f'{frame_idx}', [])
                if not bbs or bbs == []:
                    continue
                
                # Ensure class_names structure exists (list of names, one per instance)
                if 'class_names' not in annotation_results[obj_type]['bb']:
                    annotation_results[obj_type]['bb']['class_names'] = [None] * len(bbs)
                
                # Iterate through each instance
                for bb_idx in range(len(bbs)):
                    # Get the persistent display label from IdManager
                    base = configs.OBJECT_TO_ANNOTATE[obj_type]
                    ID_SPACING = 1000
                    inst_id = base * ID_SPACING + bb_idx
                    display_label = id_manager.get_display_label(inst_id)

                    # Show current name if already assigned
                    current_name = None
                    if bb_idx < len(annotation_results[obj_type]['bb']['class_names']):
                        current_name = annotation_results[obj_type]['bb']['class_names'][bb_idx]
                    
                    if current_name:
                        print(f"     [→] {display_label} current name: '{current_name}' (renaming...)")
                    
                    choose_category = 0
                    init_class = obj_type  # Start with the current object type
                    
                    while True:
                        # Create highlighted image for this specific BB
                        highlighted_base = create_highlighted_image(image_with_object, init_class, bb_idx)
                        image_with_classes = highlighted_base.copy()
                        
                        # Filter categories and get display color
                        filtered_categories = utils._filter_categories_by_supercategory(
                            args.annotations['categories'], init_class
                        )
                        color = configs.OBJECT_COLORS[init_class]
                        
                        # Draw category selection panel with BB indicator
                        panel_title = f"{display_label} ({bb_idx+1}/{len(bbs)})"
                        if current_name:
                            panel_title += f" [Current: {current_name}]"
                        metrics = utils._draw_category_selection_panel(
                            image_with_classes, filtered_categories, 
                            panel_title, y1, color
                        )
                        cv2.imshow(winname, image_with_classes)
                        
                        # Get user input
                        user_input = utils._get_category_input(
                            winname, image_with_classes, filtered_categories, init_class, metrics
                        )
                        
                        if user_input == -10:  # Change category type
                            choose_category += 1
                            available_types = list(configs.OBJECT_WITH_BB)
                            init_class = available_types[choose_category % len(available_types)]
                            continue
                            
                        if user_input == -1:  # Skip this instance
                            print(f"     [!] Skipped category selection for {obj_type} instance #{bb_idx+1}")
                            break
                        
                        # Handle category selection for this instance (applies to ALL frames)
                        # Find category by ID (not by index)
                        selected_cat = None
                        for cat in filtered_categories:
                            if cat['id'] == user_input:
                                selected_cat = cat
                                break
                        
                        if selected_cat:
                            action = "Renamed" if current_name else "Assigned"
                            annotation_results[obj_type]['bb']['class_names'][bb_idx] = selected_cat['name']
                            print(f"     [✓] {action} '{selected_cat['name']}' to {obj_type} instance #{bb_idx+1} (ALL frames)")
                            break
                        elif user_input >= 0:
                            print(f"     [!] Invalid category ID: {user_input}")
                            continue
                        else:
                            continue
            
            # After assigning all class names, go back one frame to re-accept
            frame_idx = max(0, frame_idx - 1)
            continue

        elif check_key(key, "accept"):   # ---------  Accept the current BB
            missing_class_name = False
            
            # Check if all objects with bounding boxes have class names assigned
            for obj in annotation_results.keys():
                if obj in configs.OBJECT_WITH_BB:
                    bbs = annotation_results[obj]['bb'].get(f'{frame_idx}', [])
                    if bbs and bbs != []:
                        # Check if class_names exist and are assigned for all instances
                        class_names = annotation_results[obj]['bb'].get('class_names', [])
                        if not class_names or len(class_names) < len(bbs) or any(cn is None for cn in class_names[:len(bbs)]):
                            print(f"   [!] Not all {obj} instances have categories assigned.")
                            missing_class_name = True
                            
            if missing_class_name:
                continue
            else:
                _commit_frame(frame_idx, clean_img=to_save_image, vis_img=vis_image)
        
        elif check_key(key, "help"):          # ---------  Help
            acc_key = _get_display_key("accept")
            edit_key = _get_display_key("edit")
            cat_key = _get_display_key("change_category")
            v_key = _get_display_key("preplay")
            q_key = _get_display_key("quit")
            x_key = _get_display_key("exit")
            h_key = _get_display_key("help")
            s_key = _get_display_key("skip")
            p_key = _get_display_key("previous_frame")

            help_text = ['Help Menu:',
                         f' [{acc_key}] Accept BB/Seg',
                         f' [{edit_key}] Edit BB/Seg',
                         f' [{cat_key}] Change category',
                         # f' [{v_key}] Pre-play source',
                         f' [{s_key}] Skip frame',
                         f' [{p_key}] Previous frame',
                         ' [Arrows] Navigate 1 frame',
                         f' [{q_key}] Quit (Save All)',
                         f' [{x_key}] Exit (No Save)',
                         f' [{h_key}] Show help'
            ]
            utils.draw_menu_panel(orig_frame_img, help_text, start_xy=(y0, y1 + 100),
                                  bg_color=configs.COLORS['panel_color'],
                                  text_color=configs.COLORS['menu_class'],
                                  line_gap=configs.LINE_GAP)
            cv2.imshow(winname, orig_frame_img)
            cv2.waitKey(0)
            cv2.imshow(winname, vis_image)
            continue

        else:
            print("   [!] Pressed Uknown Keywords !! try again")
            continue
        
        frame_idx += 1

    
#########################################################
#                YOLO + SAM2 Auxiliary                  #
#########################################################
def run_propagation(inference_state, start_frame_idx=None, max_frame_num_to_track=None):
  video_segments = {}  # video_segments contains the per-frame segmentation results
  for out_frame_idx, out_obj_ids, out_mask_logits in sam_video_predictor.propagate_in_video(
      inference_state, 
      start_frame_idx=start_frame_idx, 
      max_frame_num_to_track=max_frame_num_to_track
  ):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
  return video_segments


def second_stage(sam_video_predictor, output_frames_path, detected_frame_idx, max_frame_num_to_track=None):
    global object_segmentations

    inference_state = sam_video_predictor.init_state(video_path=output_frames_path)
    INST2CLASS = {}  # instance-id registry for later decoding
    # choose a spacing so ids never collide across classes
    # e.g., TOOL base 10 -> 1000, 1001, 1002... ; DEVICE base 20 -> 2000, 2001...
    ID_SPACING = 1000

    for semantic_cls, base in configs.OBJECT_TO_ANNOTATE.items():
        for k, ann in enumerate(object_segmentations[semantic_cls]):   # one entry per user instance
            inst_id = base * ID_SPACING + k
            # Use id_manager to get/create a descriptive label for this instance
            display_label = id_manager.get_or_create_label(inst_id, semantic_cls)
            INST2CLASS[inst_id] = {'class': semantic_cls, 'idx': k, 'display_label': display_label}

            # rectangle?
            if ann['start_point'] and ann['end_point']:
                x1, y1 = ann['start_point']
                x2, y2 = ann['end_point']
                xyxy = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                sam_video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=detected_frame_idx,
                    obj_id=inst_id,
                    box=xyxy
                )

            # points?
            if ann['include_points']:
                pts = np.array(ann['include_points'] + ann['exclude_points'], dtype=np.float32)
                lbl = np.array([1]*len(ann['include_points']) + [0]*len(ann['exclude_points']), dtype=np.int32)
                sam_video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=detected_frame_idx,
                    obj_id=inst_id,
                    points=pts,
                    labels=lbl
                )

    video_segments = run_propagation(inference_state, start_frame_idx=detected_frame_idx, max_frame_num_to_track=max_frame_num_to_track)   # {frame: {inst_id: mask}}
    
    # MEMORY CLEANUP: Clear state and cache to prevent OOM
    try:
        sam_video_predictor.reset_state(inference_state)
        del inference_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f" [!] Warning: Failed to clear SAM2 state: {e}")

    return video_segments, INST2CLASS


def sam_prompt_to_polygons(img_bgr, eps=1.5, min_area=10):
    global sam_predictor, object_segmentations
    sam_predictor.set_image(img_bgr[..., ::-1].copy())

    def _mask_to_polys(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.squeeze(1).tolist() for c in cnts if cv2.contourArea(c) >= min_area]

    def _run(box=None, points=None, labels=None):
        masks, _, _ = sam_predictor.predict(
            box=np.asarray([box]) if box is not None else None,
            point_coords=np.asarray(points) if points is not None else None,
            point_labels=np.asarray(labels) if labels is not None else None,
            multimask_output=False)
        return masks[0].astype(np.uint8)

    results = {cls: [] for cls in configs.OBJECT_TO_ANNOTATE.keys()}

    # rectangles
    for cls in configs.OBJECT_TO_ANNOTATE.keys():
        for ann in object_segmentations[cls]:
            if ann['start_point'] and ann['end_point']:
                x1, y1 = ann['start_point']; x2, y2 = ann['end_point']
                mask = _run(box=[min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
                entry = {"segs": _mask_to_polys(mask)}
                ys, xs = np.where(mask)
                if xs.size and cls in configs.OBJECT_WITH_BB:
                    entry["bbox"] = [[int(xs.min()), int(ys.min()),
                                      int(xs.max()-xs.min()), int(ys.max()-ys.min())]]
                results[cls].append(entry)

    # points
    for cls in configs.OBJECT_TO_ANNOTATE.keys():
        for ann in object_segmentations[cls]:
            inc, exc = ann['include_points'], ann['exclude_points']
            if not inc and not exc:
                continue
            pts = np.array(inc + exc, np.float32)
            lbl = np.array([1]*len(inc) + [0]*len(exc), np.int32)
            mask = _run(points=pts, labels=lbl)
            entry = {"segs": _mask_to_polys(mask)}
            ys, xs = np.where(mask)
            if xs.size and cls in configs.OBJECT_WITH_BB:
                entry["bbox"] = [[int(xs.min()), int(ys.min()),
                                  int(xs.max()-xs.min()), int(ys.max()-ys.min())]]
            results[cls].append(entry)

    return results

#########################################################
#             Annotation Functionality                  #
#########################################################
def locate_tool_from_frame(args,
                           frame_image,
                           last_detected_frame,
                           indices,
                           total_frames,
                           ask_user=False,):
    """
    Locate manually by points the tool in the current frame
    """
    global winname, next_video, segment_by_points, next_category
    if ask_user:
        # Add text to the frame
        question_img = frame_image.copy()
        
        acc_key = _get_display_key("accept")
        exit_key = _get_display_key("exit")
        c_key = _get_display_key("next_category")
        v_key = _get_display_key("next_video")
        
        menu_list = [f'[{acc_key}] Annotate    [{c_key}] Next Category    [{v_key}] Next Video    [{exit_key}] Exit Program']

        _left_bounds, left_metrics = utils.draw_menu_panel(question_img, menu_list,
                              start_xy=(10, 10),
                              bg_color=configs.COLORS['panel_color'],
                              text_color=configs.COLORS['menu_class'],
                              line_gap=configs.LINE_GAP)

        # Right-side DB metadata panel — draw before cv2.imshow
        db_path = getattr(args, 'db_path', None)
        vid_name = getattr(args, 'vid_name', None)
        if db_path and vid_name:
            metadata = utils.lookup_maintenance_metadata(
                db_path, vid_name, category=getattr(args, 'category', None)
            )
            if metadata:
                db_lines = [f"{label}:  {value}" for label, value in metadata.items()]
                utils.draw_menu_panel(question_img, db_lines,
                                      start_xy=(10, left_metrics["panel_bottom"] + 20),
                                      bg_color=configs.COLORS['panel_color'],
                                      text_color=configs.COLORS['menu_class'],
                                      alpha=0.8,
                                      line_gap=configs.LINE_GAP)

        cv2.imshow(winname, question_img)

        while True:
            key = cv2.waitKey(0)
            if check_key(key, "next_video"):
                next_video = True
                return False, -1, -1
            elif check_key(key, "exit"):
                cv2.destroyAllWindows()
                exit(0)
            elif check_key(key, "next_category"):
                next_category = True
                return False, -1, -1
            elif check_key(key, "accept"):
                break

        ask_user = False

    frame_tracker_text = f'Annotating {indices[0]} - {indices[-1]}/{total_frames-1}'
    ret_key = add_manually_bb(args, frame_image, frame_idx=indices[0], total_frames=total_frames, frame_tracker_text=frame_tracker_text)
    return ask_user, last_detected_frame, ret_key


def does_yolo_failed(last_detected_frame, i, boarder=9):
    """
    The yolo should run on the first 10 frames, if failed to detect,
    make the user detected manually.

    :param last_detected_frame: last frames bb detected
    :param i                  : the current frame index
    :param boarder            : number of frames for yolo to check
    :return:
    """
    if np.abs(last_detected_frame - i) > boarder:
        return True
    return False


def annotate_all_video_manually(v_path, curr_tool_output_path=None,
                                frame_names=None,
                                frame_paths=None,
                                preview_before_save=False, n=50):
    """
    Manually annotate $n samples from the input video in a roboflow scheme annotations.
    :param v_path: path to the video
    :param curr_tool_output_path: path to the result directory
    :param n: number of samples to choose
    :return:
    """
    global winname, image_copy, clean_state
    vid_name = os.path.split(v_path)[-1].split('.')[0]
    if curr_tool_output_path is None:
        sub_root_saved_path = os.path.join(curr_tool_output_path, f'{vid_name}')
    else:
        sub_root_saved_path = curr_tool_output_path

    frame_paths = os.path.join(sub_root_saved_path, 'frames')
    os.makedirs(sub_root_saved_path, exist_ok=True)
    os.makedirs(frame_paths, exist_ok=True)

    if frame_names is None:
        utils.video_to_frames(v_path, frame_paths)
        frame_names = [
            p for p in os.listdir(frame_paths)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # work only with 100 frames
        frame_names = utils.get_k_frames(frame_names)

    saved_path = os.path.join(sub_root_saved_path, f'BBvis')
    os.makedirs(saved_path, exist_ok=True)
    first_frame = cv2.imread(os.path.join(frame_paths, frame_names[0]))
    frame_length = len(frame_names)
    samples = range(len(frame_names))  # no rndom choice
    height, width, _ = first_frame.shape
    fps = 30
    if v_path is not None:
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    out = cv2.VideoWriter(os.path.join(sub_root_saved_path, f'bbVIS_{vid_name}.mp4'),
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))
    tool_bbs = {}
    for counter, sample_idx in enumerate(samples):
        frame_number = int(frame_names[sample_idx].split('.')[0])
        frame_image = cv2.imread(os.path.join(frame_paths, frame_names[sample_idx]))
        winname = f'{frame_number}/{frame_length}'
        print(f'{counter + 1}/{len(samples)}')
        image_copy = frame_image.copy()  # Create a copy for resetting during drawing
        clean_state = frame_image.copy()
        cv2.imshow(winname, frame_image)
        cv2.setMouseCallback(winname, mouse_callback)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saved_path, f'{frame_number:05d}.jpg'), image_copy)
        out.write(image_copy)
        if start_point is not None and end_point is not None:
            tool_bbs[f'{sample_idx}'] = utils.convert_bb_to_yolo_format([*start_point, *end_point], width, height)
        else:
            #   the frame
            if not utils.prompt_question(True, query="Are you sure?"):
                print(f'DELETING THIS BB {sample_idx}')
                tool_bbs[f'{sample_idx}'] = [-1, -1, -1, -1]

    out.release()

    if preview_before_save:
        run_preview(os.path.join(sub_root_saved_path, f'bbVIS_{vid_name}.mp4'))

    return tool_bbs, {}


def annotate_video_using_sam(args,
                             curr_tool_output_path=None,
                             extract_tool=True,
                             frame_names=None,
                             preview_before_save=False, debug=False):
    """
    This function is responsible for annotating the given video by:
        - point prompts
        - Hand segmentations as a matrix representation for the mask

    @param args: namespace contains all the user choices
    @param curr_tool_output_path: output path to save the results
    @param extract_tool: a flag whether to extract BB (if a tool is used in the video)
    @param frame_names: a flag whether to extract the hand segmentations
    @param opt_for_manual: a flag whether to ask the user for manually add bb
    @param preview_before_save: play the results as a video (bb results)
    @param debug: whether to debug the algorithm steps
    @return tool_bbs, hands_segmentations: a dict for each frame the bbs and hand seg.
    """
    global winname, image_copy, clean_state, segment_by_points, next_video, object_segmentations
    v_path = args.video_path
    segment_by_points = True
    if curr_tool_output_path is None: # wrong here, fix
        sub_root_saved_path = os.path.join(curr_tool_output_path, f'{args.vid_name}')
    else:
        sub_root_saved_path = curr_tool_output_path

    frame_paths = os.path.join(sub_root_saved_path, 'frames')
    os.makedirs(sub_root_saved_path, exist_ok=True)
    os.makedirs(frame_paths, exist_ok=True)

      # for now, pass the annotaTED VIDEOS (even if only 10 frames are annotated)
    if args.vid_name in args._progress_state and args._progress_state[args.vid_name].count(True) > 0 and args_parser.pass_annotated:
        print(f'   [*] Skipping video {args.vid_name} as it has {args._progress_state[args.vid_name].count(True)}/{len(args._progress_state[args.vid_name])} annotated frames')
        # utils.reset_working_dir(sub_root_saved_path, delete_=True)
        return

    if frame_names is None:
        with utils.suppress_output():  # to suppress the output to stdout
            utils.video_to_frames_slow(v_path, frame_paths)   # change this to video_to_frames
        frame_names = [
            p for p in os.listdir(frame_paths)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    current_working_file = os.path.join(sub_root_saved_path, 'working_file')  # file that holds n images each time
    os.makedirs(current_working_file, exist_ok=True)

    # Reset IdManager for new video
    id_manager.reset()
    
    # Try to load existing ID mapping if this is a resume/refine session
    # Standard location: results_coco_format/{date}/{category}/{vid_name}/id_mapping.json
    # But since date varies, we look in the output_dir root or the video's specific folder if known.
    # For now, we'll try to find it in the current sub_root_saved_path which is where we're working.
    mapping_path = os.path.join(sub_root_saved_path, 'id_mapping.json')
    if os.path.exists(mapping_path):
        id_manager.load_mapping(mapping_path)
        print(f" [✓] Loaded persistent ID mapping from {mapping_path}")

    INST2CLASS_ALL = {}
    combined_video_segments = {}
    success_indices = []
    ask_user = True   # ask if skip current video (SAM2 purpose)n
    skipped_frames = {}  # none annotated frames
    i = 0
    last_detected_frame = i
    total_frames = len(frame_names)
    # ----- init progress state for this video
    args._progress_state[args.vid_name] = utils.ensure_bitset(args._progress_state, args.vid_name, total_frames)
    winname = f'{args.vid_name} ({args.category})- # Unannotated Frames - {args._progress_state[args.vid_name].count(False)}/{total_frames}'
    utils.place_window(winname, winnsize=configs.winnsize)  # Place the window at the top-left corner
    while i < total_frames - 1:
        # print(f' current: {i}   total: {total_frames}     {args._progress_state[args.vid_name]}')
        reset_globals()
        # print(f' ------- {args._progress_state}')
        if extract_tool:
            # Stage 1: point prompts or manual annotation
            frame_number = int(frame_names[i].split('.')[0])
            if args._progress_state[args.vid_name][frame_number]:
                last_detected_frame = i
                i += 1
                continue

            # Prepare the current window of frames
            indices = utils.find_range(i, total_frames, bit_state=args._progress_state[args.vid_name], range_step=args.repeat)
            if debug:
                print(f'   > Starting with {indices[0]} - {indices[-1]}/{len(indices)} frames from {len(frame_names)} frames')

            frame_image = cv2.imread(os.path.join(frame_paths, frame_names[i]))
            if frame_image is None:
                print(f'   [!!] frame {frame_names[i]} not opening')
                continue

            ask_user, last_detected_frame, ret_key = locate_tool_from_frame(args, frame_image,
                                                                 last_detected_frame,
                                                                 indices,
                                                                 total_frames,
                                                                 ask_user=ask_user)

            
            if check_key(ret_key, "left") or check_key(ret_key, "previous_frame"): # Left Arrow or P
                i = max(0, i - 1)
                continue
            elif check_key(ret_key, "right"): # Right Arrow
                i = min(total_frames - 1, i + 1)
                continue
            elif check_key(ret_key, "segment_back"): # Segment Back
                i = max(0, i - args.repeat)
                continue
            elif check_key(ret_key, "segment_forward"): # Segment Forward
                i = min(total_frames - 1, i + args.repeat)
                continue
            elif check_key(ret_key, "jump_next"): # Jump to Next Unannotated
                next_un = args._progress_state[args.vid_name].find(0, i + 1)
                if next_un != -1:
                    i = next_un
                    print(f" [j] Jumping to next unannotated frame: {i}")
                else:
                    print(" [j] No more unannotated frames found.")
                continue
            elif check_key(ret_key, "quit"):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return 0 # return 0 to trigger @completed but with 0 frames (effectively saving what's done)

            if next_video:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return None # return None to trigger @discarded
            
            if next_category:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return None

            if utils.no_point_selected_by_user(object_segmentations):
                skipped_frames[f'{i}'] = [[-1, -1, -1, -1]]
                args._progress_state[args.vid_name][frame_number] = False
                i += 1
                continue


            # Stage 2: apply SAM2 on the manually bb
            if debug:
                for obj_id in object_segmentations.keys():
                    print(f'{obj_id}: ')
                    for stream in object_segmentations[obj_id]:
                        print(f' > {stream}: {object_segmentations[obj_id][stream]}')

            print(f'   > SAM FOR : {indices[0]} - {indices[-1]} / {total_frames}')
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            utils.initial_working_dir(frame_paths, frame_names, current_working_file, indcies=indices)
            # update the state:
            video_segments, inst2class  = second_stage(sam_video_predictor, current_working_file, 0)
            video_segments = {k + i: v for k, v in video_segments.items()}
            combined_video_segments = {**combined_video_segments, **video_segments}
            INST2CLASS_ALL.update(inst2class)
            success_indices.append(indices)

        # Algorithm step
        i = indices[-1] + 1
        last_detected_frame = indices[-1]

    cv2.destroyAllWindows()
    # SAVING SAM RESULTS
    annotation_results = utils.extract_annotations(success_indices, combined_video_segments, INST2CLASS_ALL, format='coco')

    # to ensure the correct annotation, run a preview and edit the one that needed to editted.
    if preview_before_save:
        # D-05: Standardize per-video output isolation with date hierarchy
        date_str = datetime.now().strftime("%Y-%m-%d")
        video_coco_dir = os.path.join(args.coco_data, date_str, args.category, args.vid_name)
        os.makedirs(os.path.join(video_coco_dir, 'images'), exist_ok=True)
        
        # Reset current video annotations (keeping categories)
        args.annotations['images'] = []
        args.annotations['annotations'] = []
        
        total_annotated_frames = run_preview(args, sub_root_saved_path, annotation_results, video_coco_dir, save_vis=args.save_visualization)
        
        # Save per-video results
        utils.save_json_file(args.annotations, os.path.join(video_coco_dir, 'annotations.json'))
        print(f" [✓] Saved COCO annotations to {video_coco_dir}/annotations.json")
        
        # Save persistent ID mapping
        id_manager.save_mapping(os.path.join(sub_root_saved_path, 'id_mapping.json'))
        print(f" [✓] Saved ID mapping to {sub_root_saved_path}/id_mapping.json")

    utils.reset_working_dir(frame_paths, delete_=True)
    utils.reset_working_dir(current_working_file, delete_=True)

    return total_annotated_frames

################################################################################################################ START Fix COCO

def fix_annotations(args, target_size=(640, 460)):
    """
    Fix the annotations of the given directory path
    :param args: user options
    :return:
    """
    global winname, start_point, end_point
    if 'coco_annotation_tracker_idx' not in args.tracker:
        args.tracker['coco_annotation_tracker_idx'] = -1

    start_id = args.tracker['coco_annotation_tracker_idx']

    """
    current mapping:

    adjs -> wrench
    tape -> plier
    drill -> ratchet
    plier -> tapemeasure
    ratchet -> hammer
    """

    new_coco = utils.load_coco_annotations(args.output_path) if os.path.exists(args.output_path) else {
            "images": [],
            "annotations": [],
            "categories":[],
        }
    coco = utils.load_coco_annotations(args.annotations_path)
    coco["images"].sort(key=lambda x: x["id"])  # or x["file_name"]
    coco["annotations"].sort(key=lambda x: x["id"])

    for img_idx, img_info in enumerate(coco["images"]):
        if img_info["id"] <= start_id:  # already done.
            continue
        
        reset_globals()
        image_path = Path(args.images_path) / img_info["file_name"]
        if not image_path.exists():
            print(f"[Missing] {image_path}")
            continue

        try:
            img = Image.open(image_path)
            orig_width, orig_height = img.size
            # img = img.resize(target_size)
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            scale_x = orig_width / target_size[0] 
            scale_y = orig_height / target_size[1]
        except Exception as e:
            print(f"[INVALID IMAGE] {image_path} -> {e}")
            continue
        print(f'img_dx: {img_idx} - {img_info["file_name"]}')
        anns = coco["annotations"][img_idx]

        # -----------------------------------  Edditing bb
        exit_flag = False
        acc_key = _get_display_key("accept")
        edit_key = _get_display_key("edit")
        cat_key = _get_display_key("change_category")
        q_key = _get_display_key("quit")
        while True:
            image_display = utils.draw_cocoBB_from_annotations(img_np.copy(), [anns], configs.category_id_to_name,
                                                       orig_width=orig_width, orig_height=orig_height,
                                                       target_size=(orig_width, orig_height))
            instr = f"[{acc_key}] Accept   [{edit_key}] Edit BB   [{cat_key}] Class   [{q_key}] Quit"
            cv2.putText(image_display, instr, (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                        0.75, (255, 255, 0), 2, cv2.LINE_AA)

            winname = f"Fix Annotations - class: ({orig_width},{orig_height}){configs.category_id_to_name.get(anns['category_id'], 'Unknown')} ({img_idx + 1}/{len(coco['images'])})"
            cv2.imshow(winname, image_display)
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):  # Quit
                exit_flag = True
                break

            elif key == ord('e'):  # Edit bounding box
                # cv2.destroyAllWindows()
                reset_globals()
                add_manually_bb(args, img_np, frame_idx=img_idx, total_frames=len(coco["images"]))
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    new_w, new_h = abs(x2 - x1), abs(y2 - y1)
                    anns['bbox'] = [min(x1, x2), min(y1, y2), new_w, new_h]
                    anns['area'] = new_w * new_h
                    print(f" Updated BB: {[x1, y1, new_w, new_h]}")
                else:
                    print(" No BB drawn. Annotation will be skipped.")
                    take_it = False
                    break  # skip to next image

            elif key == ord('n'):  # Change category in-GUI
                image_with_classes = image_display.copy()
                y_offset = 60
                class_instr_lines = []
                class_instr = ""
                for cid, cname in configs.category_id_to_name.items():
                    class_instr += f"[{cid}] {cname}   "
                    if cid % 3 == 2:
                        class_instr_lines.append(class_instr.strip())
                        class_instr = ""
                if class_instr:
                    class_instr_lines.append(class_instr.strip())
                for line in class_instr_lines:
                    cv2.putText(image_with_classes, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 30
                cv2.imshow(winname, image_with_classes)
                key2 = cv2.waitKey(0) & 0xFF
                selected_class = key2 - ord('0')
                if selected_class in configs.category_id_to_name:
                    anns["category_id"] = selected_class
                    print(f" Updated category to {configs.category_id_to_name[selected_class]}")
                else:
                    print(" Invalid class. No change.")

            elif key in (ord('a'), 13):  # Accept (Enter or 'a')
                take_it = True
                break

        if exit_flag:
            break

        if take_it:
            print(f"✔ Accepted: {anns['bbox']} / class: {configs.category_id_to_name[anns['category_id']]}")
            new_coco['images'].append(img_info)
            new_coco['annotations'].append(anns)

        cv2.destroyAllWindows()
        args.tracker['coco_annotation_tracker_idx'] = img_info["id"]

        # --------------------------------------------------  finnished
        # save the tracker
        with open('tracker.json', 'w') as f:
            json.dump(args.tracker, f, indent=4)

        # save the new annotations
        new_coco["categories"] = coco["categories"]
        print(f"Fixed annotations saved to {args.output_path}")
        print(f"Last processed image ID: {args.tracker['coco_annotation_tracker_idx']}")
        utils.save_json_file(new_coco, args.output_path)

################################################################################################################ End Fix COCO

def annotator(args, fixer=False):
    """
    The introduction of annotating a video. Here, the annotation scheme is done based on user options
    :param args: user options
    :return:
    """
    global segment_by_points, next_category
    tracker = utils.read_json(tracker_path)
    args.tracker = tracker

    if hasattr(args, 'vid_name') and any(m in sys.argv or args.mode == "4" for m in ["4"]):
        # Mode 4: Resume/Refine logic
        curr_tool_output_path = os.path.join(args.output_dir, args.vid_name)
        os.makedirs(curr_tool_output_path, exist_ok=True)
        # We call annotate_video_using_sam which will handle the preview/edit loop
        annotate_video_using_sam(args, curr_tool_output_path, preview_before_save=True)
        return

    if args.video_path is not None:     # Single video
        vid_name = os.path.split(args.video_path)[-1].split('.')[0]
        curr_tool_output_path = os.path.join(args.output_dir, vid_name)
        os.makedirs(curr_tool_output_path, exist_ok=True)
        if args.manually:
            total_annotated_frames = annotate_all_video_manually(args.video_path, curr_tool_output_path)
        else:
            total_annotated_frames = annotate_video_using_sam(args, curr_tool_output_path, preview_before_save=True)

        # print(f'tool_bbs: {tool_bbs}\n\nhand_segmentations: {hands_segmentations}')


    elif args.directory_path:
        tool_categories = [tool_cat for tool_cat in os.listdir(args.directory_path) if '.DS_' not in tool_cat ]#and tool_cat in configs.CATEGORIES]
        np.random.shuffle(tool_categories)
        for tool_category in tool_categories:
            if not os.path.isdir(os.path.join(args.directory_path, tool_category)) or 'annot' in tool_category.lower():
                print(f' > Skipping {tool_category} as it is not a directory')
                continue
                             
            # if tool_category not in ['Screw']:
            #     print(f' > Skipping {tool_category} as it is not in the supported categories')
            #     continue
            print(f'Annotating {tool_category} videos')
            args.curr_tool_id = args.category_id_mapping[tool_category] if tool_category in args.category_id_mapping else -1
            videos = []
            videos = [vid for vid in os.listdir(os.path.join(args.directory_path, tool_category)) if vid.endswith(('.mp4', '.MP4', '.mov'))]
            np.random.shuffle(videos)
            args.category = tool_category

            next_category = False
            for stam_idx, video in enumerate(videos):
                # if stam_idx < 100:
                #     continue
                if next_category:
                    args.timer.stop(None)
                    break
                args.timer.start()
                video_path = os.path.join(args.directory_path, tool_category, video)
                args.video_path = video_path
                vid_name = os.path.split(video_path)[-1].split('.')[0]
                if vid_name in args._progress_state and args._progress_state[vid_name].all():
                    print(f' > Already Annotated Video -- {vid_name}')
                    continue
                args.vid_name = vid_name
                curr_tool_output_path = os.path.join(args.output_dir, vid_name)
                os.makedirs(curr_tool_output_path, exist_ok=True)

                print(f' > Annotating {vid_name} --- {stam_idx}/{len(videos)}')
                if args.manually:
                    total_annotated_frames = annotate_all_video_manually(args.video_path, curr_tool_output_path)
                else:
                    total_annotated_frames = annotate_video_using_sam(args, curr_tool_output_path, preview_before_save=True)
                
                # D-03 & D-04: File Routing Lifecycle
                date_str = datetime.now().strftime("%Y-%m-%d")
                if total_annotated_frames is not None:
                    # D-03: Accepted - move to @completed
                    dest_dir = Path(f"@completed/{date_str}/{tool_category}")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    print(f" [✓] Moving {video} to {dest_dir}")
                    shutil.move(video_path, dest_dir / video)
                else:
                    # D-04: Discarded - move to @discarded and cleanup results
                    dest_dir = Path(f"@discarded/{date_str}/{tool_category}")
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    print(f" [!] Moving {video} to {dest_dir} (Discarded)")
                    shutil.move(video_path, dest_dir / video)
                    if os.path.exists(curr_tool_output_path):
                        print(f" [!] Cleaning up temporary results in {curr_tool_output_path}")
                        shutil.rmtree(curr_tool_output_path)
                
                args.timer.stop(total_frames=total_annotated_frames if total_annotated_frames is not None else 0)
                print(f' > Finished annotating {total_annotated_frames if total_annotated_frames is not None else 0} frames of {vid_name} in {args.timer.format_time()}')
                print('==========================================================')
            
            # D-02: Pause between categories
            print(f"\n[Finished Category: {tool_category}]")
            input("Press Enter to continue to the next folder...")

                # print(timer.get_timer_statistics())
                

    elif args.fixer:  # Fix annotations
        # Fix the annotations
        print(f'Fixing annotations in {args.directory_path}')
        fix_annotations(args)


def ask_user_for_run_config():
    """
    Query the user for the desired annotation mode and store the
    answers in a global variable called `args`.
    """
    global args, args_parser                    # the rest of the script expects this

    # Load settings from JSON file
    settings_path = 'settings.json'
    if os.path.exists(settings_path):
        print(f"\n=== Loading settings from {settings_path} ===")
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    else:
        print(f"\n[!] {settings_path} not found. Using default interactive setup.")
        # Fallback to interactive mode or error out. 
        # For Phase 01, we want to enforce settings.json.
        raise FileNotFoundError(f"Critical configuration file {settings_path} missing.")

    # Update shortcuts from settings if present
    if "shortcuts" in settings:
        for k, v in settings["shortcuts"].items():
            if k in configs.SHORTCUTS:
                configs.SHORTCUTS[k] = v

    mode = settings.get("mode", "2")

    weights_name_mapping = {
        't': 'tiny',
        's': 'small',
        'b': 'base_plus',
        'l': 'large',
    }

    # initialise defaults --------------------------
    args = SimpleNamespace(
        video_path=settings.get("video_path"),
        directory_path=settings.get("directory_path"),
        output_dir=settings.get("output_dir", "annotation_results"),
        weights=weights_name_mapping[args_parser.weights],
        manually=settings.get("manually", False),
        repeat=args_parser.repeat,
        fixer=False,
        timer=utils.Timer(),
        coco_data=settings.get("coco_data"),
        done_video_names=None,
        _progress_state={},
        save_visualization=settings.get("save_visualization", False),
        auto_accept=settings.get("auto_accept", configs.AUTO_ACCEPT),
        hand_segmentation_automation=settings.get("hand_segmentation_automation", configs.HAND_SEGMENTATION_AUTOMATION),
        db_path=settings.get("db_path", None),
        new_shape=args_parser.new_shape,
        mode=mode
    )
    # ---------------------------------------------- single video
    if mode == "1":                               
        if not args.video_path:
             raise ValueError("video_path must be specified in settings.json for mode 1")
    # ---------------------------------------------- Directory
    elif mode == "2":                            
        if not args.directory_path:
             raise ValueError("directory_path must be specified in settings.json for mode 2")
        if not args.coco_data:
            args.coco_data = 'results_coco_format'
        
        # Ensure images directory exists
        Path(os.path.join(args.coco_data, 'images')).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(os.path.join(args.coco_data, 'annotations.json')):
            categories = []
            category_mapping_name_to_id = {}
            obj_id = 0
            for super_category, mapping_ in configs.OBJECT_CLASSES.items():
                for _, obj in mapping_.items():
                    categories.append({
                        "id": obj_id,
                        "name": f"{obj}",
                        "supercategory": f"{super_category}"
                    })
                    category_mapping_name_to_id[obj.lower()] = obj_id
                    obj_id += 1
            args.annotations = {'images': [], 'annotations': [], "categories": categories}
            args.category_mapping_name_to_id = category_mapping_name_to_id
            with open(os.path.join(args.coco_data, 'annotations.json'), 'w') as f:
                json.dump(args.annotations, f, indent=4)
            args.done_video_names = set()
        else: 
            print(f"Loading annotations from {os.path.join(args.coco_data, 'annotations.json')}")
            with open(os.path.join(args.coco_data, 'annotations.json'), 'r') as f:
                args.annotations = json.load(f)
            
            if os.path.exists(os.path.join(args.coco_data, 'time_tracker.json')):
                with open(os.path.join(args.coco_data, 'time_tracker.json'), 'r') as f:
                    args.timer.set_timer_statistics(json.load(f))
          
            # extract all names of video
            if os.path.exists(os.path.join(args.coco_data, "done_video_names.pkl")):
                args._progress_state = utils.save_open_video_names_as_pickles(None, path=os.path.join(args.coco_data, "done_video_names.pkl"), op='open')
            
            # check for new categories
            for super_cat, mapping_ in configs.OBJECT_CLASSES.items():
                for _, obj in mapping_.items():
                    if obj.lower() not in [cat['name'].lower() for cat in args.annotations['categories']]:
                        new_cat_id = len(args.annotations['categories'])
                        args.annotations['categories'].append({
                            "id": new_cat_id,
                            "name": obj,
                            "supercategory": super_cat
                        })

            args.category_mapping_name_to_id = {cat['name'].lower(): cat['id'] for cat in args.annotations['categories']}

        atexit.register(lambda: utils.save_json_file(args.timer.timer_statistics, os.path.join(args.coco_data, 'time_tracker.json')))
        atexit.register(lambda: utils.save_json_file(args.annotations, os.path.join(args.coco_data, 'annotations.json')))
        atexit.register(lambda: utils.save_open_video_names_as_pickles(args._progress_state, path=os.path.join(args.coco_data, 'done_video_names.pkl'), op='save'))
        args.category_id_mapping = {cat['name'].lower(): cat['id'] for cat in args.annotations['categories']}
        args.id_category_mapping = {cat['id']: cat['name'].lower() for cat in args.annotations['categories']}
    # ---------------------------------------------- Fixer
    elif mode == "3":
        args.fixer = True
        coco_folder_path = settings.get("coco_data")
        if not coco_folder_path:
             raise ValueError("coco_data must be specified in settings.json for mode 3")
        args.images_path = os.path.join(coco_folder_path, 'images')
        args.annotations_path = os.path.join(coco_folder_path, 'annotations.json')
        args.output_path = os.path.join(coco_folder_path, 'fix_annotations.json')
    
    # ---------------------------------------------- Resume/Refine
    elif mode == "4":
        print("\n=== Mode 4: Resume/Refine Finished Video ===")
        resume_path = input("Enter path to video result folder (e.g. results_coco_format/2026-04-26/Piping/v16_GOPR3872): ").strip()
        if not os.path.exists(resume_path):
            raise ValueError(f"Path does not exist: {resume_path}")
        
        args.vid_name = os.path.basename(resume_path)
        args.category = os.path.basename(os.path.dirname(resume_path))
        args.coco_data = os.path.dirname(os.path.dirname(os.path.dirname(resume_path)))
        
        ann_path = os.path.join(resume_path, 'annotations.json')
        if not os.path.exists(ann_path):
             raise ValueError(f"No annotations.json found in {resume_path}")
        
        with open(ann_path, 'r') as f:
            args.annotations = json.load(f)
        
        # Determine the source assets path based on category
        # This is a bit of a guess based on standard project structure
        args.directory_path = "assets"
        v_exts = ('.mp4', '.MP4', '.mov')
        v_file = None
        for f in os.listdir(os.path.join(args.directory_path, args.category)):
            if f.startswith(args.vid_name) and f.endswith(v_exts):
                v_file = f
                break
        
        if not v_file:
            # Try searching in completed
            date_str = os.path.basename(os.path.dirname(os.path.dirname(resume_path)))
            comp_dir = os.path.join("@completed", date_str, args.category)
            if os.path.exists(comp_dir):
                for f in os.listdir(comp_dir):
                    if f.startswith(args.vid_name) and f.endswith(v_exts):
                        args.video_path = os.path.join(comp_dir, f)
                        break
        else:
            args.video_path = os.path.join(args.directory_path, args.category, v_file)
            
        if not hasattr(args, 'video_path'):
            print(f" [!] Could not find source video for {args.vid_name}. Pre-play will not work.")
            args.video_path = None

        # Setup progress state (all done)
        args._progress_state = {args.vid_name: bitarray('1' * 10000)} # Large enough buffer
        
        # Redirect output to itself for refinement
        args.output_dir = "annotation_results"
        print(f" [✓] Ready to refine: {args.vid_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n--------------------------------------------------------------------")
    print("Arguments:")
    for k, v in vars(args).items():
        if k not in ['annotations', 'category_mapping_name_to_id', 'category_id_mapping', 'id_category_mapping', '_progress_state']:
            print(f"  {k:20s}: {v}")
    print("--------------------------------------------------------------------\n")
    return args


if __name__ == "__main__":
    args = ask_user_for_run_config()
    annotator(args)
    if args.coco_data is not None:
        utils.save_open_video_names_as_pickles(args._progress_state, path=os.path.join(args.coco_data, 'done_video_names.pkl'), op='save')
    
