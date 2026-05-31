import cv2
import os
import subprocess
import shutil
import glob
import numpy as np
from itertools import chain
import json
from contextlib import contextmanager
import sys
import random
import sys
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
from copy import deepcopy
from tqdm import tqdm
import pickle
import re
import sqlite3
import bitarray
# from AnnotationScheme import configs
import configs
import time
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


#########################################################
#          Category Selection Helper Functions          #
#########################################################
def _filter_categories_by_supercategory(categories, supercategory):
    """Filter categories by supercategory matching."""
    filtered = []
    for cat in categories:
        cat_super = cat.get('supercategory', 'tool').lower()
        if cat_super == supercategory.lower():
            filtered.append(cat)
    return filtered


def _draw_category_selection_panel(image, filtered_categories, init_class, y1, color):
    """Draw category selection panel with list of categories."""
    img_h, img_w = image.shape[:2]
    
    # Calculate panel dimensions with enhanced font settings
    margin_x = max(10, int(0.02 * img_w))
    font = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font than SIMPLEX
    fscale = max(0.7, min(1.0, 0.8 * (img_w / 1280.0)))  # Slightly larger
    thick = 2
    line_gap = 1.5  # More spacing for readability
    
    # Build class lines with actual IDs
    class_lines = [f"[{cat['id']}] {cat['name']}" for cat in sorted(filtered_categories, key=lambda x: x['id'])]
    class_lines.append("[c] Custom Label (type new name)")
    
    # Measure dimensions
    line_h = 0
    max_line_w = 0
    for ln in class_lines:
        (tw, th), _ = cv2.getTextSize(ln, font, fscale, thick)
        max_line_w = max(max_line_w, tw)
        line_h = max(line_h, int(th * 1.2))
    
    # Position below banner
    pad_x, pad_t = 12, 10
    start_y = y1 + 10
    available_h = img_h - start_y - 20
    
    title = f"Category (press [n] to change):   {init_class}"
    (title_w, title_h), _ = cv2.getTextSize(title, font, fscale + 0.1, thick)
    
    panel_w = min(img_w - 2*margin_x, max(max_line_w, title_w) + 2*pad_x)
    total_content_h = pad_t + int((len(class_lines) + 1) * line_h * line_gap) + pad_t
    panel_h = min(total_content_h, available_h)
    
    # Draw the panel
    draw_translucent_panel(image, margin_x, start_y, panel_w, panel_h,
                                 color=configs.COLORS.get('classes_background', (40, 40, 40)), alpha=0.75)
    
    # Draw title with enhanced rendering
    y = start_y + pad_t + line_h
    cv2.putText(image, title, (margin_x + 5, int(y)),
                font, fscale + 0.15, color, thick, cv2.LINE_AA)
    y += int(line_h * line_gap * 1.3)
    
    # Draw class options with enhanced rendering
    x_left = margin_x + pad_x
    for i, ln in enumerate(class_lines):
        if y + line_h > start_y + panel_h - pad_t:
            break
        cv2.putText(image, ln, (x_left, int(y)),
                   font, fscale, (220, 220, 220), thick, cv2.LINE_AA)  # Brighter, thicker
        y += int(line_h * line_gap)
    
    # Calculate prompt position
    prompt_y = start_y + panel_h + int(line_h * 1.5)
    if prompt_y + line_h + 10 > img_h:
        prompt_y = int(line_h * 2)
    
    return {
        'margin_x': margin_x,
        'prompt_y': prompt_y,
        'line_h': line_h,
        'fscale': fscale,
        'thick': thick
    }


def _get_category_input(winname, image_with_classes, filtered_categories, init_class, metrics):
    """Get user input for category selection with live prompt."""
    typed = ""
    font_input = cv2.FONT_HERSHEY_DUPLEX  # Match panel font
    margin_x = metrics['margin_x']
    prompt_y = metrics['prompt_y']
    line_h = metrics['line_h']
    fscale = metrics['fscale']
    thick = metrics['thick']
    
    while True:
        vis = image_with_classes.copy()
        
        # Draw semi-transparent background for prompt with better contrast
        prompt_text = f"Enter category ID or 'c' for custom: {typed or '_'}"
        prompt_size = cv2.getTextSize(prompt_text, font_input, fscale, thick)[0]
        prompt_bg_x1 = margin_x - 5
        prompt_bg_y1 = prompt_y - line_h - 5
        prompt_bg_x2 = margin_x + prompt_size[0] + 10
        prompt_bg_y2 = prompt_y + 10
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (prompt_bg_x1, prompt_bg_y1),
                     (prompt_bg_x2, prompt_bg_y2), (30, 30, 30), -1)  # Darker background
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)  # More opaque
        
        cv2.putText(vis, prompt_text, (margin_x, prompt_y),
                   font_input, fscale, (0, 255, 255), thick, cv2.LINE_AA)
        cv2.imshow(winname, vis)
        
        key_input = cv2.waitKey(0) & 0xFF
        
        if key_input in (13, 32):  # Enter/Space - submit
            if typed.lower() == 'c':
                return -999  # Custom label marker
            elif typed.isdigit():
                typed_id = int(typed)
                # Verify this category ID exists in filtered categories
                if any(cat['id'] == typed_id for cat in filtered_categories):
                    return typed_id
                else:
                    print(f"     [!] Invalid category ID {typed_id} for {init_class}")
                    typed = ""
                    continue
            else:
                typed = ""
                continue
        elif key_input in (27, ord('q')):  # Esc/q - cancel
            return -1
        elif key_input in (ord('n'), ord('N')):  # Change category type
            return -10
        elif key_input in (8, 127):  # Backspace
            typed = typed[:-1]
        elif ord('0') <= key_input <= ord('9') or key_input in (ord('c'), ord('C')):
            typed += chr(key_input)


def _handle_category_selection(user_input, filtered_categories, init_class, args, annotation_results):
    """Handle category selection - create new or use existing category."""
    if user_input == -999:  # Custom label
        custom_label = input("\n    Enter custom label name: ").strip()
        if custom_label:
            # Create new category with proper ID
            existing_ids = [cat['id'] for cat in args.annotations['categories']]
            new_cat_id = max(existing_ids, default=-1) + 1
            
            args.annotations['categories'].append({
                'id': new_cat_id,
                'name': custom_label,
                'supercategory': init_class.lower()
            })
            args.category_mapping_name_to_id[custom_label.lower()] = new_cat_id
            
            # Update the class_name in annotation_results
            annotation_results[init_class]['bb']['class_name'] = custom_label.lower()
            print(f"    [*] Added new category: [{new_cat_id}] {custom_label}")
            return True
    elif user_input >= 0:
        # User entered valid category ID
        selected_cat = next((cat for cat in filtered_categories if cat['id'] == user_input), None)
        if selected_cat:
            annotation_results[init_class]['bb']['class_name'] = selected_cat['name'].lower()
            print(f"    [*] Updated category of {init_class} to [{user_input}] {selected_cat['name']}")
            return True
        else:
            print("     [!] Invalid category ID.")
            return False
    else:
        print("     [!] Invalid selection.")
        return False


def lookup_maintenance_metadata(db_path, vid_name, category=None):
    """
    Look up maintenance metadata for a video by its filename stem.
    Returns a dict with 7 display fields (Category, Hands, Tool, Device,
    Component, Length, Resolution), or None if not found or DB unavailable.
    @param db_path:   Path to the SQLite DB file. None disables the feature.
    @param vid_name:  Video filename stem (no extension, no directory prefix).
                      May carry a v<N>_ prefix from the local filesystem
                      (e.g. 'v682_0000000003'); the prefix is stripped before
                      matching so it doesn't need to appear in the DB.
    @param category:  Optional category name (e.g. 'Measuring'). When provided,
                      used to disambiguate rows whose DB path contains the same
                      stem in multiple categories.
    """
    if not db_path or not os.path.exists(db_path):
        return None
    import re as _re
    clean_name = _re.sub(r'^v\d+_', '', vid_name or '')
    first_match = None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT video_path, category, hands, tool, device, component, length_, resolution "
                "FROM MaintenanceActions_metadata"
            )
            for row in cur:
                db_stem = os.path.basename(row["video_path"] or "").split('.')[0]
                if db_stem != clean_name:
                    continue
                result = {
                    "Category":   row["category"] or "—",
                    "Hands":      str(row["hands"]) if row["hands"] is not None else "—",
                    "Tool":       row["tool"] or "—",
                    "Device":     row["device"] or "—",
                    "Component":  row["component"] or "—",
                    "Length":     f"{row['length_']} frames" if row["length_"] is not None else "—",
                    "Resolution": row["resolution"] or "—",
                }
                if first_match is None:
                    first_match = result
                # Prefer the row whose DB path contains the category name
                if category and category.lower() in (row["video_path"] or "").lower():
                    return result
    except Exception:
        pass  # DB unavailable — fail silently per D-03
    return first_match


#########################################################
#              Drawing & Visualization                  #
#########################################################
def draw_menu_panel(img, lines, *,
                    banner_bounds=None,   # (y_top, y_bottom) from your top menu banner
                    start_xy=None,        # (x,y) to override default placement
                    text_color=(255,240,191),  # readable cyan-white (BGR)
                    bg_color=(60,60,60),  # dark grey panel
                    alpha=0.75,
                    thickness=2,
                    font=None,
                    font_scale=None,
                    line_gap=1.35,
                    pad_x=12, pad_t=10, pad_y=10, margin_x=None):
    """
    Draw a small translucent panel with each string in `lines` on a new row.
    Returns: (panel_top, panel_bottom), metrics dict
    """
    H, W = img.shape[:2]
    if font is None:
        font = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font
    if font_scale is None:
        font_scale = max(0.75, min(1.5, 0.95 * (W / 1280.0)))  # Slightly larger
    if margin_x is None:
        margin_x = max(10, int(0.02 * W))

    # Default position: just below the top banner
    if start_xy is None:
        y0 = (banner_bounds[1] if banner_bounds else 0) + pad_y
        x0 = margin_x
    else:
        x0, y0 = start_xy

    # Measure longest line + line height
    max_w, line_h = 0, 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        max_w = max(max_w, tw)
        line_h = max(line_h, int(th * 1.2))

    panel_w = min(W - 2*margin_x, max_w + 2*pad_x)
    panel_h = int(pad_t + len(lines) * line_h * line_gap + pad_t)

    # Background
    draw_translucent_panel(img, x0, y0, panel_w, panel_h, color=bg_color, alpha=alpha)

    # Draw text
    x_text = x0 + pad_x
    y = y0 + pad_t + line_h
    for ln in lines:
        cv2.putText(img, ln, (x_text, int(y)), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += int(line_h * line_gap)

    metrics = {
        "x_left": x_text,
        "line_h": line_h,
        "font_scale": font_scale,
        "thickness": thickness,
        "panel_right": x0 + panel_w,
        "panel_bottom": y0 + panel_h,
    }
    return (y0, y0 + panel_h), metrics


def draw_menu_banner(img, items=None, frame_idx=0, total_frames=0, thickness=2, top=True, alpha=0.75, mode=None, app_mode=None):
    """
    Draw a responsive banner:
      • menu items (wrapped, with exactly 3 spaces between items)
      • a counter line BELOW the menu (alone in its own line)
      • mode name and number in the TOP RIGHT
    """
    h, w = img.shape[:2]

    # D-04: Dynamic items based on mode if not provided
    if items is None:
        if mode == 'edit':
            # Remove [l-click]/[r-click] instructions as requested
            items = ["[u] Undo", "[m] Next Obj", "[d] Toggle Mode", "[Enter] Done"]
        elif mode == 'preview':
            items = ["[a] Accept", "[e] Edit", "[n] Class", "[v] Pre-play",
                     "[q] Quit BB", "[x] Exit", "[h] Help"]
        else:
            items = []

    # Map numeric mode to English names
    mode_names = {
        "1": "Single Video",
        "2": "Directory",
        "3": "Fixer",
        "4": "Resume/Refine"
    }

    margin_x = max(10, int(0.02 * w))
    margin_y = max(8,  int(0.015 * h))
    font      = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font
    font_base = 0.85  # Slightly larger base
    font_scale = max(0.7, min(1.5, font_base * (w / 1280.0)))  # Enhanced range
    line_gap   = configs.LINE_GAP

    # --- wrap by items (keep exactly 3 spaces between items)
    sep = "   "
    lines, line = [], ""
    max_width = w - 2 * margin_x
    for item in items:
        trial = line + (sep if line else "") + item
        (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = item
    if line:
        lines.append(line)

    # --- metrics
    (_, th), _ = cv2.getTextSize("Ag", font, font_scale, thickness)
    line_h   = int(th * 1.2)
    menu_h   = int(len(lines) * line_h * line_gap)
    counter_text = f"Frame Tracker: {frame_idx}/{max(1, total_frames-1)}"   # keep your old convention
    counter_h    = line_h                                    # same line height

    # total banner height = margins + menu lines + counter line
    banner_h = int(2 * margin_y + menu_h + counter_h)

    # --- translucent panel
    y0 = 0 if top else h - banner_h
    draw_translucent_panel(img, 0, y0, w, banner_h, color=configs.COLORS['panel_color'], alpha=alpha)

    # --- draw menu lines
    y = y0 + margin_y + line_h
    for ln in lines:
        cv2.putText(img, ln, (margin_x, int(y)),
                    font, font_scale, configs.COLORS['menu_class'], thickness, cv2.LINE_AA)
        y += int(line_h * line_gap)

    # --- draw counter on its own line (below menu)
    cv2.putText(img, counter_text, (margin_x, int(y)),
                font, font_scale, configs.COLORS['menu_class'], thickness, cv2.LINE_AA)

    # --- draw mode in top right
    if app_mode:
        mode_str = mode_names.get(str(app_mode), "Unknown")
        mode_display = f"Mode {app_mode}: {mode_str}"
        (mw, mh), _ = cv2.getTextSize(mode_display, font, font_scale, thickness)
        cv2.putText(img, mode_display, (w - mw - margin_x, y0 + margin_y + mh),
                    font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return y0, y0 + banner_h


def draw_translucent_panel(img, x, y, w, h, color=(40,40,40), alpha=0.75):
    """Draw a semi-transparent rectangle over (x,y,w,h)."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)


def build_classes_list(img, init_class, banner_bounds=None,
                       margin_x=None, pad_y=10,
                       min_font=0.50,            # lower bound for readability
                       min_gap=1.00):            # lower bound for line spacing
    """
    Draw a translucent panel containing:
      - title: "Category (press [n] to change): <init_class>"
      - one row per class
      - reserved space for a 'Class id:' line below the list
    The function automatically shrinks line_gap then font_scale until it fits.
    Returns: (panel_top, panel_bottom), img, mapping, metrics
    """
    mapping = configs.OBJECT_CLASSES[init_class]
    color   = configs.OBJECT_COLORS[init_class]

    H, W = img.shape[:2]
    if margin_x is None:
        margin_x = max(10, int(0.02 * W))

    font     = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font
    fscale   = max(0.75, min(1.5, 0.95 * (W / 1280.0)))  # Enhanced scale
    thick    = 2
    line_gap = float(getattr(configs, "LINE_GAP", 1.35))

    start_y = (banner_bounds[1] if banner_bounds else 0) + pad_y
    title   = f"Category (press [n] to change):   {init_class}"
    lines   = [title] + [f"[{cid}] {cname}" for cid, cname in mapping.items()]

    pad_x, pad_t = 12, 10
    bottom_margin = 8
    reserved_extra_lines = 2  # one blank + one "Class id:" line

    def measure(scale, gap):
        max_w = 0
        line_h = 0
        for ln in lines:
            (tw, th), _ = cv2.getTextSize(ln, font, scale, thick)
            max_w = max(max_w, tw)
            line_h = max(line_h, int(th * 1.2))
        panel_w = min(W - 2 * margin_x, max_w + 2 * pad_x)
        panel_h = int(pad_t + (len(lines) + reserved_extra_lines) * line_h * gap + pad_t)
        return panel_w, panel_h, line_h

    # shrink until panel fits the available vertical space
    avail_h = H - start_y - bottom_margin
    panel_w, panel_h, line_h = measure(fscale, line_gap)
    while panel_h > avail_h and (line_gap > min_gap or fscale > min_font):
        if line_gap > min_gap:
            line_gap = max(min_gap, line_gap - 0.05)
        else:
            fscale = max(min_font, fscale * 0.95)
        panel_w, panel_h, line_h = measure(fscale, line_gap)

    # cap to available height (safety)
    panel_h = min(panel_h, avail_h)

    # draw background panel full width or measured width
    draw_translucent_panel(img, margin_x, start_y, panel_w, panel_h,
                           color=configs.COLORS['classes_background'], alpha=0.75)

    # draw text
    y = start_y + pad_t + line_h
    x_title = margin_x                    # title aligned with panel edge
    x_items = margin_x + pad_x            # items slightly indented
    for i, ln in enumerate(lines):
        x = x_title if i == 0 else x_items
        cv2.putText(img, ln, (x, int(y)),
                    font, fscale, color, thick, cv2.LINE_AA)
        y += int(line_h * line_gap)

    # where to draw "Class id:" → one blank line below the last item,
    # but never beyond the panel bottom
    y_after_list = y + int(line_h * line_gap)
    class_prompt_y = min(start_y + panel_h - pad_t, y_after_list)

    metrics = {
        "x_left": x_items,
        "y_title": y + pad_t + line_h,
        "line_h": line_h,
        "font_scale": fscale,
        "thickness": thick,
        "class_prompt_y": int(class_prompt_y)
    }
    return (start_y, start_y + panel_h), img, mapping, metrics


def ask_class(img, max_id, prompt_win=None, panel_metrics=None,
              color=(0,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    typed = ""

    # Place hint ABOVE the title line (inside panel top padding)
    if panel_metrics:
        x = panel_metrics["x_left"]
        lh = panel_metrics["line_h"]
        y = panel_metrics["y_title"] - int(0.35 * lh)   # safely above title
        fscale   = panel_metrics["font_scale"]
        thick    = panel_metrics["thickness"]
    else:
        # fallback
        h, w = img.shape[:2]
        x = max(10, int(0.02*w)); y = 120
        fscale = max(0.7, min(1.4, 0.9 * (w/1280.0)))
        thick  = 2

    while True:
        if prompt_win is not None:
            vis = img.copy()
            cv2.putText(vis, f"Class id: {typed or '_'}", (x, int(y)),
                        font, fscale, color, thick, cv2.LINE_AA)
            cv2.imshow(prompt_win, vis)

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):                 # Enter/Space
            if typed:
                cid = int(typed)
                if 0 <= cid < max_id:
                    return cid
                else:
                    print('   [!] OUT OF BOUNDRIES - select different class')
                    typed = ""
            typed = ""
            continue
        if key in (27, ord('q')):           # Esc
            return -1
        if key in (ord('n'), ord('N')):     # change group
            return -10
        if key in (8, 127):                 # backspace/delete
            typed = typed[:-1]
        elif ord('0') <= key <= ord('9'):
            typed += chr(key)


def safe_draw_polygons(img, polygons, color, alpha=0.4):
    """
    Draw filled polygons robustly.

    Accepts any of:
      - single polygon:          [[x,y], [x,y], ...]
      - list of polygons:        [poly1, poly2, ...]
      - list of instances:       [[polyA1, polyA2], [polyB1], ...]
      - dicts containing 'segs': {'segs': [poly1, poly2], ...}
    """

    def is_point_like(pt):
        try:
            return len(pt) == 2 and np.isscalar(pt[0])
        except Exception:
            return False

    def iter_leaf_polys(obj):
        """Yield leaf polygons as lists of (x,y)."""
        if obj is None:
            return
        # dict with segmentation
        if isinstance(obj, dict):
            # common keys you used: 'segs', 'seg', 'polys'
            for key in ("segs", "seg", "polys"):
                if key in obj:
                    yield from iter_leaf_polys(obj[key])
                    return
            return
        # list/tuple/ndarray
        if isinstance(obj, (list, tuple, np.ndarray)):
            if len(obj) == 0:
                return
            first = obj[0]
            # a single polygon: [[x,y], [x,y], ...]
            if is_point_like(first):
                yield obj
            else:
                # list of polygons or list of instances -> recurse
                for sub in obj:
                    yield from iter_leaf_polys(sub)

    if (
        polygons is None
        or (isinstance(polygons, (list, tuple)) and len(polygons) == 0)
        or (isinstance(polygons, np.ndarray) and polygons.size == 0)
    ):
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    for poly in iter_leaf_polys(polygons):
        p = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if p.shape[0] < 3:
            continue
        p[:, 0] = p[:, 0].clip(0, w-1)
        p[:, 1] = p[:, 1].clip(0, h-1)
        cv2.fillPoly(overlay, [p.reshape(-1, 1, 2)], color)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def place_window(winname, winnsize=(1280, 720)):
    """
    Place the window at the top-left corner of the screen.
    """
    # cv2.namedWindow(winname, cv2.WINDOW_NORMAL)       # Create once with normal mode
    cv2.namedWindow(winname, cv2.WINDOW_GUI_NORMAL)       # Create once with normal mode
    cv2.resizeWindow(winname, *winnsize)               # Resize once (you can pick any size)
    cv2.moveWindow(winname, 0, 0)


def draw_cocoBB_from_annotations(img_np, annotations, category_id_to_name, color=(0, 255, 0), orig_width=224, orig_height=224, target_size=(640, 480)):
    for ann in annotations:
        x, y, w, h = ann['bbox']
        scale_x = target_size[0] / orig_width
        scale_y = target_size[1] / orig_height
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_name = category_id_to_name.get(ann['category_id'], "Unknown")
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return img_np


def draw_cocoBB_from_dict(img_np, annotations, class_name, color=(0, 255, 0), orig_width=224, orig_height=224, target_size=(640, 480)):
    for ann in annotations:
        if ann is not None and len(ann) == 4:  # Ensure ann is a valid bbox
            x, y, w, h = ann
            scale_x = target_size[0] / orig_width
            scale_y = target_size[1] / orig_height
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # class_name = category_id_to_name.get(ann['category_id'], "Unknown")
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, class_name, (x1, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 3, cv2.LINE_AA)
        else:
            # print(f"Invalid annotation found: {ann}. Skipping drawing.")
            continue
    return img_np

#########################################################
#                    Arithmetics                        #
#########################################################
def _combined_coco_box_from_mask(bin_mask):
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def _polys_from_mask(bin_mask, min_area=100):
    contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            polys.append(c.squeeze(1).tolist())
    return polys


def extract_annotations(success_indices, combined_video_segments, INST2CLASS, format='coco'):
    """
    Per-instance decoding using INST2CLASS:
      results[<semantic>]['bb'][frame]  -> [ [x,y,w,h], ... ] (one per instance)
      results[<semantic>]['seg'][frame] -> [ poly1, poly2, ... ] (union list per instance)
    """
    success_indices = list(chain.from_iterable(success_indices))
    results = {cls: {'bb': {'class_name': None}, 'seg': {}} for cls in configs.OBJECT_TO_ANNOTATE.keys()}
    
    for frame_idx in success_indices:
        # init per frame
        for cls in results:
            results[cls]['bb'].setdefault(f'{frame_idx}', [])
            results[cls]['seg'].setdefault(f'{frame_idx}', [])

        if frame_idx not in combined_video_segments:
            continue

        for inst_id, mask_logits in combined_video_segments[frame_idx].items():
            meta = INST2CLASS.get(inst_id)
            if not meta:
                continue
            semantic = meta['class']

            mask = np.array(mask_logits)
            if mask.ndim == 3:
                mask = mask[0]
            mask = (mask > 0).astype(np.uint8)
            segs = _polys_from_mask(mask, min_area=100)
            if segs:
                results[semantic]['seg'][f'{frame_idx}'].append(segs)

            if semantic in configs.OBJECT_WITH_BB:
                coco_box = _combined_coco_box_from_mask(mask)
                if coco_box is not None:
                    if format == 'coco':
                        results[semantic]['bb'][f'{frame_idx}'].append(coco_box)
                    else:  # YOLO normalized
                        h, w = mask.shape
                        x, y, ww, hh = coco_box
                        xc = (x + ww/2.0) / w
                        yc = (y + hh/2.0) / h
                        results[semantic]['bb'][f'{frame_idx}'].append([xc, yc, ww/w, hh/h])

    return results


def convert_bb_to_yolo_format(bb, w, h):
    x1, y1, x2, y2 = bb
    combined_w, combined_h = np.abs(x2 - x1), np.abs(y2 - y1)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = combined_w / w
    height = combined_h / h

    return [x_center, y_center, width, height]


def rescale_bbox_x1y1x2y2(bbox, orig_size, new_size):
    """
    Rescale COCO bbox [x_min, y_min, x_max, y_max] from original image size to new image size.
    
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


def rescale_bbox_x1y1wh(bbox, orig_size, new_size):
    """
    Rescale a COCO-format bounding box [x, y, w, h] to a new image size.
    
    :param bbox: list or tuple [x1, y1, w, h]
    :param orig_size: (orig_width, orig_height)
    :param new_size: (new_width, new_height)
    :return: Rescaled bbox [x1', y1', w', h']
    """
    x1, y1, w, h = bbox
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    return [new_x1, new_y1, new_w, new_h]


def rescale_polygon(poly, orig_size, new_size):
    """
    Rescale a polygon from original image size to new image size.
    @param poly: List of points in the polygon, each point is a tuple (x, y).
    @param orig_size: Tuple (orig_width, orig_height) of the original image size.
    @param new_size: Tuple (new_width, new_height) of the new image size.
    @return: List of points in the rescaled polygon.
    """
    W0, H0 = orig_size
    W1, H1 = new_size
    sx, sy = W1 / W0, H1 / H0

    def _is_point_like(item):
        """True if item is a scalar or a 1-D array/seq of length 2."""
        try:
            return len(item) == 2 and np.isscalar(item[0])
        except TypeError:
            return False

    def _rescale(obj):
        # Obj is a polygon (list of points) when its first element is a point
        if _is_point_like(obj[0]):
            arr = np.asarray(obj, dtype=np.float32)
            arr *= np.array([sx, sy], np.float32)
            return arr.tolist()
        else:
            # Otherwise it is a list / ndarray of polygons – recurse
            return [_rescale(sub) for sub in obj]

    return _rescale(poly)


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.overall_time = None
        self.running = False
        self.timer_statistics = {}

    def start(self):
        """Start the timer."""
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            print("Timer is already running.")

    def stop(self, total_frames=0):
        """Stop the timer and return elapsed time in seconds."""
        if self.running:
            self.end_time = time.time()
            self.running = False
            self.overall_time =  self.end_time - self.start_time
            if int(total_frames) > 0:
                if total_frames not in self.timer_statistics:
                    self.timer_statistics[total_frames] = self.overall_time
                else:
                    previous_time = self.timer_statistics[total_frames]
                    avg_time = (self.overall_time + previous_time) / 2
                    self.timer_statistics[total_frames] = avg_time
            
            self.start_time = None

    def set_timer_statistics(self, old_statistics):
        """
        Set the timer statistics from an old statistics dictionary.
        @param old_statistics: Dictionary with frame counts as keys and time taken as values.
        """
        self.timer_statistics = old_statistics

    def format_time(self):
        """Format seconds into minutes and seconds."""
        if self.overall_time is None:
            return "No time recorded"
        minutes = int(self.overall_time // 60)
        secs = int(self.overall_time % 60)
        return f"{minutes} min {secs} sec"

    def get_timer_statistics(self):
        """Get the timer statistics."""
        summary = {}
        for frames, time_taken in self.timer_statistics.items():
            minutes = int(time_taken // 60)
            secs = int(time_taken % 60)
            summary[frames] = f"{minutes} min {secs} sec"
        return summary

#########################################################
#          Coco Annotations Functionallity              #
#########################################################
def save_open_video_names_as_pickles(set_, path="done_video_names.pkl", op='save'):
    """
    Save or load video names from a pickle file so that we can keep track of which videos have been processed.
    @param set_: Set of video names to save.
    @param path: Path to the pickle file.
    @param op: Operation to perform ('save' or 'open').
    """
    # Save to file
    if op == 'save':
        with open(path, "wb") as f:
            pickle.dump(set_, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    
    if op == 'open':
        try:
            with open(path, "rb") as f:
                loaded_set = pickle.load(f)

            return loaded_set
        
        except FileNotFoundError as e:
            print(' File not found: {path} - {e}')
            return {}


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    return coco


def save_json_file(coco, json_path):
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)


def get_image_by_id(coco, image_id):
    for img in coco["images"]:
        if img["id"] == image_id:
            return img
    return None


def get_annotations_for_image(coco, image_id):
    return [ann for ann in coco["annotations"] if ann["image_id"] == image_id]



#########################################################
#                   Scheme Helpers                      #
#########################################################
def video_to_frames(video_path, output_path):
    """
    Extract all frames from the given video and save them as .jpg images
    in the specified output directory, with filenames like 00001.jpg, 00002.jpg, etc.

    :param video_path: Full path to the video file.
    :param output_path: Directory path where extracted frames will be saved.
    """
    # Ensure the output directory exists
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


def video_to_frames_slow(video_path, output_path):
    """
    Extract all frames from the given video and save them as .jpg images
    in the specified output directory, with filenames like 00000.jpg, 00001.jpg, etc.
    
    Uses OpenCV for fast, native decoding with tqdm progress bar.

    :param video_path: Full path to the video file.
    :param output_path: Directory path where extracted frames will be saved.
    """
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Extracting frames")

    frame_idx = 0
    success, frame = cap.read()
    while success:
        filename = os.path.join(output_path, f"{frame_idx:05d}.jpg")
        # if frame_idx > 300: break
        cv2.imwrite(filename, frame)
        frame_idx += 1
        pbar.update(1)
        success, frame = cap.read()

    pbar.close()
    cap.release()


def reset_working_dir(directory, delete_=False):
    if delete_:
        shutil.rmtree(directory)
        return
    jpg_files = glob.glob(os.path.join(directory, "*.jpg"))
    # Loop through and delete each .jpg file
    for file_path in jpg_files:
        os.remove(file_path)


def initial_working_dir(output_frames_path, frame_names, current_working_file, indcies):
    reset_working_dir(current_working_file)

    for idx in indcies:
        shutil.copy(os.path.join(output_frames_path, frame_names[idx]), current_working_file)

    return np.array(frame_names)[indcies]


def find_range(curr_i, n, bit_state=None, range_step=50, total_indices=20):
    """
    every range space should contain $range_step$ images per time except last range that contains > 50 < 100
    @param bit_state: bit trace of the done frame. i.e., 00011111 (the first three frames are not annotated) 
    
    """
    # print(f' ------ {bit_state}')
    m = re.search(r'0+', bit_state.to01())
    if m.start() == curr_i and m.end() - m.start() < range_step:
        return range(m.start(), min(m.end(), n))
    
    if m.end() - m.start() > 0 and curr_i <= m.end() and m.end() <= range_step:
        return range(curr_i, min(m.end(), n))

    else:
        if curr_i + range_step < n:
            return range(curr_i,
                        min(curr_i + range_step, n))  # select_random_indices(range(curr_i, curr_i + range_step), total_indices)
        return range(curr_i, n)  # select_random_indices(range(curr_i, n), total_indices)


def ensure_bitset(state, video_name, n_frames):
    """Return bitarray for video, creating or expanding it if necessary."""
    if video_name not in state:
        state[video_name] = bitarray.bitarray(n_frames)
        state[video_name].setall(False)
    elif len(state[video_name]) < n_frames:          # video re-encoded?
        extra = n_frames - len(state[video_name])
        state[video_name].extend([False]*extra)
    return state[video_name]


def get_k_frames(frame_names, k=100, dist=None):
    # choose only k frames to annotate (no need to annotate all frames)
    if len(frame_names) > k:
        if dist:
            indices = np.arange(len(frame_names))
            mu = (len(frame_names) - 1) / 2.0  # Center of the array
            sigma = len(frame_names) / 4.0  # Adjust spread based on array length
            probs = np.exp(-(indices - mu) ** 2 / (2 * sigma ** 2))
            probs /= probs.sum()  # Normalize to sum to 1
            # Randomly select indices without replacement
            selected_indices = np.random.choice(indices, size=k, replace=False, p=probs).tolist()
            selected_indices.sort()

            return np.array(frame_names)[selected_indices].tolist()
        else:
            diff = len(frame_names) - k
            frame_names = frame_names[diff // 2: k + diff // 2]

    return frame_names


def prompt_question(ask_user, query='Next Video? [y/N]: '):
    if ask_user:
        query = ' Q) ' + query
        next_vid = input(query).lower()
        if 'y' in next_vid:  # exit this function
            return True
    return False


@contextmanager
def suppress_output():
    # Open null files for stdout and stderr
    with open(os.devnull, 'w') as devnull:
        # Save the original stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            # Redirect stdout and stderr to devnull
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def annotation_frame_info(annotation_info):

    if annotation_info['include_points'] or annotation_info['exclude_points']:
        return True
    
    if annotation_info['start_point']:
        return True
    
    return False


def no_point_selected_by_user(point_prompts=None):

    if point_prompts == None:
        return True

    for obj_id, streams in point_prompts.items():
        if len(point_prompts[obj_id]) > 0:
            for annotation in point_prompts[obj_id]:
                if annotation_frame_info(annotation):
                    return False
    
    return True


def _yes_no(prompt, default=False):
    """
    Ask a yes/no question on stdin and return True/False.
    Empty input returns the default.
    """
    while True:
        ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not ans:
            return default
        if 'y' in ans.lower():
            return True
        if 'n' in ans.lower():
            return False
        print("  ➜ Please answer with y / n")

    
def get_different_from_original(length_, k=100):
    """
    When creating the annotations, we save the bbs of the 100 choose frame to annotate with starting index as 0
    although 0 is not the original starting frame index since we choose a window of size 100 from the whole video timeline

    to get the original index, we need to shift back the window exactly the same way we shifted it to choose 100 frames

    see function utils/get_k_frames()
    """

    if length_ > k:
        return (length_ - k) // 2

    return 0    
#########################################################
#               FILES Functionality                  #
#########################################################
def write_to_json(json_path, data):
    with open(json_path, 'w') as jf:
        json.dump(data, jf, indent=4)


def read_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            data = json.load(file)  # Load JSON as dictionary
        return data
    return {}


def get_tacker_paths(tracker):
    return set(tracker.keys())


def input_with_path_completion(message="Enter path: "):
    completer = PathCompleter(only_directories=False, expanduser=True)
    return prompt(message, completer=completer)


