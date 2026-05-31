<!-- generated-by: gsd-doc-writer -->
# Configuration

This document describes the configuration options available for the Annotation-Scheme project, including environment variables, configuration files, and command-line arguments.

## Environment Variables

The project primarily uses a configuration file for settings, but certain dependencies and submodules (like the SAM2 demo backend) respect specific environment variables.

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `CUDA_VISIBLE_DEVICES` | No | (all) | Specifies which GPU devices to use for inference. |
| `MAX_UPLOAD_VIDEO_DURATION` | No | `10` | Maximum video duration (in minutes) for the demo backend. |
| `VIDEO_ENCODE_CODEC` | No | `libx264` | Codec used for video transcoding in the demo. |
| `VIDEO_ENCODE_CRF` | No | `23` | Constant Rate Factor (CRF) for video encoding. |
| `VIDEO_ENCODE_FPS` | No | `24` | Frames per second for encoded video. |
| `SAM2_DEMO_FORCE_CPU_DEVICE` | No | `0` | Set to `1` to force the demo to use CPU even if CUDA is available. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | No | `1` | Enables MPS fallback for macOS users (found in notebooks). |

## Config File Format

The primary configuration file is `settings.json`, located in the project root. This file is mandatory and bypasses interactive setup prompts.

### `settings.json` Example

```json
{
    "mode": "2",
    "directory_path": "assets",
    "output_dir": "annotation_results",
    "manually": false,
    "save_visualization": true,
    "coco_data": "results_coco_format",
    "auto_accept": false,
    "db_path": "maintenance_dataset.db",
    "hand_segmentation_automation": true,
    "shortcuts": {
        "undo": "u",
        "accept": "enter",
        "quit": "q"
    }
}
```

### Configuration Keys

| Key | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `mode` | Yes | `"2"` | Execution mode: `"1"` for single video, `"2"` for directory batch processing. |
| `directory_path` | Mode 2 | `null` | Path to the directory containing video categories. |
| `video_path` | Mode 1 | `null` | Path to a specific video file. |
| `output_dir` | No | `"annotation_results"` | Directory where per-video results are saved. |
| `coco_data` | No | `"results_coco_format"` | Directory where COCO-formatted annotations are aggregated. |
| `manually` | No | `false` | If `true`, enables manual annotation mode without SAM2 assistance. |
| `save_visualization` | No | `false` | Whether to save visualized frames (masks overlaid on images). |
| `auto_accept` | No | `false` | If `true`, automatically accepts SAM2 predictions (caution advised). |
| `hand_segmentation_automation` | No | `true` | Enables automatic hand detection using YOLO11-pose. |
| `db_path` | No | `null` | Path to the maintenance SQLite database for metadata lookup. |
| `shortcuts` | No | (see below) | Custom keyboard mappings for UI actions. |

## Internal Configurations (`AnnotationScheme/configs.py`)

Low-level settings such as UI colors, object classes, and model weight paths are defined in `AnnotationScheme/configs.py`. While these can be modified, they are not intended for frequent changes.

- **`OBJECT_TO_ANNOTATE`**: Defines the classes and their ID offsets (e.g., `TOOL: 10`, `HAND: 20`, `DEVICE: 30`).
- **`winnsize`**: Default UI window size (`1600x900`).
- **`config_weights_mapping`**: Maps weight identifiers (`t`, `s`, `b`, `l`) to their respective `.yaml` configs and `.pt` weights.

## Required vs Optional Settings

- **Required**: `settings.json` must exist in the root directory. Inside it, `mode` is required.
- **Conditional**:
  - If `mode` is `"1"`, `video_path` is required.
  - If `mode` is `"2"`, `directory_path` is required.
- **Optional**: All other keys in `settings.json` have internal defaults. If `db_path` is omitted, database-related features are disabled.

## Defaults

Default values are managed hierarchically:
1.  **Command-line arguments** (highest priority for specific flags like `--weights`).
2.  **`settings.json`** values.
3.  **Hardcoded defaults** in `AnnotationScheme/configs.py` and `AnnotationScheme/annotation_scheme.py`.

### Default Key Mappings (Shortcuts)

| Action | Default Key(s) |
| :--- | :--- |
| Undo | `u` |
| Accept | `enter`, `a`, `space` |
| Cancel/Exit | `esc`, `c` |
| Quit | `q` |
| Toggle Mode | `d` |
| Next Video | `n` |
| Next Category | `c` |
| Jump Next | `j` |

## Per-environment Overrides

The project does not currently support multiple environment-specific configuration files (e.g., `settings.dev.json`). Users typically maintain different `settings.json` files or use command-line overrides for varying environments.

<!-- VERIFY: {MAX_UPLOAD_VIDEO_DURATION environment variable usage in production environments} -->
<!-- VERIFY: {Actual base path for external SSD database if different from /Volumes/AR-FOR WIND} -->
