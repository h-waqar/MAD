<!-- generated-by: gsd-doc-writer -->
# Annotation Scheme for Object Detection & Segmentation Tasks

This repo contains the semi-automatic video annotation pipeline used in our [Maintenance Datasets] paper.

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/abs/xxx)
[![MMTL](https://img.shields.io/badge/MMTL-GitHub-black?logo=github)](https://github.com/AI-Computer-Vision-BGU/MMTL)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/your-video)

## Installation

### Prerequisites
- **Python:** >= 3.10
- **Hardware:** CUDA-capable GPU recommended for SAM2 performance.
- **Tools:** Git (with LFS support).

### Linux Setup
1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
   # Enter the cloned repository folder
   cd Annotation-Scheme
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # If you have CUDA 12.1 installed:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install SAM2 requirements and download checkpoints:**
   ```bash
   # Navigate to the segmentanything subdirectory
   cd segmentanything
   # Install SAM2 in editable mode (requires setup.py in this directory)
   pip install -e .
   # Enter the checkpoints directory within segmentanything
   cd checkpoints
   # Run the download script
   ./download_ckpts.sh
   # Return to project root
   cd ../..
   ```

### Windows Setup
1. **Clone the repository with submodules:**
   ```cmd
   git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
   :: Enter the cloned repository folder
   cd Annotation-Scheme
   ```

2. **Create and activate a virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   # If you have CUDA 12.1 installed:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install SAM2 requirements and download checkpoints:**
   ```cmd
   :: Navigate to the segmentanything subdirectory
   cd segmentanything
   :: Install SAM2 in editable mode (requires setup.py in this directory)
   pip install -e .
   :: Enter the checkpoints directory within segmentanything
   cd checkpoints
   # On Windows, you can run the script using Git Bash or download the models manually:
   sh download_ckpts.sh
   :: Return to project root
   cd ..\..
   ```

## Quick Start
1. Ensure your videos are placed in the `assets/` directory (or configure the `directory_path` in `settings.json`).
2. Launch the annotation tool:
   - **Linux:** `./run_annotation.sh`
   - **Windows:** `python AnnotationScheme/annotation_scheme.py --new_shape 680 340 --weights t`
3. Follow the interactive wizard to choose between annotating a single video, a directory, or fixing existing annotations.

## Usage Examples

### 1. Single Video Annotation
Annotate a specific video file with specific model weights and resize dimensions.
```bash
python AnnotationScheme/annotation_scheme.py --weights b --new_shape 1280 720
```

### 2. Batch Directory Processing
Iterate through a directory of videos organized by class (e.g., `assets/Hammering/video1.mp4`).
- Configure `"mode": "2"` in `settings.json`.
- The tool will automatically skip videos that are already fully annotated if `--pass_annotated` is used.

## Interactive GUI

### 1 · Initial decision screen  
The very first pop-up lets you skip videos or exit:

| Key | Action |
|-----|--------|
| **N** | Skip this video and move to the next one in the folder |
| **X** | Exit the entire program |

<p align="center">
  <img src="assets/first_window.png" width="480" alt="First window – skip or exit">
</p>

---

### 2 · Annotator GUI  
Press **any other key** (Space / Enter) to jump into the annotator loop.
| Key | Action |
|-----|--------|
| **m/M** | Toggle between objects (TOOL, HAND, DEVICE) |
| **d/D** | Toggle between bounding box or point prompts |
| **Enter/Space** | SAM2 starts with these prompts to propagate over the next **50 frames** |

<p align="center">
  <img src="assets/annotator.gif" width="640" alt="Live annotation demo">
</p>

---
### 3 · Preview & validation  
The tool replays annotated frames for acceptance or refinement:

| Key | Action |
|-----|--------|
| **e** | Edit BB / points on the current frame |
| **n** | Change the class label |
| **q** | Quit current video (saves progress) |
| **x** | Exit the program (saves progress) |

<p align="center">
  <img src="assets/annotator_res.png" width="640" alt="Preview window">
</p>

<p align="center">
  <img src="assets/edit_bb.gif"      alt="Edit bounding-box" width="45%" />
  <img src="assets/change_class.gif" alt="Change class label" width="45%" />
</p>

The loop repeats until the last frame is processed. Accepted annotations are written to `results_coco_format/` in COCO format.

## Configuration
Customize behavior in `settings.json`:
- `auto_accept`: Set to `true` for high automation (YOLO/SAM results accepted automatically).
- `shortcuts`: Rebind any key to fit your workflow.
- `output_dir`: Change where raw annotation results are stored.

## Citations
If you use this tool, please cite our work:
```bibtex
@article{maintenance_datasets,
  title={Maintenance Datasets: A Benchmark for Object Detection and Segmentation},
  author={...},
  journal={arXiv preprint arXiv:xxx},
  year={2025}
}
```
<!-- VERIFY: Actual citation details from ArXiv -->

## License
License information not found. See [MMTL Project](https://github.com/AI-Computer-Vision-BGU/MMTL) for parent project licensing.
