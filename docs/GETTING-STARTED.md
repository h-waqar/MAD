<!-- generated-by: gsd-doc-writer -->
# Getting Started

This guide provides step-by-step instructions to set up the Annotation-Scheme tool on Linux and Windows.

## Prerequisites

Before installing, ensure your system meets the following requirements:

- **Python:** `3.10.11` (Recommended)
- **Git:** Required for cloning the repository and managing submodules.
- **Hardware:** A CUDA-capable NVIDIA GPU is highly recommended for SAM2 and YOLOv11 performance.
- **Drivers:** Latest NVIDIA drivers and CUDA Toolkit (e.g., CUDA 11.8 or 12.1).

## Installation

### 1. Clone the Repository
Clone the project along with its submodules (specifically the SAM2 repository):

```bash
git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
cd Annotation-Scheme
```

### 2. Set Up Virtual Environment

#### Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install the core dependencies and the appropriate version of PyTorch for your CUDA version.

```bash
# Install base requirements (note the spelling in the filename)
pip install -r requirments.txt

# Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Initialize SAM2 Submodule
The tool relies on the SAM2 (Segment Anything Model 2) repository located in the `segmentanything/` directory.

```bash
cd segmentanything
pip install -e .
cd checkpoints
# Linux/Git Bash:
./download_ckpts.sh
# Return to root
cd ../..
```

## First Run

### Linux
Use the provided shell script to launch the tool:
```bash
chmod +x run_annotation.sh
./run_annotation.sh
```

### Windows
Run the indexer followed by the main annotation script:
```powershell
python AnnotationScheme/utils/indexer.py
python AnnotationScheme/annotation_scheme.py --new_shape 680 340 --weights l
```

The tool will open a GUI where you can select the annotation mode (Single Video, Directory, Fixer, or Resume).

## Common Setup Issues

| Issue | Solution |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'sam2'` | Ensure you ran `pip install -e .` inside the `segmentanything/` directory. |
| `CUDA out of memory` | Use a smaller SAM2 model by passing `--weights t` (Tiny) instead of `l` (Large). |
| `FileNotFoundError: yolo11n-pose.pt` | Ensure the YOLO weights file is present in the project root. |
| `ImportError: libGL.so.1` (Linux) | Install OpenGL dependencies: `sudo apt-get install libgl1`. |
| Submodule folder is empty | Run `git submodule update --init --recursive` to pull the SAM2 source code. |

## Next Steps

Once you have the tool running, explore the following documentation to customize your workflow:

- [**Configuration Guide**](CONFIGURATION.md) — Learn how to customize shortcuts, paths, and automation settings in `settings.json`.
- [**Architecture Overview**](ARCHITECTURE.md) — Understand the internal pipeline between YOLOv11 and SAM2.
- [**User Manual**](../USER_MANUAL.md) — Detailed guide on keyboard shortcuts and annotation modes.
