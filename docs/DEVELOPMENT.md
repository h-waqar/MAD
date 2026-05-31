<!-- generated-by: gsd-doc-writer -->
# Development Guide

This document provides instructions for setting up the development environment, running tests, and contributing to the Annotation Scheme project.

## Local Setup

### 1. Repository Initialization
Clone the repository and its submodules (specifically `segmentanything`).
```bash
git clone --recursive https://github.com/AI-Computer-Vision-BGU/Annotation-Scheme.git
cd Annotation-Scheme
```

### 2. Environment Configuration
Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation
Install the core dependencies. Note that the requirements file is named `requirments.txt`.
```bash
pip install -r requirments.txt
```

If you are using a CUDA-enabled GPU (highly recommended for SAM2 performance), install the appropriate PyTorch version:
```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. SAM2 Specific Setup
Install the SAM2 package in editable mode and download the necessary model checkpoints:
```bash
cd segmentanything
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..
```

## Build and Run Commands

| Command | Description |
| :--- | :--- |
| `./run_annotation.sh` | Main entry point that runs the indexer and launches the annotation GUI (Linux). |
| `pytest` | Executes the test suite located in the `tests/` directory. |
| `python AnnotationScheme/utils/indexer.py` | Indexes source videos to ensure unique prefixes. |
| `python AnnotationScheme/annotation_scheme.py` | Launches the annotator directly. Supports various flags like `--weights` and `--new_shape`. |
| `python AnnotationScheme/review_annotations.py` | Utility script for reviewing existing annotations. |

## Testing

The project uses `pytest` for testing. Configuration is handled via `conftest.py`.

### Running Tests
Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/test_db_lookup.py
```

### Test Structure
- `tests/conftest.py`: Shared fixtures and configuration.
- `tests/test_*.py`: Test modules.

## Code Style

This project follows standard Python coding conventions.

- **Style Guide:** [PEP 8](https://peps.python.org/pep-0008/) is the primary reference.
- **Linting:** No automated linter (like Flake8 or Ruff) is currently enforced in CI, but developers are encouraged to use them locally.
- **Formatting:** Standard Python formatting is expected.

## PR Process

1. **Fork the repository** and create a feature branch.
2. **Implement changes** and ensure new features are covered by tests.
3. **Run existing tests** using `pytest` to ensure no regressions.
4. **Submit a Pull Request** with a clear description of the changes and the problem they solve.
5. **Code Review:** Wait for a maintainer to review your changes. Address any feedback provided.

## Branch Conventions

No strict branch naming convention is currently documented. Descriptive names like `feat/new-feature-name` or `fix/issue-description` are recommended.
