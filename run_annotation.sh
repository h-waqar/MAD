#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/venv/bin/activate"
cd "$PROJECT_DIR"

# Fix Qt Wayland/Theme warnings by forcing XCB and standard style
export QT_QPA_PLATFORM=xcb
export QT_STYLE_OVERRIDE=Fusion
export QT_LOGGING_RULES='qt.qpa.plugin=false'

# Run video indexer to ensure all new source files have unique prefixes
python3 AnnotationScheme/utils/indexer.py

python AnnotationScheme/annotation_scheme.py \
  --new_shape 680 340 \
  --weights l
