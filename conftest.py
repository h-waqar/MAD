"""
Root conftest.py — adds AnnotationScheme/ to sys.path so that
`import configs` (used by AnnotationScheme/utils/utils.py at module level)
resolves correctly when tests run from the project root.
"""
import sys
import os

# AnnotationScheme/ must be on sys.path because utils.py does `import configs`
# at module level, and configs.py lives inside AnnotationScheme/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AnnotationScheme"))
