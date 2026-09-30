"""Robust module loader for Colab + Google Drive.

Use this instead of `sys.path.append(...) + import src.X` when Drive's FUSE
mount is being flaky about newly-updated files. It loads each module directly
from its file path and manually registers the "src" package in sys.modules,
so internal "from src.X import Y" statements inside the modules still work.
"""

import importlib.util
import os
import sys
import time

PROJECT_ROOT = "/content/drive/MyDrive/iaaa-brain-ct-triage"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def _wait_for_path(path: str, timeout_s: float = 10.0, interval_s: float = 0.5) -> None:
    """Poll until a path becomes visible through the Drive FUSE mount, or raise."""
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_s:
            raise FileNotFoundError(
                f"Path did not become visible within {timeout_s}s: {path}"
            )
        time.sleep(interval_s)


def _load_module_from_path(module_name: str, file_path: str):
    """Load a single module from an explicit file path and register it in sys.modules."""
    _wait_for_path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register BEFORE exec, so internal imports resolve
    spec.loader.exec_module(module)
    return module


def load_src_package(project_root: str = PROJECT_ROOT):
    """Load the src package (preprocessing, augmentation, dataset) directly from disk.

    Returns:
        A tuple (preprocessing, augmentation, dataset) of the loaded modules.
    """
    src_dir = os.path.join(project_root, "src")
    _wait_for_path(src_dir)

    # Register the "src" package itself first (as a namespace package), so that
    # "from src.augmentation import ..." inside dataset.py resolves correctly.
    src_init_path = os.path.join(src_dir, "__init__.py")
    _load_module_from_path("src", src_init_path)

    preprocessing = _load_module_from_path("src.preprocessing", os.path.join(src_dir, "preprocessing.py"))
    augmentation = _load_module_from_path("src.augmentation", os.path.join(src_dir, "augmentation.py"))
    dataset = _load_module_from_path("src.dataset", os.path.join(src_dir, "dataset.py"))

    return preprocessing, augmentation, dataset


if __name__ == "__main__":
    preprocessing, augmentation, dataset = load_src_package()
    print("Loaded from:", dataset.__file__)

    import inspect
    print(inspect.signature(dataset.BrainCTSeriesDataset.__init__))
