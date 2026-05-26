"""
Shared ONNX Runtime DLL discovery for ``_skellytracker_rust`` imports.

The Rust native module links against ``onnxruntime.dll`` (via the ``ort``
crate's ``load-dynamic`` feature), which in turn lazy-loads CUDA/cuDNN
provider DLLs via ``LoadLibrary``. Those calls use the process ``PATH``, not
the directories registered by ``os.add_dll_directory``, so we must do both.

Every ``rust_bridge.py`` that imports ``_skellytracker_rust`` should call
``setup()`` at module level before the lazy import gate.
"""

import logging
import os
import platform
from typing import Any

logger = logging.getLogger(__name__)

_initialized: bool = False


def setup() -> None:
    """Ensure ONNX Runtime and NVIDIA CUDA/cuDNN DLLs are discoverable.

    Idempotent — safe to call from multiple bridges in the same process.
    Best-effort: warns but never raises on missing optional dependencies.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if platform.system() != "Windows":
        return

    _setup_ort_dll_directory()
    _setup_nvidia_dll_discovery()
    _preload_ort_dll()


# ── ONNX Runtime DLL directory (ort crate's load-dynamic needs onnxruntime.dll) ─

def _setup_ort_dll_directory() -> None:
    """Add the pip-installed ``onnxruntime/capi/`` directory to the DLL search path."""
    try:
        import onnxruntime
    except ImportError:
        logger.warning("onnxruntime not installed — Rust module may fail to load")
        return

    ort_bin = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
    if not os.path.isdir(ort_bin):
        return

    try:
        os.add_dll_directory(ort_bin)
    except OSError:
        pass

    current_path = os.environ.get("PATH", "")
    if ort_bin not in current_path:
        os.environ["PATH"] = f"{ort_bin};{current_path}"

    logger.debug("ORT DLL directory added: %s", ort_bin)


# ── NVIDIA CUDA/cuDNN DLL discovery ────────────────────────────────────────────

def _setup_nvidia_dll_discovery() -> None:
    """Prepend nvidia/*/bin dirs to PATH and pre-load cuDNN DLLs.

    ORT's CUDA EP lazy-loads cuDNN via ``LoadLibrary``, which searches the
    process ``PATH`` — not the per-directory set maintained by
    ``os.add_dll_directory``.  We therefore do both:
      1. ``os.add_dll_directory``  — Python-side ``ctypes`` / ``LoadLibraryEx``
      2. ``PATH`` prepend           — ORT's internal ``LoadLibrary`` calls
      3. Full-path pre-load         — ensures the correct cuDNN wins vs. any
         stale copy elsewhere on PATH
    """
    try:
        from importlib.util import find_spec
    except ImportError:
        return

    spec = find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        logger.debug("nvidia pip namespace not found — skipping CUDA DLL setup")
        return

    from pathlib import Path

    nvidia_root = Path(spec.submodule_search_locations[0])
    bin_dirs = sorted(nvidia_root.glob("*/bin"))
    if not bin_dirs:
        logger.debug("No nvidia/*/bin dirs found — skipping CUDA DLL setup")
        return

    bin_dir_strs = [str(d) for d in bin_dirs]
    os.environ["PATH"] = os.pathsep.join([*bin_dir_strs, os.environ.get("PATH", "")])

    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            pass

    # Pre-load cuDNN DLLs by absolute path so the correct copy wins
    import ctypes

    cudnn_bin = nvidia_root / "cudnn" / "bin"
    if cudnn_bin.is_dir():
        for dll_path in sorted(cudnn_bin.glob("*.dll")):
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError:
                pass


# ── Pre-load ORT DLL by absolute path ──────────────────────────────────────────

def _preload_ort_dll() -> None:
    """Load ``onnxruntime.dll`` by absolute path before the Rust module does.

    The ort crate's ``load-dynamic`` searches PATH for ``onnxruntime.dll``.
    If a CPU-only copy exists elsewhere on PATH, it will be loaded instead of
    the GPU one.  Pre-loading the correct DLL by absolute path ensures the
    right one is used.
    """
    try:
        import onnxruntime
    except ImportError:
        return

    from pathlib import Path

    ort_dir = Path(onnxruntime.__file__).parent
    ort_dll = ort_dir / "capi" / "onnxruntime.dll"

    if not ort_dll.exists():
        logger.error(
            "ORT DLL not found at expected path: %s",
            str(ort_dll),
        )
        return

    print(f"[skellytracker-rust bridge] Pre-loading ORT DLL: {ort_dll}")

    import ctypes

    ctypes.WinDLL(str(ort_dll))
    print("[skellytracker-rust bridge] ORT DLL loaded successfully")
