"""
Shared OpenCV DLL setup. With vcpkg x64-windows-static, OpenCV is statically
linked — no DLL discovery needed. This module exists as a single import point
so we don't duplicate the setup call in every rust_bridge module.
"""
import logging

logger = logging.getLogger(__name__)

_initialized = False


def setup() -> None:
    """No-op: OpenCV is statically linked via vcpkg x64-windows-static."""
    global _initialized
    _initialized = True
