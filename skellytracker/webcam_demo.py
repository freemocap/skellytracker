"""Demo — WebcamDemoViewer with Rust/Python hot-swappable backends.

Usage:
    python -m skellytracker.webcam_demo          # Rust backend (default)
    python -m skellytracker.webcam_demo --python # Python backend

Tracker hotkeys:
    b  — BrightestPoint     c  — Charuco
    r  — RTMpose            m  — MediaPipe
    g  — CompositeGPU

Controls:
    p  — toggle Rust / Python backend for current tracker
    h  — show/hide controls    i  — toggle info overlay
    space — pause              q / ESC — quit
"""

import logging
import sys

logger = logging.getLogger(__name__)

logger.debug("Importing rust_bridge module...")
from skellytracker.trackers.brightest_point_tracker.rust_bridge import (
    get_brightest_point_tracker,
)
import skellytracker.trackers.brightest_point_tracker.rust_bridge as bridge
logger.debug("bridge module loaded, USE_RUST_BACKEND=%s", bridge.USE_RUST_BACKEND)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(levelname)s: %(message)s")

    use_python = "--python" in sys.argv
    bridge.USE_RUST_BACKEND = not use_python
    logger.info("Backend: %s", "Python" if use_python else "Rust")

    logger.debug("Creating tracker via get_brightest_point_tracker(num_points=3, luminance_threshold=200)...")
    tracker = get_brightest_point_tracker(num_points=3, luminance_threshold=200)
    logger.debug("tracker created: %s", tracker)

    backend = "Rust" if not use_python else "Python"
    logger.info("Hotkeys: b=switch tracker  r=toggle Rust/Python  h=controls  i=info  q=quit")

    logger.debug("Importing WebcamDemoViewer...")
    from skellytracker.io.demo_viewers.webcam_demo_viewer import WebcamDemoViewer

    logger.debug("Creating WebcamDemoViewer...")
    viewer = WebcamDemoViewer(tracker=tracker)
    viewer.use_rust_backend = not use_python
    logger.debug("viewer created, tracker type: %s", type(viewer.tracker).__name__)

    logger.info("Starting webcam demo — backend: %s, tracker: %s", backend, tracker.__class__.__name__)
    logger.debug("ENTERING viewer.run()...")
    viewer.run()
    logger.info("Done!")
