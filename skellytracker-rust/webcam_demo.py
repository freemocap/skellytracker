"""Demo — WebcamDemoViewer with Rust/Python hot-swappable backends.

Usage:
    python webcam_demo.py          # Rust backend (default)
    python webcam_demo.py --python # Python backend

Tracker hotkeys:
    b  — BrightestPoint     c  — Charuco
    r  — RTMpose            m  — MediaPipe
    g  — CompositeGPU

Controls:
    p  — toggle Rust / Python backend for current tracker
    h  — show/hide controls    i  — toggle info overlay
    space — pause              q / ESC — quit
"""

import sys
from pathlib import Path
from skellytracker.trackers.brightest_point_tracker.rust_bridge import (
    get_brightest_point_tracker,
)
import skellytracker.trackers.brightest_point_tracker.rust_bridge as bridge
# Make the parent skellytracker package importable from this directory.
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))



if __name__ == "__main__":
    use_python = "--python" in sys.argv
    bridge.USE_RUST_BACKEND = not use_python

    tracker = get_brightest_point_tracker(num_points=3, luminance_threshold=200)
    backend = "Rust" if not use_python else "Python"
    print("Hotkeys: b=switch tracker  r=toggle Rust/Python  h=controls  i=info  q=quit")

    from skellytracker.io.demo_viewers.webcam_demo_viewer import WebcamDemoViewer

    viewer = WebcamDemoViewer(tracker=tracker)
    viewer.use_rust_backend = not use_python
    print(f"Starting webcam demo — backend: {backend}, tracker: {tracker.__class__.__name__}")
    viewer.run()
    print("Done!")
