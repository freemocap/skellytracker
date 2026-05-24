"""Smoke test / demo for BrightestPointTracker (Rust or Python backend).

Usage:
    python test_demo.py                  # Rust backend (default)
    python test_demo.py --python         # Python fallback backend
    python test_demo.py --image <path>   # test on a single image file
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure the parent skellytracker package is importable.
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from skellytracker.trackers.brightest_point_tracker.rust_bridge import (
    get_brightest_point_tracker,
    RustBrightestPointTracker,
)


def test_import(use_python: bool):
    """Instantiate the tracker via the hot-swappable factory."""
    if use_python:
        from skellytracker.trackers.brightest_point_tracker.rust_bridge import USE_RUST_BACKEND
        import skellytracker.trackers.brightest_point_tracker.rust_bridge as bridge
        bridge.USE_RUST_BACKEND = False
        tracker = get_brightest_point_tracker(num_points=3, luminance_threshold=200)
        bridge.USE_RUST_BACKEND = True  # restore
    else:
        tracker = get_brightest_point_tracker(num_points=3, luminance_threshold=200)

    backend = type(tracker).__name__
    print(f"[OK] Backend: {backend} | {tracker}")
    return tracker


def test_on_image(tracker, image_path: str):
    """Run detection on a single image and display the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[FAIL] Could not read image: {image_path}")
        return

    print(f"Image shape: {image.shape}")

    result = tracker.process_image(frame_number=0, image=image)
    print(f"[OK] Detection result: {result}")

    annotated = tracker.annotate_image(image=image, observation=result)
    print(f"[OK] Annotated image shape: {annotated.shape}")

    stacked = np.hstack([image, annotated])
    cv2.imshow("Original (left) | Annotated (right) — press any key to close", stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_webcam(tracker):
    """Run the tracker on a live webcam feed."""
    backend = type(tracker).__name__
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Could not open webcam")
        return

    print(f"Webcam demo ({backend}) — press 'q' to quit")

    frame_count = 0
    fps_update_interval = 30
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            result = tracker.process_image(frame_number=frame_count, image=frame)
            annotated = tracker.annotate_image(image=frame, observation=result)
        except Exception as e:
            print(f"[ERROR] {e}")
            annotated = frame

        if frame_count % fps_update_interval == 0:
            elapsed = time.time() - start_time
            fps = fps_update_interval / elapsed if elapsed > 0 else 0
            start_time = time.time()

        cv2.putText(
            annotated, f"FPS: {fps:.1f}  Backend: {backend}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow(f"BrightestPointTracker — {backend}", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Test BrightestPointTracker (Rust/Python)")
    parser.add_argument("--image", type=str, help="Path to a single test image")
    parser.add_argument("--num-points", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--python", action="store_true", help="Use Python backend instead of Rust")
    args = parser.parse_args()

    tracker = test_import(use_python=args.python)

    if args.image:
        test_on_image(tracker, args.image)
    else:
        demo_webcam(tracker)


if __name__ == "__main__":
    main()
