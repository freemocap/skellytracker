"""Diagnostic: draw each face keypoint index on an image, coloured by assumed region.

Use a close-up face photo where the face fills most of the frame — the model
is a top-down estimator designed for face crops, not full-body images.

Each landmark group is drawn in a distinct colour so you can immediately see
whether the index ranges land on the right facial features:

  red    0-32    face contour
  green  33-42   right eyebrow
  cyan   43-50   left eyebrow
  yellow 51-55   nose bridge
  orange 56-65   nose bottom / nostrils
  purple 66-74   right eye
  white  75-83   left eye
  pink   84-95   outer lip
  magenta 96-103 inner lip
  lime   104-105 pupils

If a colour group appears on the wrong feature, update rtmpose_face.yaml.

Usage::

    uv run python -m skellytracker.core.detectors.rtmpose.face.probe_face_indices path/to/face.jpg
    uv run python -m skellytracker.core.detectors.rtmpose.face.probe_face_indices path/to/face.jpg --threshold 0.001
    uv run python -m skellytracker.core.detectors.rtmpose.face.probe_face_indices path/to/face.jpg --save out.jpg

Press any key to close the window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from skellytracker.core.detectors.rtmpose.face.rtmpose_face_detector import RTMPoseFaceDetector
from skellytracker.core.detectors.rtmpose.rtmpose_preprocessing import (
    rtmpose_letterbox_postprocess,
    rtmpose_letterbox_preprocess,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig

_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)

# (label, start_index, end_index_inclusive, BGR colour)
_REGIONS: list[tuple[str, int, int, tuple[int, int, int]]] = [
    ("contour",        0,   32, (0,   0,   255)),  # red
    ("right_eyebrow", 33,   42, (0,   200,   0)),  # green
    ("left_eyebrow",  43,   50, (255, 200,   0)),  # cyan
    ("nose_bridge",   51,   55, (0,   255, 255)),  # yellow
    ("nose_bottom",   56,   65, (0,   140, 255)),  # orange
    ("right_eye",     66,   74, (200,   0, 200)),  # purple
    ("left_eye",      75,   83, (255, 255, 255)),  # white
    ("outer_lip",     84,   95, (180, 105, 255)),  # pink
    ("inner_lip",     96,  103, (255,   0, 255)),  # magenta
    ("pupils",       104,  105, (0,   255,  80)),  # lime
]


def _region_colour(idx: int) -> tuple[int, int, int]:
    for _, start, end, colour in _REGIONS:
        if start <= idx <= end:
            return colour
    return (128, 128, 128)


def annotate_image(image_path: Path, threshold: float = 0.004, save_path: Path | None = None) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    session = OnnxSession.create(OnnxSessionConfig(models=[RTMPoseFaceDetector.model_spec()]))
    ort = session.get_session("rtmpose-m_256x256")
    input_name = ort.get_inputs()[0].name

    h, w = frame.shape[:2]
    crop_bbox = np.array([0.0, 0.0, float(w), float(h)])
    resized, center, scale = rtmpose_letterbox_preprocess(
        frame, crop_bbox, (256, 256), mean=_MEAN, std=_STD
    )
    inp = np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32))
    sx, sy = ort.run(None, {input_name: inp})
    kpts, scores = rtmpose_letterbox_postprocess(sx, sy, center, scale, (256, 256), 2.0)
    kpts = kpts[0]
    scores = scores[0]

    n_detected = int((scores >= threshold).sum())
    print(f"Detected {n_detected}/106 keypoints above threshold {threshold}")
    if n_detected == 0:
        print("No keypoints detected — make sure the image is a close-up face crop.")

    annotated = frame.copy()
    for i, (xy, s) in enumerate(zip(kpts, scores)):
        if s < threshold:
            continue
        x, y = int(xy[0]), int(xy[1])
        colour = _region_colour(i)
        cv2.circle(annotated, (x, y), 3, colour, -1)
        cv2.putText(annotated, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_PLAIN, 1.4, colour, 1)

    # Draw legend
    legend_x = 8
    for i, (label, start, end, colour) in enumerate(_REGIONS):
        y_pos = 20 + i * 24
        cv2.rectangle(annotated, (legend_x, y_pos - 10), (legend_x + 12, y_pos + 2), colour, -1)
        cv2.putText(annotated, f"{start}-{end} {label}", (legend_x + 16, y_pos),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, colour, 1)

    print("\nDetected keypoint positions (index | region | x | y | score):")
    for i, (xy, s) in enumerate(zip(kpts, scores)):
        if s >= threshold:
            region = next((r[0] for r in _REGIONS if r[1] <= i <= r[2]), "?")
            print(f"  {i:3d} | {region:<15s} | {int(xy[0]):4d} | {int(xy[1]):4d} | {s:.4f}")

    if save_path:
        cv2.imwrite(str(save_path), annotated)
        print(f"\nSaved annotated image to {save_path}")

    cv2.imshow("Face keypoint indices — colour by assumed region (any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe RTMPose face keypoint indices on an image")
    parser.add_argument("image", type=Path, help="Path to a close-up face image")
    parser.add_argument("--threshold", type=float, default=0.004, help="Confidence threshold (default: 0.004)")
    parser.add_argument("--save", type=Path, default=None, help="Save annotated image to this path")
    args = parser.parse_args()

    annotate_image(args.image, threshold=args.threshold, save_path=args.save)


if __name__ == "__main__":
    main()
