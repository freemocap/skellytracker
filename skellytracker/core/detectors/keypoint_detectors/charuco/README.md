# Charuco Detector

Detects a [Charuco board](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html) in images using OpenCV. Unlike the other keypoint detectors in this package, charuco detection is used for **camera calibration**, not human pose estimation.

## What it returns

`CharucoDetector.detect()` returns a `Keypoints` object containing:

- **`CharucoCorner-{id}`** — one entry per internal corner on the board (e.g. a 5×3 board has 4×2 = 8 corners, IDs 0–7). Detected corners have their (x, y) image coordinates; undetected corners are `NaN` with `visibility=0`.
- **`ArucoMarkerCorner-{id}-{j}`** — four corners per ArUco marker (j = 0–3), in the same NaN/visibility pattern.

The number of points is fixed per board regardless of how many corners were actually detected in a given frame.

## Multi-camera calibration workflow (freemocap)

Multi-camera calibration estimates the 3D pose of each camera relative to a shared world frame. The Charuco board is the shared reference: it has a known geometry, and its corners can be detected from multiple viewpoints simultaneously.

### Step 1 — Set up one detector per camera

All cameras must use the **same board definition**.

```python
from skellytracker.core.detectors.keypoint_detectors.charuco import (
    CharucoBoardDefinition,
    CharucoDetector,
    CharucoDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig

board_def = CharucoBoardDefinition(squares_x=5, squares_y=3, square_length_mm=54.0)
config = CharucoDetectorConfig(board=board_def)
session = CpuSession.create(CpuSessionConfig())

detectors = {
    camera_id: CharucoDetector.create(config, session)
    for camera_id in camera_ids
}
```

### Step 2 — Collect 2D corner observations across frames

For each synchronized frame, run detection on every camera. The returned `Keypoints.xy` array gives the 2D image coordinates of each corner (indexed by corner ID), with `NaN` where a corner wasn't visible.

```python
observations: dict[str, list[Keypoints]] = {cam: [] for cam in camera_ids}

for frame_images in synchronized_frame_stream:
    for camera_id, image in frame_images.items():
        kpts = detectors[camera_id].detect(image)
        observations[camera_id].append(kpts)
```

The `Keypoints` object provides several ways to access the 2D data:

```python
# Full (N, 2) array — NaN for undetected corners
kpts.xy

# Check which corners were actually seen
detected_ids = [i for i in range(board_def.n_corners) if kpts.visibility[i] > 0]

# Coordinate for a specific corner
kpts.xy_by_name("CharucoCorner-3")  # (2,) array
```

### Step 3 — Estimate board pose per camera per frame

Once you have initial camera intrinsics (from a prior intrinsic calibration step), use `compute_board_pose()` to find where the board was in each camera's coordinate frame for each frame. This is the key output used for extrinsic calibration.

```python
from skellytracker.core.detectors.keypoint_detectors.charuco import compute_board_pose

# camera_matrix: (3, 3) intrinsic matrix
# dist_coeffs: (4,) or (5,) distortion coefficients
result = compute_board_pose(kpts, board_def, camera_matrix, dist_coeffs)

if result is not None:
    rvec, tvec = result
    # rvec: (3,) Rodrigues rotation vector — board orientation in camera frame
    # tvec: (3,) translation vector     — board position in camera frame
```

Returns `None` when fewer than 6 corners were detected (not enough for a stable pose estimate).

### Step 4 — Transform corners to camera coordinates (optional)

If you need the 3D position of each detected corner in the camera's coordinate frame (e.g. to verify reprojection or pass to a solver):

```python
from skellytracker.core.detectors.keypoint_detectors.charuco import transform_to_camera_coordinates

# board_frame_points: (N, 3) — from board_def.corner_positions_board_frame
corners_camera = transform_to_camera_coordinates(
    board_def.corner_positions_board_frame, rvec, tvec
)  # → (N, 3)
```

### Step 5 — Export to anipose format (optional)

If your calibration solver uses the [anipose](https://anipose.readthedocs.io/) calibration pipeline, convert each observation to its expected dict format:

```python
from skellytracker.core.detectors.keypoint_detectors.charuco import to_anipose_camera_row

row = to_anipose_camera_row(kpts, board_def, frame_number=42)
# {
#   "framenum": (0, 42),
#   "corners":  (n_detected, 1, 2)  — image coords of detected corners only
#   "ids":      (n_detected,)       — corner IDs that were detected
#   "filled":   (n_corners, 1, 2)   — full array, NaN where undetected
# }
```

## Minimum viable detection

Pose estimation requires **at least 6 detected corners**. For reliable calibration across frames, aim for boards that are large enough to show 8+ corners at the typical working distance, and collect frames from a range of angles.
