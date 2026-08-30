from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.metadata import EmptyMetadata
from skellytracker.core.sessions.session import Session
from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_point_names
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import get_hand_model_path
from skellytracker.core.sessions.mediapipe_session import MediaPipeSession

_HAND_POINT_NAMES: tuple[str, ...] = load_point_names(Path(__file__).parent / "mediapipe_hand.yaml")
_NUM_HAND_LANDMARKS = len(_HAND_POINT_NAMES)

# TEMPORARY DEBUG: print crop-box vertices (original-image coords) for the first
# N frames so boxes can be drawn on frames for inspection. Set to None to disable.
_CROP_DEBUG_MAX_FRAMES: int | None = 1000

# Selects which crop strategy is used when parent (body) keypoints are available:
#   - "palm":       palm-aligned, hand-span-sized rotated crop (current behavior)
#   - "elbow_wrist": axis-aligned square crop centered on the wrist, sized from the
#                   forearm length (previous behavior)
# Flip this to switch between the two methods; both print the same debug box
# vertices (see _CROP_DEBUG_MAX_FRAMES).
_CROP_METHOD: Literal["elbow_wrist", "palm"] = "palm"

_RIGHT_NAMES: tuple[str, ...] = tuple(f"right_hand_{n}" for n in _HAND_POINT_NAMES)
_LEFT_NAMES: tuple[str, ...] = tuple(f"left_hand_{n}" for n in _HAND_POINT_NAMES)
_BOTH_HAND_NAMES: tuple[str, ...] = _RIGHT_NAMES + _LEFT_NAMES


def crop_hand_roi(
    image: NDArray[np.uint8],
    wrist_px: NDArray[np.float64],
    index_mcp_px: NDArray[np.float64] | None = None,
    pinky_mcp_px: NDArray[np.float64] | None = None,
    elbow_px: NDArray[np.float64] | None = None,
    margin: float = 2.0,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Dispatch to the active crop strategy selected by ``_CROP_METHOD``.

    Both strategies return ``(crop, inverse_affine)`` where ``inverse_affine`` is a
    (2, 3) affine matrix mapping crop-local pixel coordinates back to original-image
    coordinates, so downstream code (landmark mapping and debug box vertices) is
    identical regardless of which method is active.

    - ``_CROP_METHOD == "elbow_wrist"`` uses ``elbow_px`` (axis-aligned forearm box).
    - ``_CROP_METHOD == "palm"`` uses ``index_mcp_px``/``pinky_mcp_px``.
    """
    if _CROP_METHOD == "elbow_wrist":
        if elbow_px is None:
            raise ValueError("elbow_wrist crop method requires elbow_px")
        return _crop_hand_roi_elbow_wrist(image, wrist_px, elbow_px, margin=margin)
    if index_mcp_px is None or pinky_mcp_px is None:
        raise ValueError("palm crop method requires index_mcp_px and pinky_mcp_px")
    return _crop_hand_roi_palm(
        image, wrist_px, index_mcp_px, pinky_mcp_px, margin=margin
    )


def _crop_hand_roi_elbow_wrist(
    image: NDArray[np.uint8],
    wrist_px: NDArray[np.float64],
    elbow_px: NDArray[np.float64],
    margin: float = 1.5,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Derive an axis-aligned square crop around the hand from wrist and elbow.

    Sizes the box from the 2D forearm length (wrist→elbow). This collapses when the
    arm points toward the camera (forearm foreshortens), which is exactly the failure
    mode the palm-based strategy was designed to fix.

    Returns:
        (crop, inverse_affine): the axis-aligned crop and a (2, 3) affine matrix
        mapping crop-local pixel coordinates back to original-image coordinates.
    """
    h, w = image.shape[:2]
    forearm_len = np.linalg.norm(wrist_px - elbow_px)
    box_size = int(forearm_len * margin)

    x_min = max(0, int(wrist_px[0] - box_size // 2))
    y_min = max(0, int(wrist_px[1] - box_size // 2))
    x_max = min(w, int(wrist_px[0] + box_size // 2))
    y_max = min(h, int(wrist_px[1] + box_size // 2))

    crop = image[y_min:y_max, x_min:x_max]
    inv_affine = np.array(
        [[1.0, 0.0, float(x_min)], [0.0, 1.0, float(y_min)]],
        dtype=np.float64,
    )
    return crop, inv_affine


def _crop_hand_roi_palm(
    image: NDArray[np.uint8],
    wrist_px: NDArray[np.float64],
    index_mcp_px: NDArray[np.float64],
    pinky_mcp_px: NDArray[np.float64],
    margin: float = 2.0,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Derive a palm-aligned, hand-span-sized crop around a hand.

    Mirrors MediaPipe Holistic's hand ROI (``GetHandRectFromPoseLandmarks``):
    the box is centered on the wrist, oriented to the palm direction (wrist →
    midpoint of the index & pinky MCPs), and sized from the hand's own span
    (wrist → index/pinky MCP) rather than from the forearm. The forearm-based
    size collapses when the arm points toward the camera (2D wrist→elbow length
    foreshortens), which is why the previous axis-aligned forearm box missed
    exactly those frames.

    Returns:
        (crop, inverse_affine): the rotated crop and a (2, 3) affine matrix
        mapping crop-local pixel coordinates back to original-image coordinates.
    """
    import cv2
    import math

    h, w = image.shape[:2]
    cx, cy = float(wrist_px[0]), float(wrist_px[1])

    # Palm forward direction (toward the fingers).
    fx = (float(index_mcp_px[0]) + float(pinky_mcp_px[0])) / 2.0 - cx
    fy = (float(index_mcp_px[1]) + float(pinky_mcp_px[1])) / 2.0 - cy
    rotation = math.atan2(fy, fx)

    # Hand span: wrist -> MCP distance (use the larger of index/pinky for safety).
    span = max(
        math.hypot(float(index_mcp_px[0]) - cx, float(index_mcp_px[1]) - cy),
        math.hypot(float(pinky_mcp_px[0]) - cx, float(pinky_mcp_px[1]) - cy),
    )
    box_size = max(int(span * 2.0 * margin), 16)

    # Rotate the image so the palm direction becomes horizontal (+x). cv2's
    # rotation angle is counter-clockwise in image coordinates, so negate the
    # measured palm angle.
    M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(rotation), 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Axis-aligned box around the wrist in the rotated image.
    x_min = max(0, int(cx - box_size // 2))
    y_min = max(0, int(cy - box_size // 2))
    x_max = min(w, x_min + box_size)
    y_max = min(h, y_min + box_size)
    crop = rotated[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return crop, np.eye(2, 3, dtype=np.float64)

    # Inverse affine: crop-local -> original image.
    # forward:  p_rot = M_lin @ p_img + M_t ; p_crop = p_rot - (x_min, y_min)
    # inverse:  p_img = M_lin^-1 @ p_crop + M_lin^-1 @ ((x_min,y_min) - M_t)
    invM = cv2.invertAffineTransform(M)  # M_lin^-1, and invM_t = -M_lin^-1 @ M_t
    off = np.array([float(x_min), float(y_min)])
    inv_affine = np.zeros((2, 3), dtype=np.float64)
    inv_affine[:, :2] = invM[:, :2]
    inv_affine[:, 2] = invM[:, :2] @ off + invM[:, 2]
    return crop, inv_affine


class MediapipeHandDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["mediapipe_hand"] = "mediapipe_hand"
    session_backend: Literal["mediapipe"] = "mediapipe"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5


@dataclass
class MediapipeHandKeypointDetector(KeypointDetector):
    """Detects hand landmarks for both hands using MediaPipe HandLandmarker.

    Returns 42 named keypoints: 21 right-hand points (prefixed right_hand_)
    followed by 21 left-hand points (prefixed left_hand_). Points for an
    undetected hand have NaN coordinates and 0.0 visibility.
    """

    config: MediapipeHandDetectorConfig
    session: MediaPipeSession
    landmarker: Any = field(repr=False)
    _point_names: tuple[str, ...] = field(default_factory=lambda: _BOTH_HAND_NAMES, init=False, repr=False)

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], EmptyMetadata]:
        """Convert BGR image to RGB for MediaPipe.

        The batch path uses detect() via thread pool rather than preprocess/postprocess
        directly, but the contract must be satisfied.
        """
        rgb = _to_rgb(image)
        return rgb, EmptyMetadata()

    def postprocess(self, raw: Any, metadata: EmptyMetadata) -> Keypoints:
        """Extract hand landmarks from a MediaPipe HandLandmarkerResult.

        raw should be a (result, h, w) tuple when called from the split-path.
        """
        if isinstance(raw, tuple):
            result, h, w = raw
        else:
            return Keypoints.empty(self._point_names)

        right_xyz = np.full((_NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float64)
        left_xyz = np.full((_NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float64)
        right_vis = np.zeros(_NUM_HAND_LANDMARKS, dtype=np.float64)
        left_vis = np.zeros(_NUM_HAND_LANDMARKS, dtype=np.float64)

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i]
            label = handedness[0].category_name

            xyz = np.array(
                [(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks],
                dtype=np.float64,
            )
            vis = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks],
                dtype=np.float64,
            )

            if label == "Right":
                right_xyz = xyz
                right_vis = vis
            elif label == "Left":
                left_xyz = xyz
                left_vis = vis

        xyz = np.concatenate([right_xyz, left_xyz], axis=0)
        visibility = np.concatenate([right_vis, left_vis], axis=0)
        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        right_xyz = np.full((_NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float64)
        left_xyz = np.full((_NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float64)
        right_vis = np.zeros(_NUM_HAND_LANDMARKS, dtype=np.float64)
        left_vis = np.zeros(_NUM_HAND_LANDMARKS, dtype=np.float64)

        if context is not None and getattr(context, "parent_keypoints", None) is not None and context.parent_keypoints.n_valid > 0:
            parent_kpts = context.parent_keypoints
            # Process each hand independently using the landmarks the active crop
            # method needs: elbow for "elbow_wrist", index+pinky MCP for "palm".
            if _CROP_METHOD == "elbow_wrist":
                hand_specs = (
                    ("left", "left_wrist", "left_elbow", None, None),
                    ("right", "right_wrist", "right_elbow", None, None),
                )
            else:
                hand_specs = (
                    ("left", "left_wrist", None, "left_index", "left_pinky"),
                    ("right", "right_wrist", None, "right_index", "right_pinky"),
                )
            for side, wrist_name, elbow_name, index_name, pinky_name in hand_specs:
                required = [wrist_name]
                if _CROP_METHOD == "elbow_wrist":
                    required.append(elbow_name)
                else:
                    required.extend([index_name, pinky_name])
                if not all(parent_kpts.has_name(name) for name in required):
                    continue
                wrist_xyz = parent_kpts.xyz_by_name(wrist_name)
                if np.isnan(wrist_xyz[0]):
                    continue

                if _CROP_METHOD == "elbow_wrist":
                    elbow_xyz = parent_kpts.xyz_by_name(elbow_name)
                    if np.isnan(elbow_xyz[0]):
                        continue
                    crop, inv_affine = crop_hand_roi(
                        image, wrist_xyz[:2], elbow_px=elbow_xyz[:2], margin=2.5
                    )
                else:
                    index_xyz = parent_kpts.xyz_by_name(index_name)
                    pinky_xyz = parent_kpts.xyz_by_name(pinky_name)
                    if np.isnan(index_xyz[0]) or np.isnan(pinky_xyz[0]):
                        continue
                    crop, inv_affine = crop_hand_roi(
                        image, wrist_xyz[:2], index_xyz[:2], pinky_xyz[:2], margin=3.0
                    )
                crop_h, crop_w = crop.shape[:2]
                if crop_h <= 0 or crop_w <= 0:
                    continue

                # TEMPORARY DEBUG: print crop-box vertices (original-image coords)
                # for the first N frames so the box can be drawn for inspection.
                if _CROP_DEBUG_MAX_FRAMES is not None:
                    frame_num = context.frame_number if context is not None else -1
                    if frame_num < _CROP_DEBUG_MAX_FRAMES:
                        corners_local = np.array(
                            [[0.0, 0.0], [crop_w, 0.0], [crop_w, crop_h], [0.0, crop_h]],
                            dtype=np.float64,
                        )
                        corners_img = corners_local @ inv_affine[:, :2].T + inv_affine[:, 2]
                        print(
                            f"[CROP-DEBUG] frame={frame_num} side={side} "
                            f"wrist=({wrist_xyz[0]:.1f},{wrist_xyz[1]:.1f}) "
                            f"crop={crop_w}x{crop_h} "
                            f"box_vertices_img={[tuple(round(v, 1) for v in c) for c in corners_img]}",
                            flush=True,
                        )

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(crop))
                result = self.landmarker.detect(mp_image)
                if not result.hand_landmarks:
                    continue
                # Take the single best-scored hand found in this ROI and assign it
                # to the side we seeded, regardless of MediaPipe's handedness label
                # (which is unreliable on a tight single-hand crop).
                hand_landmarks = max(
                    result.hand_landmarks,
                    key=lambda lm_list: float(
                        np.mean([lm.presence if lm.presence is not None else 1.0 for lm in lm_list])
                    ),
                )
                # Map crop-local pixel coordinates back to full-frame image
                # coordinates through the inverse affine of the rotated crop.
                crop_pts = np.array(
                    [[lm.x * crop_w, lm.y * crop_h] for lm in hand_landmarks],
                    dtype=np.float64,
                )
                full_pts = crop_pts @ inv_affine[:, :2].T + inv_affine[:, 2]
                z = np.array([lm.z * crop_w for lm in hand_landmarks], dtype=np.float64)
                xyz = np.column_stack([full_pts, z])
                vis = np.array(
                    [lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks],
                    dtype=np.float64,
                )
                if side == "right":
                    right_xyz, right_vis = xyz, vis
                else:
                    left_xyz, left_vis = xyz, vis
        else:
            # Fallback to full image
            h, w = image.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(image))
            result = self.landmarker.detect(mp_image)
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[i]
                label = handedness[0].category_name
                xyz = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in hand_landmarks], dtype=np.float64)
                vis = np.array([lm.presence if lm.presence is not None else 1.0 for lm in hand_landmarks], dtype=np.float64)
                if label == "Right":
                    right_xyz, right_vis = xyz, vis
                elif label == "Left":
                    left_xyz, left_vis = xyz, vis

        xyz = np.concatenate([right_xyz, left_xyz], axis=0)
        visibility = np.concatenate([right_vis, left_vis], axis=0)
        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def close(self) -> None:
        self.landmarker.close()

    def reset_temporal_state(self) -> None:
        self.landmarker.close()
        self.landmarker = type(self).create(self.config, self.session).landmarker

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_connections
        raw = load_connections(Path(__file__).parent / "mediapipe_hand.yaml")
        right = tuple((f"right_hand_{a}", f"right_hand_{b}") for a, b in raw)
        left = tuple((f"left_hand_{a}", f"left_hand_{b}") for a, b in raw)
        return right + left

    @classmethod
    def canonical_mapping_path(cls) -> Path:
        return Path(__file__).parent / "mediapipe_hand_to_canonical_mapping.yaml"

    @classmethod
    def create(
        cls,
        config: KeypointDetectorConfig,
        session: Session,
    ) -> MediapipeHandKeypointDetector:
        if not isinstance(session, MediaPipeSession):
            raise TypeError(f"Expected MediaPipeSession, got {type(session).__name__}")
        if not isinstance(config, MediapipeHandDetectorConfig):
            raise TypeError(f"Expected MediapipeHandDetectorConfig, got {type(config).__name__}")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        mp_running_mode = RunningMode.IMAGE
        hand_path = get_hand_model_path()
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            running_mode=mp_running_mode,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_hand_tracking_confidence,
        )
        landmarker = HandLandmarker.create_from_options(opts)
        return cls(config=config, session=session, landmarker=landmarker)


KEYPOINT_DETECTOR_REGISTRY["mediapipe_hand"] = MediapipeHandKeypointDetector


def _to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    import cv2
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_demo(config: MediapipeHandDetectorConfig | None = None, camera_index: int = 0) -> None:
    from skellytracker.core.config.detection_stage_config import DetectionStageConfig
    from skellytracker.core.config.tracker_config import TrackerConfig
    from skellytracker.core.io.demo_manager import DemoManager
    from skellytracker.core.keypoint_annotator import KeypointAnnotator, KeypointAnnotatorConfig, StageAnnotationSchema
    from skellytracker.core.tracker.tracker import Tracker
    if config is None:
        config = MediapipeHandDetectorConfig()
    session = MediaPipeSession.create()
    stage = DetectionStageConfig(name="hands", keypoint_detectors=[config])
    tracker = Tracker.create(TrackerConfig(stages=[stage]), {"mediapipe": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "hands": StageAnnotationSchema(connections=MediapipeHandKeypointDetector.connections()),
    }))
    DemoManager(tracker=tracker, annotator=annotator, window_title="MediaPipe Hand Demo").run_webcam(camera_index=camera_index)


if __name__ == "__main__":
    run_demo()
