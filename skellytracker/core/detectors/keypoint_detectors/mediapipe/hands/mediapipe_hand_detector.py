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
_NUM_HAND_KEYPOINTS = len(_HAND_POINT_NAMES)

_RIGHT_NAMES: tuple[str, ...] = tuple(f"right_hand_{n}" for n in _HAND_POINT_NAMES)
_LEFT_NAMES: tuple[str, ...] = tuple(f"left_hand_{n}" for n in _HAND_POINT_NAMES)
_BOTH_HAND_NAMES: tuple[str, ...] = _RIGHT_NAMES + _LEFT_NAMES


class MediapipeHandDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["mediapipe_hand"] = "mediapipe_hand"
    session_backend: Literal["mediapipe"] = "mediapipe"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5


@dataclass
class MediapipeHandKeypointDetector(KeypointDetector):
    """Detects hand keypoints for both hands using MediaPipe HandLandmarker.

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
        """Extract hand keypoints from a MediaPipe HandLandmarkerResult.

        raw should be a (result, h, w) tuple when called from the split-path.
        """
        if isinstance(raw, tuple):
            result, h, w = raw
        else:
            return Keypoints.empty(self._point_names)

        right_xyz = np.full((_NUM_HAND_KEYPOINTS, 3), np.nan, dtype=np.float64)
        left_xyz = np.full((_NUM_HAND_KEYPOINTS, 3), np.nan, dtype=np.float64)
        right_vis = np.zeros(_NUM_HAND_KEYPOINTS, dtype=np.float64)
        left_vis = np.zeros(_NUM_HAND_KEYPOINTS, dtype=np.float64)

        for i, lms in enumerate(result.hand_landmarks):
            handedness = result.handedness[i]
            label = handedness[0].category_name

            xyz = np.array(
                [(lm.x * w, lm.y * h, lm.z * w) for lm in lms],
                dtype=np.float64,
            )
            vis = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in lms],
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
        h, w = image.shape[:2]
        rgb = _to_rgb(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.session.running_mode == "video":
            ts = (
                context.timestamp_ms
                if (context is not None and context.timestamp_ms is not None)
                else int(time.monotonic() * 1000)
            )
            result = self.landmarker.detect_for_video(mp_image, ts)
        else:
            result = self.landmarker.detect(mp_image)

        right_xyz = np.full((_NUM_HAND_KEYPOINTS, 3), np.nan, dtype=np.float64)
        left_xyz = np.full((_NUM_HAND_KEYPOINTS, 3), np.nan, dtype=np.float64)
        right_vis = np.zeros(_NUM_HAND_KEYPOINTS, dtype=np.float64)
        left_vis = np.zeros(_NUM_HAND_KEYPOINTS, dtype=np.float64)

        for i, lms in enumerate(result.hand_landmarks):
            handedness = result.handedness[i]
            label = handedness[0].category_name

            xyz = np.array(
                [(lm.x * w, lm.y * h, lm.z * w) for lm in lms],
                dtype=np.float64,
            )
            vis = np.array(
                [lm.presence if lm.presence is not None else 1.0 for lm in lms],
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
    def standard_human_mapping_path(cls) -> Path:
        return Path(__file__).parent / "mediapipe_hand_to_standard_human_mapping.yaml"

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

        mp_running_mode = RunningMode.VIDEO if session.running_mode == "video" else RunningMode.IMAGE
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
