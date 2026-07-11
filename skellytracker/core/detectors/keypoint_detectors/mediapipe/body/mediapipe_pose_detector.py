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
from skellytracker.core.sessions.session import Session
from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_point_names
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import (
    MediapipePoseModelComplexity,
    get_pose_model_path,
)
from skellytracker.core.sessions.mediapipe_session import MediaPipeSession

_POINT_NAMES: tuple[str, ...] = load_point_names(Path(__file__).parent / "mediapipe_body.yaml")
_NUM_LANDMARKS = len(_POINT_NAMES)


class MediapipePoseDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["mediapipe_pose"] = "mediapipe_pose"
    session_backend: Literal["mediapipe"] = "mediapipe"
    model_complexity: MediapipePoseModelComplexity = MediapipePoseModelComplexity.HEAVY
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5


@dataclass
class MediapipePoseKeypointDetector(KeypointDetector):
    """Detects body pose landmarks using MediaPipe PoseLandmarker.

    Returns 33 named keypoints in pixel space. Points not detected have NaN
    coordinates and 0.0 visibility.
    """

    config: MediapipePoseDetectorConfig
    session: MediaPipeSession
    landmarker: Any = field(repr=False)
    _point_names: tuple[str, ...] = field(default_factory=lambda: _POINT_NAMES, init=False, repr=False)

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

        if not result.pose_landmarks:
            return Keypoints.empty(self._point_names)

        landmarks = result.pose_landmarks[0]
        xyz = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks],
            dtype=np.float64,
        )
        visibility = np.array(
            [lm.visibility if lm.visibility is not None else 0.0 for lm in landmarks],
            dtype=np.float64,
        )
        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def close(self) -> None:
        self.landmarker.close()

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_connections
        return load_connections(Path(__file__).parent / "mediapipe_body.yaml")

    @classmethod
    def create(
        cls,
        config: KeypointDetectorConfig,
        session: Session,
    ) -> MediapipePoseKeypointDetector:
        if not isinstance(session, MediaPipeSession):
            raise TypeError(f"Expected MediaPipeSession, got {type(session).__name__}")
        if not isinstance(config, MediapipePoseDetectorConfig):
            raise TypeError(f"Expected MediapipePoseDetectorConfig, got {type(config).__name__}")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        mp_running_mode = RunningMode.VIDEO if session.running_mode == "video" else RunningMode.IMAGE
        pose_path = get_pose_model_path(config.model_complexity)
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)),
            running_mode=mp_running_mode,
            num_poses=config.num_poses,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_pose_tracking_confidence,
        )
        landmarker = PoseLandmarker.create_from_options(opts)
        return cls(config=config, session=session, landmarker=landmarker)


KEYPOINT_DETECTOR_REGISTRY["mediapipe_pose"] = MediapipePoseKeypointDetector


def _to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    import cv2
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_demo(config: MediapipePoseDetectorConfig | None = None, camera_index: int = 0) -> None:
    from skellytracker.core.config.detection_stage_config import DetectionStageConfig
    from skellytracker.core.config.tracker_config import TrackerConfig
    from skellytracker.core.demo_manager import DemoManager
    from skellytracker.core.keypoint_annotator import KeypointAnnotator, KeypointAnnotatorConfig, StageAnnotationSchema
    from skellytracker.core.tracker.tracker import Tracker
    if config is None:
        config = MediapipePoseDetectorConfig()
    session = MediaPipeSession.create()
    stage = DetectionStageConfig(name="body", keypoint_detectors=[config])
    tracker = Tracker.create(TrackerConfig(stages=[stage]), {"mediapipe": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "body": StageAnnotationSchema(connections=MediapipePoseKeypointDetector.connections()),
    }))
    DemoManager(tracker=tracker, annotator=annotator, window_title="MediaPipe Pose Demo").run_webcam(camera_index=camera_index)


if __name__ == "__main__":
    run_demo()
