from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.keypoint_annotator import KeypointAnnotator, KeypointAnnotatorConfig, StageAnnotationSchema
from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.sessions.session import Session
from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_point_names
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import get_face_model_path
from skellytracker.core.sessions.mediapipe_session import MediaPipeSession

_POINT_NAMES: tuple[str, ...] = load_point_names(Path(__file__).parent / "mediapipe_face_contour.yaml")

# FaceLandmarker returns 478 points (468 mesh + 10 iris). Extract contour subset
# by parsing the numeric index encoded in each name (e.g. "face_0033" → 33).
_CONTOUR_INDICES: NDArray[np.intp] = np.array(
    [int(name.split("_")[1]) for name in _POINT_NAMES],
    dtype=np.intp,
)


class MediapipeFaceDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["mediapipe_face"] = "mediapipe_face"
    session_backend: Literal["mediapipe"] = "mediapipe"
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5


@dataclass
class MediapipeFaceKeypointDetector(KeypointDetector):
    """Detects face contour landmarks using MediaPipe FaceLandmarker.

    Returns a named subset of the 478-point face tessellation corresponding to
    the contour defined in mediapipe_face_contour.yaml. Points not detected have
    NaN coordinates and 0.0 visibility.
    """

    config: MediapipeFaceDetectorConfig
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

        if not result.face_landmarks:
            return Keypoints.empty(self._point_names)

        landmarks = result.face_landmarks[0]
        n_full = len(landmarks)

        full_xyz = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks],
            dtype=np.float64,
        )
        full_vis = np.array(
            [lm.presence if lm.presence is not None else 1.0 for lm in landmarks],
            dtype=np.float64,
        )

        # Guard against models that return fewer than expected points
        valid_mask = _CONTOUR_INDICES < n_full
        xyz = np.full((len(self._point_names), 3), np.nan, dtype=np.float64)
        vis = np.zeros(len(self._point_names), dtype=np.float64)
        xyz[valid_mask] = full_xyz[_CONTOUR_INDICES[valid_mask]]
        vis[valid_mask] = full_vis[_CONTOUR_INDICES[valid_mask]]

        return Keypoints(names=self._point_names, xyz=xyz, visibility=vis)

    def close(self) -> None:
        self.landmarker.close()

    @classmethod
    def connections(cls) -> tuple[tuple[str, str], ...]:
        from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_connections
        return load_connections(Path(__file__).parent / "mediapipe_face_contour.yaml")

    @classmethod
    def create(
        cls,
        config: KeypointDetectorConfig,
        session: Session,
    ) -> MediapipeFaceKeypointDetector:
        if not isinstance(session, MediaPipeSession):
            raise TypeError(f"Expected MediaPipeSession, got {type(session).__name__}")
        if not isinstance(config, MediapipeFaceDetectorConfig):
            raise TypeError(f"Expected MediapipeFaceDetectorConfig, got {type(config).__name__}")

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        mp_running_mode = RunningMode.VIDEO if session.running_mode == "video" else RunningMode.IMAGE
        face_path = get_face_model_path()
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_path)),
            running_mode=mp_running_mode,
            num_faces=config.num_faces,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_face_tracking_confidence,
        )
        landmarker = FaceLandmarker.create_from_options(opts)
        return cls(config=config, session=session, landmarker=landmarker)


KEYPOINT_DETECTOR_REGISTRY["mediapipe_face"] = MediapipeFaceKeypointDetector


def _to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    import cv2
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_demo(config: MediapipeFaceDetectorConfig | None = None, camera_index: int = 0) -> None:
    from skellytracker.core.config.detection_stage_config import DetectionStageConfig
    from skellytracker.core.config.tracker_config import TrackerConfig
    from skellytracker.core.demo_manager import DemoManager

    from skellytracker.core.tracker.tracker import Tracker
    if config is None:
        config = MediapipeFaceDetectorConfig()
    session = MediaPipeSession.create()
    stage = DetectionStageConfig(name="face", keypoint_detectors=[config])
    tracker = Tracker.create(TrackerConfig(stages=[stage]), {"mediapipe": session})
    annotator = KeypointAnnotator.create(KeypointAnnotatorConfig(stage_schemas={
        "face": StageAnnotationSchema(connections=MediapipeFaceKeypointDetector.connections()),
    }))
    DemoManager(tracker=tracker, annotator=annotator, window_title="MediaPipe Face Demo").run_webcam(camera_index=camera_index)


if __name__ == "__main__":
    run_demo()
