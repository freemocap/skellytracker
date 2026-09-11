"""DeepLabCut-Live (PyTorch) keypoint detector.

Wraps dlclive.DLCLive using the "pytorch" engine only (not TensorFlow). Unlike
RTMPose/MediaPipe, DLC-Live models are custom-trained per project, so keypoint
names are not declared in a checked-in YAML schema — they are read from the
exported model's own embedded config the first time a frame is processed.

Runs full-frame only: no built-in top-down cropping logic. Can still be
composed behind an ObjectDetector via the normal
DetectionStageConfig.object_detector mechanism, same as any other
KeypointDetector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.sessions.dlc_live_session import DLCLiveSession
from skellytracker.core.sessions.session import Session


class DLCLiveKeypointDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["dlclive"] = "dlclive"
    session_backend: Literal["dlclive"] = "dlclive"
    model_path: str
    confidence_threshold: float = 0.0
    dynamic: tuple[bool, float, float] = (False, 0.5, 10)
    resize: float | None = None


@dataclass
class DLCLiveKeypointDetector(KeypointDetector):
    """DeepLabCut-Live PyTorch-engine detector. Runs on the full (cropped) image.

    Keypoint names/count are only known after the first call to detect(), when
    dlclive.DLCLive.init_inference() loads the model and exposes its config.
    """

    config: DLCLiveKeypointDetectorConfig
    session: DLCLiveSession
    dlc_live: Any = field(repr=False)
    _point_names: tuple[str, ...] = field(default=(), init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def detect(
        self,
        image: NDArray[np.uint8],
        context: DetectionContext | None = None,
    ) -> Keypoints:
        if not self._initialized:
            pose = self.dlc_live.init_inference(image)
            self._point_names = _extract_bodypart_names(self.dlc_live.cfg)
            self._initialized = True
        else:
            pose = self.dlc_live.get_pose(image)

        pose = np.asarray(pose, dtype=np.float64)
        n = len(self._point_names)
        xyz = np.zeros((n, 3), dtype=np.float64)
        xyz[:, 0] = pose[:, 0]
        xyz[:, 1] = pose[:, 1]

        visibility = pose[:, 2].copy()
        below_threshold = visibility < self.config.confidence_threshold
        xyz[below_threshold] = np.nan
        visibility[below_threshold] = 0.0

        return Keypoints(names=self._point_names, xyz=xyz, visibility=visibility)

    def preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Any]:
        """Not used: dlclive owns its own preprocessing internally.

        DetectionStage.run_batch only calls preprocess/postprocess for
        ONNX-backed detectors; non-ONNX detectors like this one go through the
        per-camera detect() thread-pool path instead (see
        MediapipePoseKeypointDetector for the same contract).
        """
        raise NotImplementedError(
            "DLCLiveKeypointDetector does not support the batched ONNX path"
        )

    def postprocess(self, raw: Any, metadata: Any) -> Keypoints:
        raise NotImplementedError(
            "DLCLiveKeypointDetector does not support the batched ONNX path"
        )

    def close(self) -> None:
        self.dlc_live.close()

    def reset_temporal_state(self) -> None:
        self.dlc_live.close()
        self.dlc_live = type(self).create(self.config, self.session).dlc_live
        self._initialized = False

    @classmethod
    def create(
        cls, config: KeypointDetectorConfig, session: Session
    ) -> DLCLiveKeypointDetector:
        if not isinstance(session, DLCLiveSession):
            raise TypeError(f"Expected DLCLiveSession, got {type(session).__name__}")
        if not isinstance(config, DLCLiveKeypointDetectorConfig):
            raise TypeError(
                f"Expected DLCLiveKeypointDetectorConfig, got {type(config).__name__}"
            )

        from dlclive import DLCLive

        dlc_live = DLCLive(
            config.model_path,
            model_type="pytorch",
            precision=session.precision,
            device=session.device,
            single_animal=True,
            dynamic=config.dynamic,
            resize=config.resize,
        )
        return cls(config=config, session=session, dlc_live=dlc_live)


def _extract_bodypart_names(cfg: dict) -> tuple[str, ...]:
    """Pull the ordered bodypart/keypoint names out of a DLCLive pytorch cfg dict.

    cfg is dlclive.DLCLive.cfg, i.e. the "config" entry embedded in the
    exported .pt checkpoint. The exact key path was not confirmed against a
    real exported model at implementation time — verify against one and
    extend the candidate paths below if needed.
    """
    candidate_paths: tuple[tuple[str, ...], ...] = (
        ("metadata", "bodyparts"),
        ("bodyparts",),
        ("metadata", "all_joints_names"),
        ("all_joints_names",),
    )
    for path in candidate_paths:
        node: Any = cfg
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (list, tuple)) and len(node) > 0:
            return tuple(str(name) for name in node)

    raise ValueError(
        "Could not find bodypart/keypoint names in DLCLive model config. "
        f"Tried key paths: {candidate_paths}. Available top-level keys: {list(cfg.keys())}"
    )


KEYPOINT_DETECTOR_REGISTRY["dlclive"] = DLCLiveKeypointDetector
