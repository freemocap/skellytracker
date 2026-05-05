from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_config import (
    CompositeGPUDetectorConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_observation import (
    CompositeGPUObservation,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_session import (
    CompositeGPUSession,
    CompositeGPUSessionConfig,
)
from skellytracker.trackers.composite_gpu_tracker.names_and_connections import (
    RTMO_HYBRID_DEFINITION,
)


@dataclass
class CompositeGPUDetector(BaseDetector):
    config: CompositeGPUDetectorConfig
    session: CompositeGPUSession

    @classmethod
    def create(cls, config: CompositeGPUDetectorConfig | None = None) -> "CompositeGPUDetector":
        if config is None:
            config = CompositeGPUDetectorConfig()

        session_config = config.session_config
        session_config.detect_hands = config.detect_hands
        session_config.detect_face = config.detect_face

        session = CompositeGPUSession.create(session_config)

        return cls(
            config=config,
            session=session,
            tracked_object=RTMO_HYBRID_DEFINITION,
        )

    def detect(self, frame_number: int, image: NDArray[np.uint8]) -> CompositeGPUObservation:
        result = self.session.predict_single(image)

        # RTMO returns float32; cast keypoints to float64 for beartype + PointCloud.
        raw_hands = result.get("raw_hands")
        return CompositeGPUObservation.from_detection_results(
            frame_number=frame_number,
            image_size=(int(image.shape[0]), int(image.shape[1])),
            body_keypoints=result["body"][0].astype(np.float64),
            body_scores=result["body"][1],
            hands_keypoints=result["hands"][0].astype(np.float64),
            hands_scores=result["hands"][1],
            face_keypoints=result["face"][0].astype(np.float64),
            face_scores=result["face"][1],
            right_hand_roi=result.get("right_hand_roi"),
            left_hand_roi=result.get("left_hand_roi"),
            face_roi=result.get("face_roi"),
            raw_hands_keypoints=raw_hands[0].astype(np.float64) if raw_hands is not None else None,
            raw_hands_scores=raw_hands[1] if raw_hands is not None else None,
        )
