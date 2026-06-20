from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker, BaseTrackerConfig, BaseRecorder, BaseObservation
from skellytracker.trackers.rtmpose_tracker.rtmpose_annotator import RTMPoseImageAnnotator, RTMPoseImageAnnotatorConfig
from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import RTMPoseDetector, RTMPoseDetectorConfig


class RTMPoseTrackerConfig(BaseTrackerConfig):
    detector_config: RTMPoseDetectorConfig = Field(default_factory=RTMPoseDetectorConfig)
    annotator_config: RTMPoseImageAnnotatorConfig | None = Field(default_factory=lambda: RTMPoseImageAnnotatorConfig(draw_debug_bbox=True))


class RTMPoseRecorder(BaseRecorder):
    pass


class RTMPoseTracker(BaseTracker):
    config: RTMPoseTrackerConfig
    detector: RTMPoseDetector
    annotator: RTMPoseImageAnnotator | None = None
    recorder: RTMPoseRecorder | None = None

    @classmethod
    def create(cls, config: RTMPoseTrackerConfig | None = None):
        if config is None:
            config = RTMPoseTrackerConfig()
        detector = RTMPoseDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=RTMPoseImageAnnotator.create(config.annotator_config),
            recorder=RTMPoseRecorder(),
        )

    def process_image(
        self,
        frame_number: int,
        image: NDArray[np.uint8],
        record_observation: bool = True,
    ) -> BaseObservation:
        obs = super().process_image(frame_number, image, record_observation)

        # Draw debug bboxes if enabled.
        annot_cfg = self.annotator.config if self.annotator else None
        if annot_cfg is not None and annot_cfg.draw_debug_bbox:
            bboxes = self.detector.session.last_bboxes
            from_detector_list = self.detector.session.last_bboxes_from_detector
            if bboxes and from_detector_list:
                for bbox, from_det in zip(bboxes, from_detector_list):
                    if bbox is not None:
                        RTMPoseImageAnnotator.draw_bbox_on_image(
                            image, bbox,
                            from_detector=from_det,
                            label="YOLOX" if from_det else "track",
                        )

        return obs


if __name__ == "__main__":
    import onnxruntime as ort

    ort.preload_dlls()
    print(f"ort.get_available_providers() -> {ort.get_available_providers()}")
    RTMPoseTracker.create().demo()
