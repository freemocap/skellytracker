from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest

from skellytracker.core.sessions.cpu_session import CpuSession
from skellytracker.core.io.tracker_mapping import TrackerMapping


@pytest.mark.parametrize("part", ["body", "hand", "face"])
def test_standalone_pose_contract_and_precomputed_normalization(part):
    module = import_module(
        f"skellytracker.core.detectors.keypoint_detectors.rtmpose.{part}.rtmpose_{part}_detector"
    )
    detector_class = getattr(module, f"RTMPose{part.title()}Detector")
    config_class = getattr(module, f"RTMPose{part.title()}DetectorConfig")
    detector = detector_class(config_class(), CpuSession())
    with patch(
        "skellytracker.core.detectors.processing.image_preprocessing.build_normalization_fn",
        side_effect=AssertionError("normalization must be prepared before processing"),
    ):
        tensor, metadata = detector.preprocess(np.zeros((80, 120, 3), dtype=np.uint8))
    assert tensor.shape[1:] == detector.config.input_size
    height, width = detector.config.input_size
    wrong_count = len(detector._point_names) + 1
    with pytest.raises(ValueError, match="landmark count"):
        detector.postprocess(
            [
                np.zeros((1, wrong_count, width * 2)),
                np.zeros((1, wrong_count, height * 2)),
            ],
            metadata,
        )
    with pytest.raises(KeyError):
        _ = config_class(model_name="undeclared_model").input_size


def test_standalone_body_mapping_only_references_its_declared_points():
    from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
        RTMPoseBodyDetector,
        RTMPoseBodyDetectorConfig,
    )

    detector = RTMPoseBodyDetector(RTMPoseBodyDetectorConfig(), CpuSession())
    TrackerMapping.from_yaml(
        detector.standard_human_mapping_path(),
        known_tracker_keypoints=set(detector._point_names),
    )


def test_old_and_new_wholebody_imports_share_one_detector_registration():
    from skellytracker.core.detectors.keypoint_detectors.rtmpose import (
        RTMPoseKeypointDetector,
        RTMPoseDetectorConfig,
    )
    from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import (
        RTMWWholebodyDetector,
        RTMWWholebodyDetectorConfig,
    )
    from skellytracker.core.detectors.detector_base_classes import (
        KEYPOINT_DETECTOR_REGISTRY,
    )

    assert RTMPoseKeypointDetector is RTMWWholebodyDetector
    assert RTMPoseDetectorConfig is RTMWWholebodyDetectorConfig
    assert KEYPOINT_DETECTOR_REGISTRY["rtmpose"] is RTMWWholebodyDetector
