"""CPU-only reference verification through the public Tracker factory and pool.

Run in a Tracker environment, not the optional conversion environment. No DLC,
Core or Forge imports are used. Inputs are the previously captured raw fixtures.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.detectors.detector_base_classes import ObjectDetector
from skellytracker.core.detectors.keypoint_detectors.dlc import (
    DlcDetector,
    DlcDetectorConfig,
)
from skellytracker.core.sessions.shared_sessions import SharedSessionPool
from skellytracker.core.tracker.detection_stage import DetectionStage
from skellytracker.core.tracker.tracker_factory import tracker_session_requests
from skellytracker.core.tracker.tracker_state import StageState, TrackerState


def compare(actual, expected):
    position = float(np.max(np.abs(actual.xyz[:, :2] - expected[:, :2])))
    confidence = float(np.max(np.abs(actual.visibility - expected[:, 2])))
    if position > 0.01 or confidence > 1e-4:
        raise AssertionError((position, confidence))
    return {"max_position_error_px": position, "max_confidence_error": confidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for bundle, fixture in (
        ("onnx-full-002", "full_tail"),
        ("onnx-crop", "offset_crop"),
    ):
        sidecar_path = args.artifact_root / bundle / "dlc_resnet50_gn.yaml"
        detector_config = DlcDetectorConfig(
            model_name=str(sidecar_path), confidence_threshold=0.0
        )
        config = TrackerConfig(
            stages=[
                DetectionStageConfig(
                    name="animal_pose", keypoint_detectors=[detector_config]
                )
            ]
        )
        config = TrackerConfig.model_validate(config.model_dump())
        requests = tracker_session_requests(
            config,
            batch_size=3,
            onnx_overrides={
                "execution_provider": "cpu",
                "fp16": False,
                "intra_op_num_threads": 2,
            },
        )
        pool = SharedSessionPool()
        lease = pool.create_tracker(config=config, requests=requests)
        second = pool.create_tracker(config=config, requests=requests)
        try:
            if lease._entries[0].session is not second._entries[0].session:
                raise AssertionError("Expected identical session reuse")
            reference = np.load(args.artifact_root / "reference" / f"{fixture}.npz")
            orders = ((0,), (2, 0), (2, 0, 1)) if fixture == "full_tail" else ((0,),)
            for order in orders:
                images = {
                    f"camera_{i}": np.ascontiguousarray(
                        reference["rgb_images"][i, ..., ::-1]
                    )
                    for i in order
                }
                start = time.perf_counter()
                observations, _ = lease.tracker.process_batch(
                    images, 0, {key: TrackerState() for key in images}
                )
                elapsed = time.perf_counter() - start
                metrics = {
                    key: compare(
                        observations[key].stages["animal_pose"].keypoints,
                        reference["keypoint_values"][i, 0],
                    )
                    for key, i in zip(images, order, strict=True)
                }
                results.append(
                    {
                        "case": fixture,
                        "camera_order": order,
                        "batch_seconds": elapsed,
                        "metrics": metrics,
                    }
                )
            if fixture == "offset_crop":
                detector = DlcDetector.create(
                    detector_config, lease._entries[0].session
                )
                source = np.zeros((480, 640, 3), dtype=np.uint8)
                source[37:264, 53:444] = reference["rgb_images"][0, ..., ::-1]
                boxes = Mock(spec=ObjectDetector)
                boxes.detect.return_value = [BoundingBox(53, 37, 444, 264)]
                stage = DetectionStage(
                    name="animal_pose",
                    keypoint_detectors=[detector],
                    object_detector=boxes,
                )
                try:
                    observation, _ = stage.run(source, StageState())
                    expected = reference["keypoint_values"][0, 0].copy()
                    expected[:, :2] += (53, 37)
                    results.append(
                        {
                            "case": "stage_crop_inverse_once",
                            **compare(observation.keypoints, expected),
                        }
                    )
                finally:
                    stage.close()
        finally:
            second.close()
            lease.close()
            pool.close()
    forbidden = [
        name
        for name in sys.modules
        if name.split(".")[0] in ("deeplabcut", "freemocap", "skellyforge")
    ]
    if forbidden:
        raise AssertionError(forbidden)
    report = {
        "provider": "CPUExecutionProvider",
        "results": results,
        "forbidden_imports": forbidden,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
