"""Standalone arbitrary-model recipes and declared source-coordinate restoration."""

from dataclasses import FrozenInstanceError
import hashlib
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.detectors.keypoint_detectors.dlc import (
    DlcDetector,
    DlcDetectorConfig,
)
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.tracker.tracker_factory import tracker_session_requests
from skellytracker.core.tracker.tracker_presets import (
    dlc_tracker_config,
    dlc_tracker_metadata,
)


def write_sidecar(tmp_path, resize):
    data = {
        "schema_version": "v2024.09.1019",
        "model_id": "fixture_points",
        "display_name": "Arbitrary feature model",
        "role": ["pose_estimator"],
        "input": {
            "name": "images",
            "dtype": {"fp32": "float32"},
            "layout": "NCHW",
            "normalization": "imagenet_rgb",
            "resize": resize,
        },
        "outputs": [
            {
                "name": "keypoints",
                "dtype": {"fp32": "float32"},
                "shape": [-1, 3, 3],
                "semantic": "keypoints",
                "fields": ["x", "y", "confidence"],
                "keypoint_axis": 1,
                "keypoint_count": 3,
            }
        ],
        "pose": {
            "estimator_type": "bottom_up",
            "requires_object_detector": False,
            "tracked_points": ["feature_z", "feature_a", "feature_q"],
            "connections": [
                {"name": "feature_edges", "edges": [["feature_q", "feature_z"]]}
            ],
            "decode": {"method": "coordinate", "coordinate_scale": "pixel"},
        },
        "sizes": {
            "native": {
                "input": {"shape": [-1, 3, 8, 12]},
                "onnx": {
                    "batch_artifacts": {
                        "dynamic": {
                            "precision_artifacts": {
                                "fp32": {
                                    "filename": "pose_model.onnx",
                                    "url_sha256": hashlib.sha256(b"fixture artifact").hexdigest(),
                                    "cuda_use_tf32": False,
                                }
                            }
                        }
                    }
                },
            }
        },
    }
    path = tmp_path / "fixture_points.yaml"
    (tmp_path / "pose_model.onnx").write_bytes(b"fixture artifact")
    path.write_text(yaml.safe_dump(data), encoding="utf8")
    return path


def test_recipe_metadata_and_serialization_are_standalone(tmp_path):
    path = write_sidecar(tmp_path, {"method": "none"})
    metadata = dlc_tracker_metadata(sidecar_path=path)
    assert metadata.keypoint_names == ("feature_z", "feature_a", "feature_q")
    assert metadata.connections == (("feature_q", "feature_z"),)
    assert metadata.connection_groups == (("feature_edges", (("feature_q", "feature_z"),)),)
    assert metadata.input_size == (8, 12)
    with pytest.raises(FrozenInstanceError):
        metadata.model_name = "changed"
    config = dlc_tracker_config(sidecar_path=path, confidence_threshold=0.73)
    assert config.preset_name == "dlc"
    assert config.stages[0].name == metadata.stage_path[0]
    assert config.stages[0].object_detector is None and config.keypoint_assembly is None
    assert config.stages[0].keypoint_detectors[0].confidence_threshold == 0.73
    assert TrackerConfig.model_validate_json(config.model_dump_json()) == config
    requests = tracker_session_requests(
        config,
        batch_size=3,
        onnx_overrides={"execution_provider": "cpu", "fp16": False},
    )
    assert len(requests) == 1
    assert requests[0].config.models[0].cuda_use_tf32 is False
    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        tracker_session_requests(
            config,
            batch_size=1,
            onnx_overrides={"execution_provider": "trt", "fp16": False},
        )


def test_metadata_rejects_missing_or_modified_artifact(tmp_path):
    path = write_sidecar(tmp_path, {"method": "none"})
    artifact = tmp_path / "pose_model.onnx"
    artifact.write_bytes(b"incorrect content")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dlc_tracker_metadata(sidecar_path=path)
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        dlc_tracker_metadata(sidecar_path=path)


@pytest.mark.parametrize(
    "method,source_size,model_xy",
    [
        ("fit", (10, 10), (3.6, 2.4)),
        ("fit", (20, 10), (4.8, 1.2)),
        ("letterbox", (7, 9), (1 + 20 / 9, 24 / 7)),
    ],
)
def test_declared_resize_restores_source_coordinates_and_owns_output(
    tmp_path, method, source_size, model_xy
):
    resize = {"method": method, "target_size": [8, 12], "pad_value": 0}
    if method == "letterbox":
        resize.update(pad_position="center", rounding="half_even")
    path = write_sidecar(tmp_path, resize)
    session = Mock(spec=OnnxSession)
    raw = np.array(
        [[(*model_xy, 0.9), (*model_xy, 0.8), (*model_xy, 0.1)]], dtype=np.float32
    )
    session.run.return_value = [raw]
    detector = DlcDetector(DlcDetectorConfig(model_name=str(path)), session)
    image = np.zeros((*source_size, 3), dtype=np.uint8)
    points = detector.detect(image)
    np.testing.assert_allclose(points.xyz[:2], [[2, 3, 0], [2, 3, 0]], atol=1e-6)
    assert np.isnan(points.xyz[2]).all()
    tensor = session.run.call_args.args[1]["images"]
    assert tensor.shape == (1, 3, 8, 12) and tensor.dtype == np.float32
    previous = points.xyz.copy()
    raw[..., :2] += 1
    detector.detect(image)
    np.testing.assert_array_equal(points.xyz, previous)


def test_native_policy_preserved_and_wrong_image_rejected(tmp_path):
    path = write_sidecar(tmp_path, {"method": "none"})
    detector = DlcDetector(
        DlcDetectorConfig(model_name=str(path)), Mock(spec=OnnxSession)
    )
    tensor, transform = detector.preprocess(np.zeros((8, 12, 3), dtype=np.uint8))
    assert tensor.shape == (3, 8, 12)
    assert transform.scale_x == transform.scale_y == 1
    with pytest.raises(ValueError, match="native preprocessing"):
        detector.preprocess(np.zeros((9, 12, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "resize",
    [
        {"method": "stretch", "target_size": [8, 12]},
        {"method": "fit", "target_size": [7, 12], "pad_value": 0},
        {"method": "fit", "target_size": [8, 12]},
        {"method": "letterbox", "target_size": [8, 12], "pad_value": 0},
    ],
)
def test_unsupported_or_incomplete_resize_fails_before_session_creation(
    tmp_path, resize
):
    with pytest.raises(ValueError, match="DLC|target_size"):
        dlc_tracker_config(sidecar_path=write_sidecar(tmp_path, resize))
