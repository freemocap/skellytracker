"""Tests for skellytracker.core.sidecar.model.SidecarModel against the spec's
"Validation rules" checklist (specs/sidecar-spec.md). Each test is commented
with the rule it exercises.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from skellytracker.core.sidecar.model import SidecarModel


def _minimal_detector() -> dict:
    """A minimal, valid object_detector sidecar (dict form, pre-validation)."""
    return {
        "schema_version": "v2024.09.1019",
        "model_id": "toy_detector",
        "display_name": "Toy Detector",
        "role": ["object_detector"],
        "input": {
            "name": "images",
            "dtype": {"fp32": "float32"},
            "normalization": "unit_float",
            "resize": {"method": "letterbox", "preserve_aspect_ratio": True, "pad_value": 114},
        },
        "outputs": [
            {
                "name": "output0",
                "dtype": {"fp32": "float32"},
                "shape": [1, 300, 6],
                "rank": 3,
                "semantic": "detections",
                "fields": ["x1", "y1", "x2", "y2", "score", "class_id"],
            }
        ],
        "decode": {
            "box_format": "xyxy",
            "score_field": "score",
            "class_field": "class_id",
        },
        "sizes": {
            "nano": {
                "input": {"shape": [-1, 3, 640, 640], "resize": {"target_size": [640, 640]}},
                "onnx": {
                    "batch_artifacts": {
                        "2": {
                            "precision_artifacts": {"fp32": {"filename": "toy-nano_b2_fp32.onnx"}},
                        }
                    }
                },
            }
        },
    }


def _minimal_pose() -> dict:
    """A minimal, valid pose_estimator sidecar (top-down, simcc decode)."""
    return {
        "schema_version": "v2024.09.1019",
        "model_id": "toy_pose",
        "display_name": "Toy Pose",
        "role": ["pose_estimator"],
        "input": {
            "name": "input",
            "dtype": {"fp32": "float32"},
            "normalization": "imagenet_bgr",
            "resize": {"method": "affine_person_crop", "crop_policy": {"expand_ratio": 1.25}},
        },
        "outputs": [
            {"name": "simcc_x", "dtype": {"fp32": "float32"}, "semantic": "simcc_x", "keypoint_axis": 1},
            {"name": "simcc_y", "dtype": {"fp32": "float32"}, "semantic": "simcc_y", "keypoint_axis": 1},
        ],
        "pose": {
            "estimator_type": "top_down_single_person",
            "requires_object_detector": True,
            "tracked_points": ["nose", "left_shoulder", "right_shoulder"],
            "connections": [{"name": "skel", "edges": [["nose", "left_shoulder"]]}],
            "decode": {"method": "simcc"},
        },
        "sizes": {
            "m": {
                "input": {"shape": [-1, 3, 256, 192], "resize": {"target_size": [256, 192]}},
                "onnx": {"batch_artifacts": {"dynamic": {"precision_artifacts": {"fp32": {"filename": "toy_pose-m_fp32.onnx"}}}}},
            }
        },
    }


def _mutate(base: dict, **overrides) -> dict:
    data = copy.deepcopy(base)
    data.update(overrides)
    return data


class TestBaselineFixturesAreValid:
    def test_minimal_detector_validates(self):
        SidecarModel.model_validate(_minimal_detector())

    def test_minimal_pose_validates(self):
        SidecarModel.model_validate(_minimal_pose())

    def test_resolved_size_validates(self):
        sidecar = SidecarModel.model_validate(_minimal_detector())
        size = sidecar.resolved_size("nano")
        assert size.input.shape == [-1, 3, 640, 640]
        assert size.onnx.native_batch_sizes == [2]

    def test_resolved_size_dynamic_batch(self):
        sidecar = SidecarModel.model_validate(_minimal_pose())
        size = sidecar.resolved_size("m")
        assert size.onnx.supports_dynamic_batch is True


class TestIdentityAndRole:
    def test_role_must_be_nonempty(self):
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(_mutate(_minimal_detector(), role=[]))

    def test_schema_version_pattern_enforced(self):
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(_mutate(_minimal_detector(), schema_version="not-a-version"))

    def test_schema_version_rejects_prerelease_tag(self):
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(_mutate(_minimal_detector(), schema_version="v2024.09.1019-beta"))

    def test_sizes_must_be_nonempty(self):
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(_mutate(_minimal_detector(), sizes={}))


class TestRoleRequirements:
    def test_object_detector_role_requires_decode(self):
        data = _minimal_detector()
        del data["decode"]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_decode_present_without_object_detector_role_is_invalid(self):
        data = _minimal_pose()
        data["decode"] = {"box_format": "xyxy"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_pose_estimator_role_requires_pose(self):
        data = _minimal_pose()
        del data["pose"]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)


class TestPoseValidation:
    def test_requires_object_detector_must_match_estimator_type(self):
        data = _minimal_pose()
        data["pose"]["requires_object_detector"] = False
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_bottom_up_requires_object_detector_false(self):
        data = _minimal_pose()
        data["pose"]["estimator_type"] = "bottom_up"
        data["pose"]["requires_object_detector"] = True
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_requires_object_detector_requires_affine_crop_resize(self):
        data = _minimal_pose()
        data["input"]["resize"] = {"method": "letterbox"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_tracked_points_must_be_unique(self):
        data = _minimal_pose()
        data["pose"]["tracked_points"] = ["nose", "nose"]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_connection_endpoint_must_exist_in_tracked_points(self):
        data = _minimal_pose()
        data["pose"]["connections"] = [{"name": "skel", "edges": [["nose", "ghost_point"]]}]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_skeleton_names_must_be_unique(self):
        data = _minimal_pose()
        data["pose"]["connections"] = [
            {"name": "skel", "edges": [["nose", "left_shoulder"]]},
            {"name": "skel", "edges": [["nose", "right_shoulder"]]},
        ]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_derived_points_source_must_exist(self):
        data = _minimal_pose()
        data["pose"]["derived_points"] = {"neck": ["ghost_point"]}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_derived_points_mean_form_accepted(self):
        data = _minimal_pose()
        data["pose"]["derived_points"] = {"neck": ["left_shoulder", "right_shoulder"]}
        SidecarModel.model_validate(data)

    def test_canonical_mapping_weighted_form_accepted(self):
        data = _minimal_pose()
        data["pose"]["canonical_mapping"] = {"head_center": {"nose": 1.0}}
        SidecarModel.model_validate(data)

    def test_canonical_mapping_prefixes_expansion(self):
        data = _minimal_pose()
        data["pose"]["tracked_points"] = ["right_hand_root", "left_hand_root"]
        data["pose"]["connections"] = [{"name": "skel", "edges": [["right_hand_root", "left_hand_root"]]}]
        data["pose"]["canonical_mapping"] = {
            "prefixes": ["right_hand_", "left_hand_"],
            "[prefix]wrist": "root",
        }
        sidecar = SidecarModel.model_validate(data)
        assert sidecar.pose.canonical_mapping == {
            "right_hand_wrist": "right_hand_root",
            "left_hand_wrist": "left_hand_root",
        }

    def test_heatmap_3d_is_rejected(self):
        data = _minimal_pose()
        data["pose"]["decode"] = {"method": "heatmap", "is_3d": True, "depth_unit": "mm"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_is_3d_requires_depth_unit(self):
        data = _minimal_pose()
        data["pose"]["decode"] = {"method": "coordinate", "is_3d": True}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_simcc_3d_requires_depth_range(self):
        data = _minimal_pose()
        data["pose"]["decode"] = {"method": "simcc", "is_3d": True, "depth_unit": "mm"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_simcc_decode_requires_matching_outputs(self):
        data = _minimal_pose()
        data["outputs"] = data["outputs"][:1]  # drop simcc_y
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_poses_output_requires_keypoint_axis_2(self):
        data = _minimal_pose()
        data["pose"]["decode"] = {"method": "coordinate"}
        data["outputs"] = [
            {"name": "det", "dtype": {"fp32": "float32"}, "semantic": "detections", "fields": ["x1", "y1", "x2", "y2", "score"]},
            {"name": "poses", "dtype": {"fp32": "float32"}, "semantic": "poses", "keypoint_axis": 1},
        ]
        data["role"] = ["pose_estimator"]
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)


class TestDetectionDecodeValidation:
    def test_detections_output_requires_fields(self):
        data = _minimal_detector()
        data["outputs"][0]["fields"] = None
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_exactly_one_detections_output_required(self):
        data = _minimal_detector()
        data["outputs"].append(copy.deepcopy(data["outputs"][0]))
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_score_field_must_appear_in_fields(self):
        data = _minimal_detector()
        data["decode"]["score_field"] = "confidence"
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_confidence_threshold_default_must_be_in_unit_range(self):
        data = _minimal_detector()
        data["decode"]["confidence_threshold_default"] = 1.5
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_person_class_id_must_be_gte_class_id_base(self):
        data = _minimal_detector()
        data["decode"]["class_id_base"] = 5
        data["decode"]["person_class_id"] = 0
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_max_detections_must_be_positive(self):
        data = _minimal_detector()
        data["decode"]["max_detections"] = 0
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)


class TestInputAndNormalization:
    def test_dtype_keys_restricted_to_precision_enum(self):
        data = _minimal_detector()
        data["input"]["dtype"] = {"bogus": "float32"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_normalization_named_mode_accepted(self):
        data = _minimal_detector()
        data["input"]["normalization"] = "imagenet_rgb"
        SidecarModel.model_validate(data)

    def test_normalization_custom_object_accepted(self):
        data = _minimal_detector()
        data["input"]["normalization"] = {"mode": "custom", "scale": 0.0039, "mean": [0, 0, 0], "std": [1, 1, 1]}
        SidecarModel.model_validate(data)

    def test_normalization_custom_mean_must_be_length_3(self):
        data = _minimal_detector()
        data["input"]["normalization"] = {"mode": "custom", "mean": [0, 0]}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_normalization_by_precision_values_restricted_to_named_modes(self):
        data = _minimal_detector()
        data["input"]["normalization_by_precision"] = {"fp32": "unit_float"}
        SidecarModel.model_validate(data)

    def test_layout_restricted_to_nchw_nhwc(self):
        data = _minimal_detector()
        data["input"]["layout"] = "HWC"
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)


class TestResize:
    def test_crop_policy_only_valid_for_affine_person_crop(self):
        data = _minimal_detector()
        data["input"]["resize"] = {"method": "letterbox", "crop_policy": {"expand_ratio": 1.25}}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_pad_value_only_valid_for_letterbox(self):
        data = _minimal_pose()
        data["input"]["resize"] = {"method": "affine_person_crop", "pad_value": 114}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_target_size_must_be_length_2(self):
        data = _minimal_detector()
        data["sizes"]["nano"]["input"]["resize"]["target_size"] = [640, 640, 640]
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_shape_batch_axis_must_be_minus_one(self):
        data = _minimal_detector()
        data["sizes"]["nano"]["input"]["shape"] = [1, 3, 640, 640]
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_shape_spatial_dims_must_match_target_size(self):
        data = _minimal_detector()
        data["sizes"]["nano"]["input"]["shape"] = [-1, 3, 320, 320]
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_target_size_required_unless_supports_dynamic_size(self):
        data = _minimal_detector()
        del data["sizes"]["nano"]["input"]["resize"]["target_size"]
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_supports_dynamic_size_omits_target_size(self):
        data = _minimal_detector()
        data["input"]["resize"]["supports_dynamic_size"] = True
        del data["sizes"]["nano"]["input"]["resize"]["target_size"]
        data["sizes"]["nano"]["input"]["shape"] = [-1, 3, -1, -1]
        sidecar = SidecarModel.model_validate(data)
        sidecar.resolved_size("nano")  # should not raise


class TestOnnxArtifacts:
    def test_batch_artifacts_cannot_mix_dynamic_and_integer_keys(self):
        data = _minimal_detector()
        data["sizes"]["nano"]["onnx"]["batch_artifacts"]["dynamic"] = {
            "precision_artifacts": {"fp32": {"filename": "x.onnx"}}
        }
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_url_present_requires_url_sha256(self):
        data = _minimal_detector()
        data["sizes"]["nano"]["onnx"]["batch_artifacts"]["2"]["precision_artifacts"]["fp32"]["url"] = "https://example.com/x.zip"
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_url_with_sha256_is_valid(self):
        data = _minimal_detector()
        artifact = data["sizes"]["nano"]["onnx"]["batch_artifacts"]["2"]["precision_artifacts"]["fp32"]
        artifact["url"] = "https://example.com/x.zip"
        artifact["url_sha256"] = "a" * 64
        sidecar = SidecarModel.model_validate(data)
        sidecar.resolved_size("nano")

    def test_multi_batch_requires_output_shapes(self):
        data = _minimal_detector()
        artifacts = data["sizes"]["nano"]["onnx"]["batch_artifacts"]
        artifacts["4"] = {"precision_artifacts": {"fp32": {"filename": "toy-nano_b4_fp32.onnx"}}}
        sidecar = SidecarModel.model_validate(data)
        with pytest.raises(ValidationError):
            sidecar.resolved_size("nano")

    def test_multi_batch_with_output_shapes_is_valid(self):
        data = _minimal_detector()
        artifacts = data["sizes"]["nano"]["onnx"]["batch_artifacts"]
        artifacts["2"]["output_shapes"] = [[2, 300, 6]]
        artifacts["4"] = {
            "precision_artifacts": {"fp32": {"filename": "toy-nano_b4_fp32.onnx"}},
            "output_shapes": [[4, 300, 6]],
        }
        sidecar = SidecarModel.model_validate(data)
        sidecar.resolved_size("nano")


class TestOverlay:
    def test_overlay_skeleton_string_must_name_existing_connection(self):
        data = _minimal_pose()
        data["overlay"] = {"skeleton": "does_not_exist"}
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_overlay_group_cannot_use_both_connections_and_prefix(self):
        data = _minimal_pose()
        data["overlay"] = {
            "groups": {
                "body": {
                    "connections": [["nose", "left_shoulder"]],
                    "prefix": "nose",
                    "connection_color": [0, 200, 100],
                }
            }
        }
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_at_most_one_default_group(self):
        data = _minimal_pose()
        data["overlay"] = {
            "groups": {
                "a": {"connection_color": [0, 0, 0]},
                "b": {"connection_color": [1, 1, 1]},
            }
        }
        with pytest.raises(ValidationError):
            SidecarModel.model_validate(data)

    def test_valid_overlay_with_prefix_group(self):
        data = _minimal_pose()
        data["overlay"] = {
            "skeleton": "skel",
            "groups": {"body": {"connection_color": [0, 200, 100]}},
        }
        SidecarModel.model_validate(data)
