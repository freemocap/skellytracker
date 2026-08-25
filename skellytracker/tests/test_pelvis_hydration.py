"""Pelvis interior hydration makes the root a non-degenerate rigid body.

Only pelvis_origin + the two hips are directly tracked, and they are collinear
(pelvis_origin is their midpoint). The anatomical_offset entries add the sacral top,
the iliac crests, and the pubic symphysis so the pelvis Kabsch fit has a real 3D
cloud. The offsets must reproduce SkellyForge's authored rest positions at the
T-pose so ``identity == T-pose`` is preserved.

The rest-pose-driven test derives its synthetic observations from the authored
skeleton itself rather than from hand-copied coordinates - hand-copied numbers
drift silently when the model moves on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skellytracker.core.io.mapping_paths import (
    MEDIAPIPE_BODY_MAPPING,
    RTMPOSE_BODY_MAPPING,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping

_BODY_MAPPINGS = [MEDIAPIPE_BODY_MAPPING, RTMPOSE_BODY_MAPPING]

# Landmarks whose authored rest position sits structurally off the plane their
# entry's two-axis frame spans, so no ratio can land them exactly (measured
# residual ~6.4 mm, documented the same way in skellyforge's round-trip guard).
_UNREACHABLE_ALLOWANCE_MM = 7.0
_DEFAULT_TOLERANCE_MM = 1.0


def _authored_rest_pose():
    pytest.importorskip("skellyforge", reason="skellyforge is a dev dependency")
    from skellyforge.core.skeleton.pose.rest_pose import RestPose
    from skellyforge.core.skeleton.skeleton_definition import SkeletonDefinition

    skeleton = SkeletonDefinition.from_default_yaml()
    return RestPose.from_default_yaml(skeleton=skeleton)


def _tpose_tracker_body(rest_pose) -> dict[str, np.ndarray]:
    """Tracker keypoints standing in at the authored T-pose.

    The pelvis offsets consume only the hips and shoulders, so those four
    stand-ins are sufficient; they mirror the table in skellyforge's
    ``test_tracker_mapping_offset_round_trip.py``.
    """
    landmark_positions = rest_pose.landmark_positions
    return {
        "left_hip": landmark_positions["left_hip_socket"].array,
        "right_hip": landmark_positions["right_hip_socket"].array,
        "left_shoulder": landmark_positions["left_acromion"].array,
        "right_shoulder": landmark_positions["right_acromion"].array,
    }


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_interior_lands_on_rest_positions(mapping_path):
    rest_pose = _authored_rest_pose()
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(_tpose_tracker_body(rest_pose))

    for landmark_name in (
        "sacrum_top",
        "left_iliac_crest",
        "right_iliac_crest",
        "pubic_symphysis",
    ):
        allowance_mm = (
            _UNREACHABLE_ALLOWANCE_MM
            if "iliac_crest" in landmark_name
            else _DEFAULT_TOLERANCE_MM
        )
        error_mm = float(
            np.linalg.norm(
                np.asarray(result[landmark_name])
                - rest_pose.landmark_positions[landmark_name].array
            )
        )
        assert error_mm <= allowance_mm, (
            f"{landmark_name}: {error_mm:.2f} mm from its authored rest position "
            f"(allowed {allowance_mm} mm) - regenerate the mapping ratios"
        )


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_cloud_is_non_collinear_after_hydration(mapping_path):
    """The hydrated pelvis landmarks span a plane (rank >= 2), so a Kabsch fit is
    no longer degenerate — the whole point of hydrating them."""
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(
        {
            "left_hip": np.array([-88.0, 0.0, 0.0]),
            "right_hip": np.array([88.0, 0.0, 0.0]),
            "left_shoulder": np.array([-160.0, 0.0, 470.0]),
            "right_shoulder": np.array([160.0, 0.0, 470.0]),
        }
    )
    cloud = np.stack([
        result["pelvis_origin"],
        result["left_hip_joint"],
        result["right_hip_joint"],
        result["sacrum_top"],
        result["left_iliac_crest"],
        result["pubic_symphysis"],
    ])
    centered = cloud - cloud.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    assert singular_values[1] > 0.1 * singular_values[0]
