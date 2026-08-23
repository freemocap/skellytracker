"""Pelvis interior hydration makes the root a non-degenerate rigid body.

Only pelvis_origin + the two hips are directly tracked, and they are collinear
(pelvis_origin is their midpoint). The anatomical_offset entries add the sacral top,
the iliac crests, and the pubic symphysis so the pelvis Kabsch fit has a real 3D
cloud. The offsets must reproduce pelvis.yaml's rest positions at the T-pose so
``identity == T-pose`` is preserved.
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


def _tpose_body() -> dict[str, np.ndarray]:
    """A T-pose torso in SkellyForge (Blender) world axes (+X right, +Y forward, +Z up).

    Hips at +/-88 mm (hip_width 176), shoulders at +/-160 mm (shoulder_width 320)
    directly above so the trunk vertical (hip midpoint -> shoulder midpoint) is pure +Z.
    """
    return {
        "left_hip": np.array([-88.0, 0.0, 0.0]),
        "right_hip": np.array([88.0, 0.0, 0.0]),
        "left_shoulder": np.array([-160.0, 0.0, 470.0]),
        "right_shoulder": np.array([160.0, 0.0, 470.0]),
    }


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_interior_lands_on_rest_positions(mapping_path):
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(_tpose_body())

    # rest positions from pelvis.yaml (SkellyForge, Blender axes), at the T-pose.
    assert np.allclose(result["sacrum_top"], [0.0, -35.0, 95.0], atol=0.5)
    assert np.allclose(result["left_iliac_crest"], [-145.0, 0.0, 100.0], atol=0.5)
    assert np.allclose(result["right_iliac_crest"], [145.0, 0.0, 100.0], atol=0.5)
    assert np.allclose(result["pubic_symphysis"], [0.0, 60.0, -10.0], atol=0.5)


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_cloud_is_non_collinear_after_hydration(mapping_path):
    """The hydrated pelvis landmarks span a plane (rank >= 2), so a Kabsch fit is
    no longer degenerate — the whole point of hydrating them."""
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(_tpose_body())
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
