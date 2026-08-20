"""Pelvis interior hydration makes the root a non-degenerate rigid body.

Only hips_center + the two hips are directly tracked, and they are collinear
(hips_center is their midpoint). The anatomical_offset entries add the sacral top,
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
    """A T-pose torso in standard-human world axes (+X fwd, +Y left, +Z up).

    Hips at +/-88 mm (hip_width 176), shoulders directly above so the trunk
    vertical (hips_center -> neck_center) is pure +Z.
    """
    return {
        "left_hip": np.array([0.0, 88.0, 0.0]),
        "right_hip": np.array([0.0, -88.0, 0.0]),
        "left_shoulder": np.array([0.0, 88.0, 500.0]),
        "right_shoulder": np.array([0.0, -88.0, 500.0]),
    }


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_interior_lands_on_rest_positions(mapping_path):
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(_tpose_body())

    # rest positions from pelvis.yaml, expressed in world at the T-pose:
    # pelvis local +Y -> world +Z (up), +X -> world -Y (right), origin at world 0.
    # Tolerance is physical (sub-mm): the offset ratios are anthropometric
    # estimates far below tracker noise, not exact reproductions.
    assert np.allclose(result["lumbosacral_junction"], [0.0, 0.0, 120.0], atol=0.5)
    assert np.allclose(result["left_iliac_crest"], [0.0, 88.0, 80.0], atol=0.5)
    assert np.allclose(result["right_iliac_crest"], [0.0, -88.0, 80.0], atol=0.5)
    assert np.allclose(result["pubic_symphysis"], [0.0, 0.0, -40.0], atol=0.5)


@pytest.mark.parametrize("mapping_path", _BODY_MAPPINGS)
def test_pelvis_cloud_is_non_collinear_after_hydration(mapping_path):
    """The hydrated pelvis landmarks span a plane (rank >= 2), so a Kabsch fit is
    no longer degenerate — the whole point of hydrating them."""
    mapping = TrackerMapping.from_yaml(Path(mapping_path))
    result = mapping.apply(_tpose_body())
    cloud = np.stack([
        result["hips_center"],
        result["left_hip"],
        result["right_hip"],
        result["lumbosacral_junction"],
        result["left_iliac_crest"],
        result["pubic_symphysis"],
    ])
    centered = cloud - cloud.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    # second singular value is a healthy fraction of the first -> not collinear
    assert singular_values[1] > 0.1 * singular_values[0]
