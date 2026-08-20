"""A mean / weighted landmark with a missing input is omitted, not relabelled.

``hips_center: [left_hip, right_hip]`` must be the true midpoint or nothing. If
only one hip is visible, returning it under the ``hips_center`` name is an ~88 mm
lie with no signal -- and this landmark is the pelvis (root) origin, so the lie
displaces the whole skeleton. The rigidifier gap-fills a missing origin correctly;
it must not be handed a mislabelled one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skellytracker.core.io.mapping_paths import MEDIAPIPE_BODY_MAPPING
from skellytracker.core.io.tracker_mapping import TrackerMapping


def test_full_mean_produces_the_midpoint():
    mapping = TrackerMapping.from_yaml(Path(MEDIAPIPE_BODY_MAPPING))
    frame = {
        "left_hip": np.array([0.0, 90.0, 0.0]),
        "right_hip": np.array([0.0, -90.0, 0.0]),
    }
    result = mapping.apply(frame)
    assert np.allclose(result["hips_center"], [0.0, 0.0, 0.0])


def test_partial_mean_is_omitted_not_relabelled():
    mapping = TrackerMapping.from_yaml(Path(MEDIAPIPE_BODY_MAPPING))
    frame = {"left_hip": np.array([0.0, 90.0, 0.0])}  # right hip occluded this frame
    result = mapping.apply(frame)
    # hips_center (the mean of both hips) cannot be computed from one hip:
    assert "hips_center" not in result
    # but the direct 1:1 passthrough of the visible hip still hydrates:
    assert np.allclose(result["left_hip"], [0.0, 90.0, 0.0])
