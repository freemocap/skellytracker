"""A mean / weighted landmark with a missing input is omitted, not relabelled.

A mean (or weighted sum) whose inputs are incomplete would silently relabel a different
point, so a missing input omits the landmark rather than producing it from survivors.
"""

from __future__ import annotations

import numpy as np

from skellytracker.core.io.tracker_mapping import TrackerMapping


def test_full_mean_produces_the_midpoint() -> None:
    mapping = TrackerMapping(entries={"midpoint": ["left", "right"]})
    frame = {
        "left": np.array([0.0, 90.0, 0.0]),
        "right": np.array([0.0, -90.0, 0.0]),
    }
    result = mapping.apply(frame)
    assert np.allclose(result["midpoint"], [0.0, 0.0, 0.0])


def test_partial_mean_is_omitted_not_relabelled() -> None:
    mapping = TrackerMapping(entries={"midpoint": ["left", "right"], "left_copy": "left"})
    frame = {"left": np.array([0.0, 90.0, 0.0])}  # right occluded this frame
    result = mapping.apply(frame)
    # midpoint (the mean of left and right) cannot be computed from one input:
    assert "midpoint" not in result
    # but the direct 1:1 passthrough of the visible input still hydrates:
    assert np.allclose(result["left_copy"], [0.0, 90.0, 0.0])
