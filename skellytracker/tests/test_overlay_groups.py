"""Tests for generic overlay-group resolution."""

from __future__ import annotations

from skellytracker.core.detectors.overlay_groups import resolve_overlay_groups
from skellytracker.core.sidecar.model import OverlaySpec


def _overlay(**groups) -> OverlaySpec:
    return OverlaySpec.model_validate({"groups": groups} if groups else {})


class TestResolveOverlayGroups:
    def test_no_groups_returns_empty(self):
        overlay = _overlay()
        assert resolve_overlay_groups(overlay, (("a", "b"),)) == {}

    def test_prefix_matching_and_default(self):
        overlay = _overlay(
            right_hand={"prefix": "right_hand_", "connection_color": [0, 100, 255]},
            left_hand={"prefix": "left_hand_", "connection_color": [255, 100, 0]},
            body={"connection_color": [0, 200, 100]},
        )
        edges = (
            ("nose", "left_eye"),
            ("right_hand_root", "right_hand_thumb1"),
            ("left_hand_root", "left_hand_thumb1"),
        )
        result = resolve_overlay_groups(overlay, edges)
        assert result["right_hand"] == (("right_hand_root", "right_hand_thumb1"),)
        assert result["left_hand"] == (("left_hand_root", "left_hand_thumb1"),)
        assert result["body"] == (("nose", "left_eye"),)

    def test_explicit_connections_take_priority_over_prefix(self):
        overlay = _overlay(
            special={
                "connections": [["right_hand_root", "right_hand_thumb1"]],
                "connection_color": [1, 2, 3],
            },
            right_hand={"prefix": "right_hand_", "connection_color": [0, 100, 255]},
        )
        edges = (
            ("right_hand_root", "right_hand_thumb1"),
            ("right_hand_thumb1", "right_hand_thumb2"),
        )
        result = resolve_overlay_groups(overlay, edges)
        assert result["special"] == (("right_hand_root", "right_hand_thumb1"),)
        assert result["right_hand"] == (("right_hand_thumb1", "right_hand_thumb2"),)

    def test_explicit_connections_match_reversed_edge(self):
        overlay = _overlay(
            special={
                "connections": [["b", "a"]],
                "connection_color": [1, 2, 3],
            },
        )
        result = resolve_overlay_groups(overlay, (("a", "b"),))
        assert result["special"] == (("a", "b"),)

    def test_unmatched_edge_dropped_when_no_default_group(self):
        overlay = _overlay(
            right_hand={"prefix": "right_hand_", "connection_color": [0, 100, 255]}
        )
        result = resolve_overlay_groups(overlay, (("nose", "left_eye"),))
        assert result["right_hand"] == ()
