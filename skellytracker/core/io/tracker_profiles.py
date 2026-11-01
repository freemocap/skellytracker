"""Tracker-owned schemas and mapping selection for supported human presets."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from skellytracker.core.detectors.keypoint_detectors._schema_loader import load_connections, load_point_names
from skellytracker.core.io.mapping_paths import (
    RTMPOSE_BODY_MAPPING, RTMPOSE_HAND_MAPPING, MEDIAPIPE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING,
)
from skellytracker.core.io.composite_tracker_mapping import CompositeTrackerMapping
from skellytracker.core.io.tracker_mapping import TrackerMapping

_DETECTORS = Path(__file__).parent.parent / "detectors" / "keypoint_detectors"


@dataclass(frozen=True)
class TrackerProfile:
    name: str
    keypoint_names: tuple[str, ...]
    connections: tuple[tuple[str, str], ...]
    mapping_paths: tuple[Path, ...]
    crop_stage_paths: tuple[tuple[str, ...], ...] = ()

    def load_mapping(self) -> CompositeTrackerMapping:
        return CompositeTrackerMapping(tuple(TrackerMapping.from_yaml(path) for path in self.mapping_paths))


def mediapipe_hand_names(side: str) -> tuple[str, ...]:
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side {side!r}")
    names = load_point_names(_DETECTORS / "mediapipe/hands/mediapipe_hand.yaml")
    return tuple(f"{side}_hand_{name}" for name in names)


@lru_cache(maxsize=3)
def tracker_profile(preset: str) -> TrackerProfile:
    hand_path = _DETECTORS / "mediapipe/hands/mediapipe_hand.yaml"
    hand_names = tuple(name for side in ("right", "left") for name in mediapipe_hand_names(side))
    hand_edges = tuple((f"{side}_hand_{a}", f"{side}_hand_{b}")
                       for side in ("right", "left") for a, b in load_connections(hand_path))
    if preset in ("rtmpose", "rtmw_mediapipe_hands"):
        # Same authored public schema consumed by RTMW's native-order permutation.
        schema_path = _DETECTORS / "rtmpose/wholebody/rtmpose_wholebody.yaml"
        names, edges = load_point_names(schema_path), load_connections(schema_path)
        if preset == "rtmpose":
            return TrackerProfile(preset, names, edges, (RTMPOSE_BODY_MAPPING, RTMPOSE_HAND_MAPPING), (('body',),))
        selected = tuple(name for name in names if not name.startswith(("left_hand_", "right_hand_")))
        selected_set = set(selected)
        edges = tuple((a, b) for a, b in edges if a in selected_set and b in selected_set)
        return TrackerProfile(preset, selected + hand_names, edges + hand_edges,
                              (RTMPOSE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING),
                              (('body',), ('body', 'right_hand'), ('body', 'left_hand')))
    if preset == "mediapipe":
        body = _DETECTORS / "mediapipe/body/mediapipe_body.yaml"
        face = _DETECTORS / "mediapipe/face/mediapipe_face_contour.yaml"
        return TrackerProfile(preset, load_point_names(body) + hand_names + load_point_names(face),
                              load_connections(body) + hand_edges + load_connections(face),
                              (MEDIAPIPE_BODY_MAPPING, MEDIAPIPE_HAND_MAPPING),
                              (('body', 'right_hand'), ('body', 'left_hand')))
    raise ValueError(f"Unknown tracker preset {preset!r}")
