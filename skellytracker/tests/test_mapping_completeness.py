"""The Phase-B completeness contract, per tracker family.

Every tracker family (body + hand mappings together) must produce every
keypoint the standard human's model declares — a gap fails at load, not at
solve time. The hand mapping is authored side-agnostically (it names `wrist`,
`thumb_cmc`, …); the per-side `left_`/`right_` names are applied at
instantiation time by the consumer, exactly as the segment parts instantiate per
side — so each hand name must be produced under both side prefixes. And a
mapping referencing a keypoint the tracker NEVER produces raises at
construction; a keypoint missing this frame (occlusion) is still skipped
silently at apply time.
"""

from pathlib import Path

import pytest

from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import (
    MediapipePoseKeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import (
    MediapipeHandKeypointDetector,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose.body.rtmpose_body_detector import (
    RTMPoseBodyDetector,
)
from skellytracker.core.detectors.keypoint_detectors.rtmpose.hand.rtmpose_hand_detector import (
    RTMPoseHandDetector,
)
from skellytracker.core.io.tracker_mapping import TrackerMapping

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "standard_human_required_keypoints.txt"
)

TRACKER_FAMILIES = {
    "rtmpose": (RTMPoseBodyDetector, RTMPoseHandDetector),
    "mediapipe": (MediapipePoseKeypointDetector, MediapipeHandKeypointDetector),
}


def _required_keypoints() -> set[str]:
    """The model's required-keypoint set — a golden fixture (see its header for
    the regeneration command; skellytracker's tests cannot import skellyforge).
    """
    return {
        line.strip()
        for line in _FIXTURE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@pytest.mark.parametrize("family", list(TRACKER_FAMILIES))
def test_every_tracker_family_produces_the_full_required_keypoint_set(family):
    body_detector, hand_detector = TRACKER_FAMILIES[family]
    produced: set[str] = set()
    body_mapping = TrackerMapping.from_yaml(
        body_detector.standard_human_mapping_path()
    )
    produced |= set(body_mapping.keypoint_names)
    # The hand mapping is authored side-agnostically: its 21 names instantiate
    # per side, exactly as the segment parts do — the contract requires both.
    hand_mapping = TrackerMapping.from_yaml(
        hand_detector.standard_human_mapping_path()
    )
    for hand_name in hand_mapping.keypoint_names:
        produced.add(f"left_{hand_name}")
        produced.add(f"right_{hand_name}")
    missing = _required_keypoints() - produced
    assert not missing, f"{family} is missing {sorted(missing)}"


def test_a_mapping_referencing_an_unproduced_keypoint_raises_at_load():
    with pytest.raises(ValueError, match="never produces"):
        TrackerMapping(
            entries={"shoulder": "ghost_keypoint"},
            known_tracker_keypoints={"nose"},
        )


def test_occlusion_is_still_a_silent_skip():
    # D24's other half: a keypoint missing THIS FRAME is data, not an error.
    mapping = TrackerMapping(
        entries={"shoulder": "left_shoulder"},
        known_tracker_keypoints={"left_shoulder"},
    )
    assert mapping.apply({}) == {}
