import numpy as np

from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.io.tracker_profiles import tracker_profile, mediapipe_hand_names
from skellytracker.core.tracker.hand_tracking_stage import HandTracks


def detection(x, side):
    names = tuple(name.removeprefix("right_hand_") for name in mediapipe_hand_names("right"))
    points = np.random.default_rng(1).uniform(0, 30, (len(names), 3)) + [x, 30, 0]
    return side, Keypoints(names, points, np.ones(len(names)))


def test_independent_tracks_survive_reordering_occlusion_and_duplicate_sides():
    tracks = HandTracks()
    a, b, c = [detection(x, side) for x, side in [(10, "right"), (300, "left"), (600, "right")]]
    assert [h.track_id for h in tracks.update([a, b, c])] == [0, 1, 2]
    assert [h.track_id for h in tracks.update([c, a])] == [2, 0]
    hands = tracks.update([b, c, a])
    assert [(h.track_id, h.handedness) for h in hands] == [(1, "left"), (2, "right"), (0, "right")]
    flipped = tracks.update([("left", a[1])])
    assert (flipped[0].track_id, flipped[0].handedness) == (0, "right")
    profile = tracker_profile("mediapipe_hands")
    mapping = profile.load_mapping()
    for hand in hands:
        points = hand.sided_keypoints()
        edges = [edge for edge in profile.connections if edge[0] in points.names]
        assert edges and all(b in points.names for _, b in edges)
        landmarks = mapping.apply(points.to_named_dict(3))
        np.testing.assert_array_equal(landmarks[f"{hand.handedness}_carpal_origin"], hand.keypoints.xyz[0])
        assert f"{hand.handedness}_index_cmc" in landmarks
        sided_mapping = profile.hand_mapping(hand.handedness)
        snapshot, = sided_mapping.mapping_snapshots()
        assert all(name.startswith(f"{hand.handedness}_") for name in snapshot.entries)
        for name, value in sided_mapping.apply(points.to_named_dict(3)).items():
            np.testing.assert_array_equal(value, landmarks[name])


def test_crossing_right_hands_retain_ids_when_detection_order_reverses():
    tracks = HandTracks()
    for i, (x_a, x_b) in enumerate(((20, 132), (40, 112), (60, 92), (80, 72), (100, 52))):
        points = [detection(x_a, "right"), detection(x_b, "right")]
        if i >= 3:
            points.reverse()
        ids = [hand.track_id for hand in tracks.update(points, i * 33)]
        assert ids == ([1, 0] if i >= 3 else [0, 1])
