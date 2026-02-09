"""
names and orders taken from the RTMLib COCO133 keypoints and order: https://github.com/Tau-J/rtmlib/blob/main/rtmlib/visualization/skeleton/coco133.py
"""

# -------------------------
# Body (23)
# -------------------------

BODY_LANDMARK_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

# -------------------------
# Face (68)
# -------------------------

FACE_LANDMARK_NAMES = [
    f"face-{i}" for i in range(68)
]

# -------------------------
# Left hand (21)
# -------------------------

LEFT_HAND_LANDMARK_NAMES = [
    "left_hand_root",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_forefinger1",
    "left_forefinger2",
    "left_forefinger3",
    "left_forefinger4",
    "left_middle_finger1",
    "left_middle_finger2",
    "left_middle_finger3",
    "left_middle_finger4",
    "left_ring_finger1",
    "left_ring_finger2",
    "left_ring_finger3",
    "left_ring_finger4",
    "left_pinky_finger1",
    "left_pinky_finger2",
    "left_pinky_finger3",
    "left_pinky_finger4",
]

# -------------------------
# Right hand (21)
# -------------------------

RIGHT_HAND_LANDMARK_NAMES = [
    "right_hand_root",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_forefinger1",
    "right_forefinger2",
    "right_forefinger3",
    "right_forefinger4",
    "right_middle_finger1",
    "right_middle_finger2",
    "right_middle_finger3",
    "right_middle_finger4",
    "right_ring_finger1",
    "right_ring_finger2",
    "right_ring_finger3",
    "right_ring_finger4",
    "right_pinky_finger1",
    "right_pinky_finger2",
    "right_pinky_finger3",
    "right_pinky_finger4",
]


ALL_LANDMARK_NAMES = (
    BODY_LANDMARK_NAMES
    + FACE_LANDMARK_NAMES
    + LEFT_HAND_LANDMARK_NAMES
    + RIGHT_HAND_LANDMARK_NAMES
)

assert len(ALL_LANDMARK_NAMES) == 133
