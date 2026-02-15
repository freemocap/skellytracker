"""
Landmark names, skeleton connection topology, and fusion index mappings
for MediaPipe Tasks API pose/hand/face landmarkers.
"""

# =============================================================================
# POSE LANDMARKS (33 points) — same as BlazePose / legacy MediaPipe Pose
# =============================================================================

POSE_LANDMARK_NAMES: list[str] = [
    "body.nose",                     # 0
    "body.left_eye_inner",           # 1
    "body.left_eye",                 # 2
    "body.left_eye_outer",           # 3
    "body.right_eye_inner",          # 4
    "body.right_eye",                # 5
    "body.right_eye_outer",          # 6
    "body.left_ear",                 # 7
    "body.right_ear",                # 8
    "body.mouth_left",               # 9
    "body.mouth_right",              # 10
    "body.left_shoulder",            # 11
    "body.right_shoulder",           # 12
    "body.left_elbow",               # 13
    "body.right_elbow",              # 14
    "body.left_wrist",               # 15
    "body.right_wrist",              # 16
    "body.left_pinky",               # 17
    "body.right_pinky",              # 18
    "body.left_index",               # 19
    "body.right_index",              # 20
    "body.left_thumb",               # 21
    "body.right_thumb",              # 22
    "body.left_hip",                 # 23
    "body.right_hip",                # 24
    "body.left_knee",                # 25
    "body.right_knee",               # 26
    "body.left_ankle",               # 27
    "body.right_ankle",              # 28
    "body.left_heel",                # 29
    "body.right_heel",               # 30
    "body.left_foot_index",          # 31
    "body.right_foot_index",         # 32
]

NUM_POSE_LANDMARKS: int = 33

POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # left eye → left ear
    (0, 4), (4, 5), (5, 6), (6, 8),       # right eye → right ear
    (9, 10),                                # mouth
    (11, 12),                               # shoulders
    (11, 13), (13, 15),                     # left arm
    (12, 14), (14, 16),                     # right arm
    (15, 17), (15, 19), (15, 21),           # left wrist → fingers
    (16, 18), (16, 20), (16, 22),           # right wrist → fingers
    (17, 19), (18, 20),                     # pinky-index connections
    (11, 23), (12, 24),                     # torso
    (23, 24),                               # hips
    (23, 25), (25, 27),                     # left leg
    (24, 26), (26, 28),                     # right leg
    (27, 29), (29, 31), (27, 31),           # left foot
    (28, 30), (30, 32), (28, 32),           # right foot
]

# =============================================================================
# HAND LANDMARKS (21 points per hand)
# =============================================================================

HAND_LANDMARK_NAMES: list[str] = [
    "wrist",              # 0
    "thumb_cmc",          # 1
    "thumb_mcp",          # 2
    "thumb_ip",           # 3
    "thumb_tip",          # 4
    "index_finger_mcp",   # 5
    "index_finger_pip",   # 6
    "index_finger_dip",   # 7
    "index_finger_tip",   # 8
    "middle_finger_mcp",  # 9
    "middle_finger_pip",  # 10
    "middle_finger_dip",  # 11
    "middle_finger_tip",  # 12
    "ring_finger_mcp",    # 13
    "ring_finger_pip",    # 14
    "ring_finger_dip",    # 15
    "ring_finger_tip",    # 16
    "pinky_mcp",          # 17
    "pinky_pip",          # 18
    "pinky_dip",          # 19
    "pinky_tip",          # 20
]

NUM_HAND_LANDMARKS: int = 21

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

RIGHT_HAND_LANDMARK_NAMES: list[str] = [f"right_hand.{name}" for name in HAND_LANDMARK_NAMES]
LEFT_HAND_LANDMARK_NAMES: list[str] = [f"left_hand.{name}" for name in HAND_LANDMARK_NAMES]

# =============================================================================
# FACE LANDMARKS (478 points with irises)
# =============================================================================

NUM_FACE_LANDMARKS: int = 478
NUM_FACE_LANDMARKS_WITHOUT_IRISES: int = 468

# Face mesh contour connections (subset used for visualization)
# These are the same connection sets from mediapipe's FACEMESH_CONTOURS
# Defined here to avoid depending on the legacy solutions module

FACEMESH_LIPS: list[tuple[int, int]] = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402),
    (402, 318), (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
]

FACEMESH_LEFT_EYE: list[tuple[int, int]] = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]

FACEMESH_RIGHT_EYE: list[tuple[int, int]] = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
]

FACEMESH_LEFT_EYEBROW: list[tuple[int, int]] = [
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
]

FACEMESH_RIGHT_EYEBROW: list[tuple[int, int]] = [
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
]

FACEMESH_FACE_OVAL: list[tuple[int, int]] = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

FACEMESH_LEFT_IRIS: list[tuple[int, int]] = [
    (474, 475), (475, 476), (476, 477), (477, 474),
]

FACEMESH_RIGHT_IRIS: list[tuple[int, int]] = [
    (469, 470), (470, 471), (471, 472), (472, 469),
]

# All face contour connections combined
FACEMESH_CONTOURS: list[tuple[int, int]] = (
    FACEMESH_LIPS
    + FACEMESH_LEFT_EYE
    + FACEMESH_RIGHT_EYE
    + FACEMESH_LEFT_EYEBROW
    + FACEMESH_RIGHT_EYEBROW
    + FACEMESH_FACE_OVAL
    + FACEMESH_LEFT_IRIS
    + FACEMESH_RIGHT_IRIS
)

# Iris landmark indices
LEFT_IRIS_INDICES: list[int] = [474, 475, 476, 477]
RIGHT_IRIS_INDICES: list[int] = [469, 470, 471, 472]

# =============================================================================
# FUSION MAPPINGS: face mesh index → pose landmark index
#
# Maps face mesh landmark indices to the corresponding pose body landmark
# indices, for splicing higher-precision face data into the body skeleton.
# =============================================================================

# Face mesh indices for key facial features
FACE_MESH_NOSE_TIP_INDEX: int = 1  # tip of nose
FACE_MESH_LEFT_EAR_INDEX: int = 234
FACE_MESH_RIGHT_EAR_INDEX: int = 454
FACE_MESH_MOUTH_LEFT_INDEX: int = 61
FACE_MESH_MOUTH_RIGHT_INDEX: int = 291

# Tear duct (inner eye corner) and outer eye corner indices
FACE_MESH_LEFT_EYE_INNER_INDEX: int = 133   # left tear duct (from camera's perspective: right side of image)
FACE_MESH_LEFT_EYE_OUTER_INDEX: int = 33    # left outer eye corner
FACE_MESH_RIGHT_EYE_INNER_INDEX: int = 362  # right tear duct
FACE_MESH_RIGHT_EYE_OUTER_INDEX: int = 263  # right outer eye corner

# Pose body landmark indices for head/face points
POSE_NOSE_INDEX: int = 0
POSE_LEFT_EYE_INNER_INDEX: int = 1
POSE_LEFT_EYE_INDEX: int = 2
POSE_LEFT_EYE_OUTER_INDEX: int = 3
POSE_RIGHT_EYE_INNER_INDEX: int = 4
POSE_RIGHT_EYE_INDEX: int = 5
POSE_RIGHT_EYE_OUTER_INDEX: int = 6
POSE_LEFT_EAR_INDEX: int = 7
POSE_RIGHT_EAR_INDEX: int = 8
POSE_MOUTH_LEFT_INDEX: int = 9
POSE_MOUTH_RIGHT_INDEX: int = 10

# Pose body landmark indices for wrist/hand points
POSE_LEFT_ELBOW_INDEX: int = 13
POSE_RIGHT_ELBOW_INDEX: int = 14
POSE_LEFT_WRIST_INDEX: int = 15
POSE_RIGHT_WRIST_INDEX: int = 16
POSE_LEFT_PINKY_INDEX: int = 17
POSE_RIGHT_PINKY_INDEX: int = 18
POSE_LEFT_INDEX_INDEX: int = 19
POSE_RIGHT_INDEX_INDEX: int = 20
POSE_LEFT_THUMB_INDEX: int = 21
POSE_RIGHT_THUMB_INDEX: int = 22

# Hand landmark indices used for fusion with body
HAND_WRIST_INDEX: int = 0
HAND_PINKY_MCP_INDEX: int = 17
HAND_INDEX_MCP_INDEX: int = 5
HAND_THUMB_CMC_INDEX: int = 1

# Direct face→pose replacement mapping:
# {pose_body_index: face_mesh_index}
FACE_TO_POSE_DIRECT_MAP: dict[int, int] = {
    POSE_NOSE_INDEX: FACE_MESH_NOSE_TIP_INDEX,
    POSE_LEFT_EYE_INNER_INDEX: FACE_MESH_LEFT_EYE_INNER_INDEX,   # tear duct
    POSE_LEFT_EYE_OUTER_INDEX: FACE_MESH_LEFT_EYE_OUTER_INDEX,   # outer lid corner
    POSE_RIGHT_EYE_INNER_INDEX: FACE_MESH_RIGHT_EYE_INNER_INDEX, # tear duct
    POSE_RIGHT_EYE_OUTER_INDEX: FACE_MESH_RIGHT_EYE_OUTER_INDEX, # outer lid corner
    POSE_LEFT_EAR_INDEX: FACE_MESH_LEFT_EAR_INDEX,
    POSE_RIGHT_EAR_INDEX: FACE_MESH_RIGHT_EAR_INDEX,
    POSE_MOUTH_LEFT_INDEX: FACE_MESH_MOUTH_LEFT_INDEX,
    POSE_MOUTH_RIGHT_INDEX: FACE_MESH_MOUTH_RIGHT_INDEX,
}

# Pose eye indices that should be replaced by iris centroid (mean of iris contour)
# {pose_body_index: list of iris contour face mesh indices}
IRIS_TO_POSE_MAP: dict[int, list[int]] = {
    POSE_LEFT_EYE_INDEX: LEFT_IRIS_INDICES,    # left eye → mean of left iris contour (pupil center)
    POSE_RIGHT_EYE_INDEX: RIGHT_IRIS_INDICES,  # right eye → mean of right iris contour (pupil center)
}

# Hand→pose replacement mapping (averaged with body wrist for wrist, direct for finger bases):
# {pose_body_index: hand_landmark_index}
LEFT_HAND_TO_POSE_MAP: dict[int, int] = {
    POSE_LEFT_PINKY_INDEX: HAND_PINKY_MCP_INDEX,
    POSE_LEFT_INDEX_INDEX: HAND_INDEX_MCP_INDEX,
    POSE_LEFT_THUMB_INDEX: HAND_THUMB_CMC_INDEX,
}

RIGHT_HAND_TO_POSE_MAP: dict[int, int] = {
    POSE_RIGHT_PINKY_INDEX: HAND_PINKY_MCP_INDEX,
    POSE_RIGHT_INDEX_INDEX: HAND_INDEX_MCP_INDEX,
    POSE_RIGHT_THUMB_INDEX: HAND_THUMB_CMC_INDEX,
}

# Wrist indices — these are averaged (mean of body wrist + hand wrist)
# rather than replaced directly
POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST: tuple[int, int] = (POSE_LEFT_WRIST_INDEX, HAND_WRIST_INDEX)
POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST: tuple[int, int] = (POSE_RIGHT_WRIST_INDEX, HAND_WRIST_INDEX)
