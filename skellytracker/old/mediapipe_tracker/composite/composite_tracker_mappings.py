"""
Cross-tracker index mappings used by the composite (holistic) detector to
splice higher-precision face and hand landmarks into the body pose.

These are NOT the schema of a single tracker — they relate indices between
three independent trackers (body pose, face mesh, hand) and are therefore
kept separate from the per-tracker YAML definitions.

All indices are into the raw per-tracker coordinate arrays (i.e. the order
produced by mediapipe's *Landmarker APIs), not into any composite PointCloud.
"""

# ---------------------------------------------------------------------------
# Pose body landmark indices (33 points from mediapipe PoseLandmarker)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Hand landmark indices (21 points per hand from mediapipe HandLandmarker)
# ---------------------------------------------------------------------------
HAND_WRIST_INDEX: int = 0
HAND_THUMB_CMC_INDEX: int = 1
HAND_INDEX_MCP_INDEX: int = 5
HAND_PINKY_MCP_INDEX: int = 17

# ---------------------------------------------------------------------------
# Face mesh landmark indices (478 points from mediapipe FaceLandmarker)
# ---------------------------------------------------------------------------
FACE_MESH_NOSE_TIP_INDEX: int = 1
FACE_MESH_LEFT_EAR_INDEX: int = 234
FACE_MESH_RIGHT_EAR_INDEX: int = 454
FACE_MESH_MOUTH_LEFT_INDEX: int = 61
FACE_MESH_MOUTH_RIGHT_INDEX: int = 291

FACE_MESH_LEFT_EYE_INNER_INDEX: int = 133   # left tear duct
FACE_MESH_LEFT_EYE_OUTER_INDEX: int = 33
FACE_MESH_RIGHT_EYE_INNER_INDEX: int = 362  # right tear duct
FACE_MESH_RIGHT_EYE_OUTER_INDEX: int = 263

LEFT_IRIS_INDICES: list[int] = [474, 475, 476, 477]
RIGHT_IRIS_INDICES: list[int] = [469, 470, 471, 472]

# ---------------------------------------------------------------------------
# Fusion maps
# ---------------------------------------------------------------------------

# Direct face→pose replacement: {pose_body_index: face_mesh_index}
FACE_TO_POSE_DIRECT_MAP: dict[int, int] = {
    POSE_NOSE_INDEX: FACE_MESH_NOSE_TIP_INDEX,
    POSE_LEFT_EYE_INNER_INDEX: FACE_MESH_LEFT_EYE_INNER_INDEX,
    POSE_LEFT_EYE_OUTER_INDEX: FACE_MESH_LEFT_EYE_OUTER_INDEX,
    POSE_RIGHT_EYE_INNER_INDEX: FACE_MESH_RIGHT_EYE_INNER_INDEX,
    POSE_RIGHT_EYE_OUTER_INDEX: FACE_MESH_RIGHT_EYE_OUTER_INDEX,
    POSE_LEFT_EAR_INDEX: FACE_MESH_LEFT_EAR_INDEX,
    POSE_RIGHT_EAR_INDEX: FACE_MESH_RIGHT_EAR_INDEX,
    POSE_MOUTH_LEFT_INDEX: FACE_MESH_MOUTH_LEFT_INDEX,
    POSE_MOUTH_RIGHT_INDEX: FACE_MESH_MOUTH_RIGHT_INDEX,
}

# Iris centroid replacement: {pose_body_index: iris_contour_indices}
IRIS_TO_POSE_MAP: dict[int, list[int]] = {
    POSE_LEFT_EYE_INDEX: LEFT_IRIS_INDICES,
    POSE_RIGHT_EYE_INDEX: RIGHT_IRIS_INDICES,
}

# Hand→pose finger-base replacement: {pose_body_index: hand_landmark_index}
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

# Wrist averaging — body wrist is averaged with hand wrist rather than replaced
POSE_LEFT_WRIST_FUSE_WITH_HAND_WRIST: tuple[int, int] = (POSE_LEFT_WRIST_INDEX, HAND_WRIST_INDEX)
POSE_RIGHT_WRIST_FUSE_WITH_HAND_WRIST: tuple[int, int] = (POSE_RIGHT_WRIST_INDEX, HAND_WRIST_INDEX)
