"""Minimal skeleton visualization — vendored from rtmlib to remove the dependency.

Only supports mmpose-style skeletons (coco17, coco133, hand21) since those
are the only ones ever used by skellytracker at runtime.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Skeleton definitions (extracted from rtmlib/visualization/skeleton/)
# ---------------------------------------------------------------------------

_COCO17_SKELETON = {
    "keypoint_info": {
        0: {"name": "nose", "id": 0, "color": [51, 153, 255]},
        1: {"name": "left_eye", "id": 1, "color": [51, 153, 255]},
        2: {"name": "right_eye", "id": 2, "color": [51, 153, 255]},
        3: {"name": "left_ear", "id": 3, "color": [51, 153, 255]},
        4: {"name": "right_ear", "id": 4, "color": [51, 153, 255]},
        5: {"name": "left_shoulder", "id": 5, "color": [0, 255, 0]},
        6: {"name": "right_shoulder", "id": 6, "color": [255, 128, 0]},
        7: {"name": "left_elbow", "id": 7, "color": [0, 255, 0]},
        8: {"name": "right_elbow", "id": 8, "color": [255, 128, 0]},
        9: {"name": "left_wrist", "id": 9, "color": [0, 255, 0]},
        10: {"name": "right_wrist", "id": 10, "color": [255, 128, 0]},
        11: {"name": "left_hip", "id": 11, "color": [0, 255, 0]},
        12: {"name": "right_hip", "id": 12, "color": [255, 128, 0]},
        13: {"name": "left_knee", "id": 13, "color": [0, 255, 0]},
        14: {"name": "right_knee", "id": 14, "color": [255, 128, 0]},
        15: {"name": "left_ankle", "id": 15, "color": [0, 255, 0]},
        16: {"name": "right_ankle", "id": 16, "color": [255, 128, 0]},
    },
    "skeleton_info": {
        0: {"link": ("left_ankle", "left_knee"), "id": 0, "color": [0, 255, 0]},
        1: {"link": ("left_knee", "left_hip"), "id": 1, "color": [0, 255, 0]},
        2: {"link": ("right_ankle", "right_knee"), "id": 2, "color": [255, 128, 0]},
        3: {"link": ("right_knee", "right_hip"), "id": 3, "color": [255, 128, 0]},
        4: {"link": ("left_hip", "right_hip"), "id": 4, "color": [51, 153, 255]},
        5: {"link": ("left_shoulder", "left_hip"), "id": 5, "color": [51, 153, 255]},
        6: {"link": ("right_shoulder", "right_hip"), "id": 6, "color": [51, 153, 255]},
        7: {"link": ("left_shoulder", "right_shoulder"), "id": 7, "color": [51, 153, 255]},
        8: {"link": ("left_shoulder", "left_elbow"), "id": 8, "color": [0, 255, 0]},
        9: {"link": ("right_shoulder", "right_elbow"), "id": 9, "color": [255, 128, 0]},
        10: {"link": ("left_elbow", "left_wrist"), "id": 10, "color": [0, 255, 0]},
        11: {"link": ("right_elbow", "right_wrist"), "id": 11, "color": [255, 128, 0]},
        12: {"link": ("left_eye", "right_eye"), "id": 12, "color": [51, 153, 255]},
        13: {"link": ("nose", "left_eye"), "id": 13, "color": [51, 153, 255]},
        14: {"link": ("nose", "right_eye"), "id": 14, "color": [51, 153, 255]},
        15: {"link": ("left_eye", "left_ear"), "id": 15, "color": [51, 153, 255]},
        16: {"link": ("right_eye", "right_ear"), "id": 16, "color": [51, 153, 255]},
        17: {"link": ("left_ear", "left_shoulder"), "id": 17, "color": [51, 153, 255]},
        18: {"link": ("right_ear", "right_shoulder"), "id": 18, "color": [51, 153, 255]},
    },
}

_HAND21_SKELETON = {
    "keypoint_info": {
        0: {"name": "wrist", "id": 0, "color": [255, 255, 255]},
        1: {"name": "thumb1", "id": 1, "color": [255, 128, 0]},
        2: {"name": "thumb2", "id": 2, "color": [255, 128, 0]},
        3: {"name": "thumb3", "id": 3, "color": [255, 128, 0]},
        4: {"name": "thumb4", "id": 4, "color": [255, 128, 0]},
        5: {"name": "forefinger1", "id": 5, "color": [255, 153, 255]},
        6: {"name": "forefinger2", "id": 6, "color": [255, 153, 255]},
        7: {"name": "forefinger3", "id": 7, "color": [255, 153, 255]},
        8: {"name": "forefinger4", "id": 8, "color": [255, 153, 255]},
        9: {"name": "middle_finger1", "id": 9, "color": [102, 178, 255]},
        10: {"name": "middle_finger2", "id": 10, "color": [102, 178, 255]},
        11: {"name": "middle_finger3", "id": 11, "color": [102, 178, 255]},
        12: {"name": "middle_finger4", "id": 12, "color": [102, 178, 255]},
        13: {"name": "ring_finger1", "id": 13, "color": [255, 51, 51]},
        14: {"name": "ring_finger2", "id": 14, "color": [255, 51, 51]},
        15: {"name": "ring_finger3", "id": 15, "color": [255, 51, 51]},
        16: {"name": "ring_finger4", "id": 16, "color": [255, 51, 51]},
        17: {"name": "pinky_finger1", "id": 17, "color": [0, 255, 0]},
        18: {"name": "pinky_finger2", "id": 18, "color": [0, 255, 0]},
        19: {"name": "pinky_finger3", "id": 19, "color": [0, 255, 0]},
        20: {"name": "pinky_finger4", "id": 20, "color": [0, 255, 0]},
    },
    "skeleton_info": {
        0: {"link": ("wrist", "thumb1"), "id": 0, "color": [255, 128, 0]},
        1: {"link": ("thumb1", "thumb2"), "id": 1, "color": [255, 128, 0]},
        2: {"link": ("thumb2", "thumb3"), "id": 2, "color": [255, 128, 0]},
        3: {"link": ("thumb3", "thumb4"), "id": 3, "color": [255, 128, 0]},
        4: {"link": ("wrist", "forefinger1"), "id": 4, "color": [255, 153, 255]},
        5: {"link": ("forefinger1", "forefinger2"), "id": 5, "color": [255, 153, 255]},
        6: {"link": ("forefinger2", "forefinger3"), "id": 6, "color": [255, 153, 255]},
        7: {"link": ("forefinger3", "forefinger4"), "id": 7, "color": [255, 153, 255]},
        8: {"link": ("wrist", "middle_finger1"), "id": 8, "color": [102, 178, 255]},
        9: {"link": ("middle_finger1", "middle_finger2"), "id": 9, "color": [102, 178, 255]},
        10: {"link": ("middle_finger2", "middle_finger3"), "id": 10, "color": [102, 178, 255]},
        11: {"link": ("middle_finger3", "middle_finger4"), "id": 11, "color": [102, 178, 255]},
        12: {"link": ("wrist", "ring_finger1"), "id": 12, "color": [255, 51, 51]},
        13: {"link": ("ring_finger1", "ring_finger2"), "id": 13, "color": [255, 51, 51]},
        14: {"link": ("ring_finger2", "ring_finger3"), "id": 14, "color": [255, 51, 51]},
        15: {"link": ("ring_finger3", "ring_finger4"), "id": 15, "color": [255, 51, 51]},
        16: {"link": ("wrist", "pinky_finger1"), "id": 16, "color": [0, 255, 0]},
        17: {"link": ("pinky_finger1", "pinky_finger2"), "id": 17, "color": [0, 255, 0]},
        18: {"link": ("pinky_finger2", "pinky_finger3"), "id": 18, "color": [0, 255, 0]},
        19: {"link": ("pinky_finger3", "pinky_finger4"), "id": 19, "color": [0, 255, 0]},
    },
}

# ---------------------------------------------------------------------------
# Minimal coco133 — body (23) + right_hand (21) + left_hand (21) + face (68)
# ---------------------------------------------------------------------------

_COCO133_KEYPOINT_COLORS: list[list[int]] = [
    # body 0-22 (23 points)
    [51, 153, 255], [51, 153, 255], [51, 153, 255],  # nose, left_eye, right_eye
    [51, 153, 255], [51, 153, 255],                    # left_ear, right_ear
    [0, 255, 0], [255, 128, 0],                        # left_shoulder, right_shoulder
    [0, 255, 0], [255, 128, 0],                        # left_elbow, right_elbow
    [0, 255, 0], [255, 128, 0],                        # left_wrist, right_wrist
    [0, 255, 0], [255, 128, 0],                        # left_hip, right_hip
    [0, 255, 0], [255, 128, 0],                        # left_knee, right_knee
    [0, 255, 0], [255, 128, 0],                        # left_ankle, right_ankle
    [255, 128, 0], [255, 128, 0], [255, 128, 0],       # left_big_toe, left_small_toe, left_heel
    [255, 128, 0], [255, 128, 0], [255, 128, 0],       # right_big_toe, right_small_toe, right_heel
    # face 23-90 (68 points)
    *[[255, 255, 255]] * 68,
    # left_hand 91-111 (21 points)
    [255, 255, 255], *[[255, 128, 0]] * 4,              # root, thumb1-4
    *[[255, 153, 255]] * 4, *[[102, 178, 255]] * 4,     # forefinger, middle
    *[[255, 51, 51]] * 4, *[[0, 255, 0]] * 4,           # ring, pinky
    # right_hand 112-132 (21 points)
    [255, 255, 255], *[[255, 128, 0]] * 4,              # root, thumb1-4
    *[[255, 153, 255]] * 4, *[[102, 178, 255]] * 4,     # forefinger, middle
    *[[255, 51, 51]] * 4, *[[0, 255, 0]] * 4,           # ring, pinky
]

_COCO133_SKELETON_LINKS: list[tuple[int, int, list[int]]] = [
    # Body (0-22)
    (15, 13, [0, 255, 0]), (13, 11, [0, 255, 0]), (16, 14, [255, 128, 0]),
    (14, 12, [255, 128, 0]), (11, 12, [51, 153, 255]),
    (5, 11, [51, 153, 255]), (6, 12, [51, 153, 255]), (5, 6, [51, 153, 255]),
    (5, 7, [0, 255, 0]), (6, 8, [255, 128, 0]),
    (7, 9, [0, 255, 0]), (8, 10, [255, 128, 0]),
    (1, 2, [51, 153, 255]), (0, 1, [51, 153, 255]), (0, 2, [51, 153, 255]),
    (1, 3, [51, 153, 255]), (2, 4, [51, 153, 255]),
    (3, 5, [51, 153, 255]), (4, 6, [51, 153, 255]),
    (15, 17, [0, 255, 0]), (15, 18, [0, 255, 0]), (15, 19, [0, 255, 0]),
    (16, 20, [255, 128, 0]), (16, 21, [255, 128, 0]), (16, 22, [255, 128, 0]),
    # Face (23-90) — skip detailed links; face contour is 68 dense points
    *[(i, i + 1, [255, 255, 255]) for i in range(23, 90)],
    # Left hand (91-111)
    (91, 92, [255, 128, 0]), (92, 93, [255, 128, 0]), (93, 94, [255, 128, 0]), (94, 95, [255, 128, 0]),
    (91, 96, [255, 153, 255]), (96, 97, [255, 153, 255]), (97, 98, [255, 153, 255]), (98, 99, [255, 153, 255]),
    (91, 100, [102, 178, 255]), (100, 101, [102, 178, 255]), (101, 102, [102, 178, 255]), (102, 103, [102, 178, 255]),
    (91, 104, [255, 51, 51]), (104, 105, [255, 51, 51]), (105, 106, [255, 51, 51]), (106, 107, [255, 51, 51]),
    (91, 108, [0, 255, 0]), (108, 109, [0, 255, 0]), (109, 110, [0, 255, 0]), (110, 111, [0, 255, 0]),
    # Right hand (112-132)
    (112, 113, [255, 128, 0]), (113, 114, [255, 128, 0]), (114, 115, [255, 128, 0]), (115, 116, [255, 128, 0]),
    (112, 117, [255, 153, 255]), (117, 118, [255, 153, 255]), (118, 119, [255, 153, 255]), (119, 120, [255, 153, 255]),
    (112, 121, [102, 178, 255]), (121, 122, [102, 178, 255]), (122, 123, [102, 178, 255]), (123, 124, [102, 178, 255]),
    (112, 125, [255, 51, 51]), (125, 126, [255, 51, 51]), (126, 127, [255, 51, 51]), (127, 128, [255, 51, 51]),
    (112, 129, [0, 255, 0]), (129, 130, [0, 255, 0]), (130, 131, [0, 255, 0]), (131, 132, [0, 255, 0]),
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_skeleton(
    img: NDArray[np.uint8],
    keypoints: NDArray,
    scores: NDArray,
    kpt_thr: float = 0.5,
    radius: int = 2,
    line_width: int = 2,
) -> NDArray[np.uint8]:
    """Draw skeleton keypoints and connections on *img* (mmpose-style).

    Skeleton type is inferred from ``keypoints.shape[1]``:
    17 → coco17, 21 → hand21, 133 → coco133.
    """
    n_kpt = keypoints.shape[1]

    if n_kpt == 17:
        skel = _COCO17_SKELETON
    elif n_kpt == 21:
        skel = _HAND21_SKELETON
    elif n_kpt == 133:
        return _draw_coco133(img, keypoints, scores, kpt_thr, radius, line_width)
    else:
        raise NotImplementedError(f"No skeleton for {n_kpt} keypoints")

    return _draw_mmpose(img, keypoints, scores, skel, kpt_thr, radius, line_width)


def _draw_mmpose(
    img: NDArray[np.uint8],
    keypoints: NDArray,
    scores: NDArray,
    skeleton: dict,
    kpt_thr: float,
    radius: int,
    line_width: int,
) -> NDArray[np.uint8]:
    keypoint_info = skeleton["keypoint_info"]
    skeleton_info = skeleton["skeleton_info"]

    if keypoints.ndim == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :]

    for inst in range(keypoints.shape[0]):
        k = keypoints[inst]
        s = scores[inst]
        vis = [sc >= kpt_thr for sc in s]

        # Build name → index map
        name_to_idx: dict[str, int] = {}
        for i, info in keypoint_info.items():
            name_to_idx[info["name"]] = info["id"]
            if vis[i]:
                color = tuple(info["color"])
                cv2.circle(img, (int(k[i][0]), int(k[i][1])), radius, color, -1)

        for _, info in skeleton_info.items():
            n0, n1 = info["link"]
            pi0, pi1 = name_to_idx[n0], name_to_idx[n1]
            if vis[pi0] and vis[pi1]:
                color = tuple(info["color"])
                cv2.line(
                    img,
                    (int(k[pi0][0]), int(k[pi0][1])),
                    (int(k[pi1][0]), int(k[pi1][1])),
                    color,
                    thickness=line_width,
                )

    return img


def _draw_coco133(
    img: NDArray[np.uint8],
    keypoints: NDArray,
    scores: NDArray,
    kpt_thr: float,
    radius: int,
    line_width: int,
) -> NDArray[np.uint8]:
    if keypoints.ndim == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :]

    colors = _COCO133_KEYPOINT_COLORS
    links = _COCO133_SKELETON_LINKS

    for inst in range(keypoints.shape[0]):
        k = keypoints[inst]
        s = scores[inst]
        vis = [sc >= kpt_thr for sc in s]

        # Draw keypoints
        for i, c in enumerate(colors):
            if vis[i]:
                cv2.circle(img, (int(k[i][0]), int(k[i][1])), radius, tuple(c), -1)

        # Draw skeleton lines
        for pi0, pi1, c in links:
            if pi0 < len(vis) and pi1 < len(vis) and vis[pi0] and vis[pi1]:
                cv2.line(
                    img,
                    (int(k[pi0][0]), int(k[pi0][1])),
                    (int(k[pi1][0]), int(k[pi1][1])),
                    tuple(c),
                    thickness=line_width,
                )

    return img
