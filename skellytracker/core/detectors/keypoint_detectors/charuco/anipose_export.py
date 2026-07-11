from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)


@dataclass
class _AniposeCameraRow:
    framenum: tuple[int, int]
    corners: NDArray
    ids: NDArray
    filled: NDArray

    def to_dict(self) -> dict:
        return {
            "framenum": self.framenum,
            "corners": self.corners,
            "ids": self.ids,
            "filled": self.filled,
        }


def to_anipose_camera_row(
    keypoints: Keypoints,
    board_def: CharucoBoardDefinition,
    frame_number: int,
) -> dict:
    """Build an anipose-compatible camera row dict from CharucoDetector output.

    The "filled" array has shape (n_corners, 1, 2), with detected corner positions
    placed at their corresponding IDs and NaN elsewhere.
    """
    n_corners = board_def.n_corners
    all_ids = list(range(n_corners))
    nan_filled = np.full((n_corners, 1, 2), fill_value=np.nan, dtype=np.float64)

    detected_ids = []
    raw_corners_list = []

    for corner_id in all_ids:
        name = f"CharucoCorner-{corner_id}"
        if not keypoints.has_name(name):
            continue
        idx = keypoints.index_of(name)
        if keypoints.visibility[idx] > 0.0 and not np.isnan(keypoints.xyz[idx, 0]):
            xy = keypoints.xyz[idx, :2]
            nan_filled[corner_id, 0] = xy
            detected_ids.append(corner_id)
            raw_corners_list.append(xy[np.newaxis, np.newaxis, :])  # (1, 1, 2)

    if not detected_ids:
        return _AniposeCameraRow(
            framenum=(0, frame_number),
            corners=nan_filled,
            ids=np.array(all_ids),
            filled=nan_filled,
        ).to_dict()

    raw_corners = np.concatenate(raw_corners_list, axis=0)  # (n_detected, 1, 2)
    return _AniposeCameraRow(
        framenum=(0, frame_number),
        corners=raw_corners,
        ids=np.array(detected_ids),
        filled=nan_filled,
    ).to_dict()
