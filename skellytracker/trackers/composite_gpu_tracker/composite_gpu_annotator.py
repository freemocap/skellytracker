from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseImageAnnotator,
    BaseImageAnnotatorConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_config import (
    CompositeGPUImageAnnotatorConfig,
)
from skellytracker.trackers.composite_gpu_tracker.composite_gpu_observation import (
    CompositeGPUObservation,
)
from skellytracker.trackers.composite_gpu_tracker.names_and_connections import (
    RTMO_HYBRID_DEFINITION,
)

# Slice boundaries from the composition: body(0:17), right_hand(17:38),
# left_hand(38:59), face(59:165)
_BODY_START = 0
_BODY_END = 17
_RHAND_START = 17
_RHAND_END = 38
_LHAND_START = 38
_LHAND_END = 59
_FACE_START = 59

# BGR colors
BODY_COLOR = (0, 255, 0)     # green
RIGHT_HAND_COLOR = (0, 0, 255)  # red (BGR)
LEFT_HAND_COLOR = (255, 0, 0)   # blue (BGR)
FACE_COLOR = (0, 255, 255)   # yellow


@dataclass
class CompositeGPUImageAnnotator(BaseImageAnnotator):
    config: CompositeGPUImageAnnotatorConfig
    observations: list[CompositeGPUObservation] = field(default_factory=list)

    _connection_indices: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: RTMO_HYBRID_DEFINITION.connection_indices(),
        init=False,
        repr=False,
    )

    @classmethod
    def create(cls, config: CompositeGPUImageAnnotatorConfig | None = None) -> "CompositeGPUImageAnnotator":
        if config is None:
            config = CompositeGPUImageAnnotatorConfig()
        return cls(config=config)

    def annotate_image(
        self,
        image: NDArray[np.uint8],
        observation: CompositeGPUObservation | None = None,
    ) -> NDArray[np.uint8]:
        if observation is None:
            return image

        annotated = image.copy()
        pts_2d = observation.points.xy  # (127, 2)
        visibility = observation.points.visibility  # (127,)

        # --- Draw skeleton connections, color-coded by component ---
        for start_idx, end_idx in self._connection_indices:
            pt1 = pts_2d[start_idx]
            pt2 = pts_2d[end_idx]
            if np.isnan(pt1).any() or np.isnan(pt2).any():
                continue

            if start_idx < _BODY_END:
                color = BODY_COLOR
            elif start_idx < _RHAND_END:
                color = RIGHT_HAND_COLOR
            elif start_idx < _LHAND_END:
                color = LEFT_HAND_COLOR
            else:
                color = FACE_COLOR

            cv2.line(annotated,
                     (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])),
                     color, 2)

        # --- Draw keypoints ---
        for i in range(len(pts_2d)):
            if visibility[i] < 0.3 or np.isnan(pts_2d[i]).any():
                continue
            if i < _BODY_END:
                color = BODY_COLOR
            elif i < _RHAND_END:
                color = RIGHT_HAND_COLOR
            elif i < _LHAND_END:
                color = LEFT_HAND_COLOR
            else:
                color = FACE_COLOR
            cv2.circle(annotated, (int(pts_2d[i, 0]), int(pts_2d[i, 1])), 2, color, -1)

        # --- Draw ROI debug boxes ---
        self._draw_roi_box(annotated, observation.right_hand_roi, RIGHT_HAND_COLOR, "R Hand")
        self._draw_roi_box(annotated, observation.left_hand_roi, LEFT_HAND_COLOR, "L Hand")
        self._draw_roi_box(annotated, observation.face_roi, FACE_COLOR, "Face")

        # --- Draw wrist debug markers (body keypoint indices 9=left, 10=right) ---
        if observation.body_keypoints.shape[0] > 0:
            body = observation.body_keypoints[0]
            for wrist_idx, wrist_name, wrist_color in [
                (9, "L wrist", LEFT_HAND_COLOR),
                (10, "R wrist", RIGHT_HAND_COLOR),
            ]:
                if wrist_idx < body.shape[0]:
                    wx, wy = body[wrist_idx, 0], body[wrist_idx, 1]
                    if not np.isnan(wx) and not np.isnan(wy):
                        cv2.circle(annotated, (int(wx), int(wy)), 8, wrist_color, 2)
                        cv2.circle(annotated, (int(wx), int(wy)), 3, (255, 255, 255), -1)
                        cv2.putText(annotated, wrist_name, (int(wx) + 10, int(wy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, wrist_color, 1)

        return annotated

    @staticmethod
    def _draw_roi_box(
        image: NDArray[np.uint8],
        roi,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        if roi is None:
            return
        cv2.rectangle(
            image,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            color=color,
            thickness=2,
        )
        cv2.putText(
            image, label,
            (roi.x, roi.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
