from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints


@dataclass
class ArucoAnnotatorConfig:
    show_tracks: int | None = 15
    marker_lines_thickness: int = 2
    marker_lines_color: tuple[int, int, int] = (0, 255, 0)
    corner_marker_type: int = cv2.MARKER_CROSS
    corner_marker_size: int = 8
    corner_marker_thickness: int = 2
    corner_marker_color: tuple[int, int, int] = (255, 0, 255)
    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = 0.5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ArucoAnnotator:
    """Annotates images with ArUco marker outlines and IDs.

    Maintains a rolling history of past Keypoints for a track-fade effect.
    """

    config: ArucoAnnotatorConfig
    aruco_ids: tuple[int, ...]
    _history: deque[Keypoints] = field(default_factory=deque, repr=False)

    def annotate(
        self,
        image: NDArray[np.uint8],
        keypoints: Keypoints,
    ) -> NDArray[np.uint8]:
        image_height, image_width = image.shape[:2]
        text_offset = max(1, int(image_height * 0.01))
        annotated = image.copy()

        self._history.append(keypoints)
        if self.config.show_tracks is None or self.config.show_tracks < 1:
            while len(self._history) > 1:
                self._history.popleft()
        else:
            while len(self._history) > self.config.show_tracks:
                self._history.popleft()

        history_list = list(self._history)
        n_history = len(history_list)

        for obs_count, kpts in enumerate(reversed(history_list)):
            scale = 1.0 - (obs_count / n_history)
            line_color = tuple(int(c * scale) for c in self.config.marker_lines_color)
            corner_color = tuple(int(c * scale) for c in self.config.corner_marker_color)
            line_thickness = max(1, int(self.config.marker_lines_thickness * scale))
            corner_size = max(1, int(self.config.corner_marker_size * scale))
            is_latest = obs_count == 0

            for marker_id in self.aruco_ids:
                corners = self._get_marker_corners(kpts, marker_id)
                if corners is None:
                    continue

                for corner in corners:
                    cv2.drawMarker(
                        annotated,
                        corner,
                        corner_color,
                        markerType=self.config.corner_marker_type,
                        markerSize=corner_size,
                        thickness=self.config.corner_marker_thickness,
                    )

                corners_arr = np.array(corners, dtype=np.int32)
                cv2.polylines(
                    annotated,
                    [corners_arr],
                    isClosed=True,
                    color=line_color,
                    thickness=line_thickness,
                )

                if is_latest:
                    _draw_doubled_text(
                        annotated,
                        f"Aruco#{marker_id}",
                        corners[0][0] + text_offset,
                        corners[0][1] + text_offset,
                        self.config.text_font,
                        self.config.text_size,
                        self.config.text_color,
                        self.config.text_thickness,
                    )

        return annotated

    def _get_marker_corners(
        self,
        kpts: Keypoints,
        marker_id: int,
    ) -> list[tuple[int, int]] | None:
        corners = []
        for j in range(4):
            name = f"ArucoMarker-{marker_id}-corner-{j}"
            if not kpts.has_name(name):
                return None
            idx = kpts.index_of(name)
            if kpts.visibility[idx] == 0.0 or np.isnan(kpts.xyz[idx, 0]):
                return None
            corners.append((int(kpts.xyz[idx, 0]), int(kpts.xyz[idx, 1])))
        return corners


def _draw_doubled_text(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    font: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
