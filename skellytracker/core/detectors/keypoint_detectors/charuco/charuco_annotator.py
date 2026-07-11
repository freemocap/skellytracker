from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import (
    CharucoBoardDefinition,
)


@dataclass
class CharucoAnnotatorConfig:
    show_tracks: int | None = 15
    corner_marker_type: int = cv2.MARKER_DIAMOND
    corner_marker_size: int = 10
    corner_marker_thickness: int = 2
    corner_marker_color: tuple[int, int, int] = (255, 0, 255)
    aruco_lines_thickness: int = 2
    aruco_lines_color: tuple[int, int, int] = (0, 255, 0)
    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = 0.5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class CharucoAnnotator:
    """Annotates images with charuco corner markers and ArUco bounding boxes.

    Maintains a rolling history of past Keypoints for a track-fade effect.
    """

    config: CharucoAnnotatorConfig
    board_def: CharucoBoardDefinition
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
            marker_color = tuple(int(c * scale) for c in self.config.corner_marker_color)
            marker_thickness = max(1, int(self.config.corner_marker_thickness * scale))
            marker_size = max(1, int(self.config.corner_marker_size * scale))
            is_latest = obs_count == 0

            for corner_id in range(self.board_def.n_corners):
                name = f"CharucoCorner-{corner_id}"
                if not kpts.has_name(name):
                    continue
                idx = kpts.index_of(name)
                if kpts.visibility[idx] == 0.0 or np.isnan(kpts.xyz[idx, 0]):
                    continue
                x, y = int(kpts.xyz[idx, 0]), int(kpts.xyz[idx, 1])
                cv2.drawMarker(
                    annotated,
                    (x, y),
                    marker_color,
                    markerType=self.config.corner_marker_type,
                    markerSize=marker_size,
                    thickness=marker_thickness,
                )
                if is_latest:
                    _draw_doubled_text(
                        annotated,
                        f"Corner#{corner_id}",
                        x + text_offset,
                        y + text_offset,
                        self.config.text_font,
                        self.config.text_size,
                        self.config.text_color,
                        self.config.text_thickness,
                    )

            if is_latest:
                self._draw_aruco_markers(annotated, kpts, text_offset)

        self._draw_undetected_list(annotated, keypoints, image_width)
        return annotated

    def _draw_aruco_markers(
        self,
        image: NDArray[np.uint8],
        kpts: Keypoints,
        text_offset: int,
    ) -> None:
        # Collect ArUco marker corners from keypoints
        seen_markers: dict[int, list[tuple[int, int]]] = {}
        for name in kpts.names:
            if not name.startswith("ArucoMarkerCorner-"):
                continue
            parts = name.split("-")
            if len(parts) != 3:
                continue
            marker_id = int(parts[1])
            j = int(parts[2])
            idx = kpts.index_of(name)
            if kpts.visibility[idx] == 0.0 or np.isnan(kpts.xyz[idx, 0]):
                continue
            if marker_id not in seen_markers:
                seen_markers[marker_id] = [None, None, None, None]
            seen_markers[marker_id][j] = (int(kpts.xyz[idx, 0]), int(kpts.xyz[idx, 1]))

        for marker_id, corners in seen_markers.items():
            if any(c is None for c in corners):
                continue
            corners_arr = np.array(corners, dtype=np.int32)
            cv2.polylines(
                image,
                [corners_arr],
                isClosed=True,
                color=self.config.aruco_lines_color,
                thickness=self.config.aruco_lines_thickness,
            )
            _draw_doubled_text(
                image,
                f"Aruco#{marker_id}",
                corners[0][0] + text_offset,
                corners[0][1] + text_offset,
                self.config.text_font,
                self.config.text_size,
                (255, 125, 0),
                1,
            )

    def _draw_undetected_list(
        self,
        image: NDArray[np.uint8],
        kpts: Keypoints,
        image_width: int,
    ) -> None:
        undetected = []
        for corner_id in range(self.board_def.n_corners):
            name = f"CharucoCorner-{corner_id}"
            if not kpts.has_name(name):
                undetected.append(corner_id)
                continue
            idx = kpts.index_of(name)
            if kpts.visibility[idx] == 0.0 or np.isnan(kpts.xyz[idx, 0]):
                undetected.append(corner_id)

        if not undetected:
            return

        _draw_doubled_text(
            image,
            "Undetected Corners:",
            image_width - 200,
            20,
            self.config.text_font,
            self.config.text_size,
            self.config.text_color,
            self.config.text_thickness,
        )
        for i, corner_id in enumerate(undetected):
            _draw_doubled_text(
                image,
                f" - {corner_id}",
                image_width - 200,
                40 + i * 20,
                self.config.text_font,
                self.config.text_size,
                self.config.text_color,
                self.config.text_thickness,
            )


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
