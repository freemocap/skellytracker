from dataclasses import field, dataclass

import cv2
import numpy as np
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.charuco_tracker.charuco_observations import CharucoObservations, CharucoObservation


class CharucoAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    marker_type: int = cv2.MARKER_DIAMOND
    marker_size: int = 10
    marker_thickness: int = 2
    marker_color: tuple[int, int, int] = (0, 0, 255)

    text_color: tuple[int, int, int] = (255, 255, 0)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class CharucoImageAnnotator(BaseImageAnnotator):
    config: CharucoAnnotatorConfig
    observations: dict[int, CharucoObservations]  = field(default_factory=dict)

    @classmethod
    def create(cls, config: CharucoAnnotatorConfig):
        return cls(config=config)

    def annotate_image(
            self,
            image: np.ndarray,
            latest_observation: CharucoObservation|None = None,
            camera_id: int = 0,

    ) -> np.ndarray:
        if latest_observation is None:
            return image
        # Copy the original image for annotation
        annotated_image = image.copy()
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)

        if not camera_id in self.observations:
            self.observations[camera_id] = []
        self.observations[camera_id].append(latest_observation)
        if self.config.show_tracks is None or self.config.show_tracks < 1:
            self.observations[camera_id] = [latest_observation]
        elif len(self.observations[camera_id]) > self.config.show_tracks:
            self.observations[camera_id] = self.observations[camera_id][-self.config.show_tracks:]

        # Draw a marker for each tracked corner
        for obs_count, observation in enumerate(self.observations[camera_id][::-1]):
            obs_count_scale = 1 - (obs_count / len(self.observations[camera_id]))

            marker_color = tuple(int(element * obs_count_scale) for element in self.config.marker_color)
            marker_thickness = max(1, int(self.config.marker_thickness * obs_count_scale))
            marker_size = max(1, int(self.config.marker_size * obs_count_scale))

            for corner_id, corner in observation.charuco_corners_image.items():
                if corner is not None:
                    cv2.drawMarker(
                        annotated_image,
                        (int(corner[0]), int(corner[1])),
                        marker_color,
                        markerType=self.config.marker_type,
                        markerSize=marker_size,
                        thickness=marker_thickness,
                    )
                    if obs_count == 0:
                        cv2.putText(
                            annotated_image,
                            f"Corner#{corner_id}",
                            (int(corner[0]) + text_offset, int(corner[1]) + text_offset),
                            self.config.text_font,
                            self.config.text_size,
                            self.config.text_color,
                            self.config.text_thickness,
                        )
                        for aruco_id, aruco_bounding_box in observation.aruco_marker_corners.items():
                            if aruco_bounding_box is not None:
                                cv2.polylines(
                                    annotated_image,
                                    [np.array(aruco_bounding_box, dtype=np.int32)],
                                    isClosed=True,
                                    color=(255, 125, 0),
                                    thickness=1,
                                )
                                cv2.putText(
                                    annotated_image,
                                    f"Aruco#{aruco_id}",
                                    (int(aruco_bounding_box[0][0]) + text_offset,
                                     int(aruco_bounding_box[0][1]) + text_offset),
                                    self.config.text_font,
                                    self.config.text_size,
                                    (255, 125, 0),
                                    1,
                                )

        # List undetected markers
        undetected_markers = []
        for key, value in latest_observation.charuco_corners_image.items():
            if value is None:
                undetected_markers.append(key)
        if len(undetected_markers) > 0:
            cv2.putText(
                annotated_image,
                "Undetected Corners:",
                (image_width - 200, 20),
                self.config.text_font,
                self.config.text_size,
                self.config.text_color,
                self.config.text_thickness,
            )
            for undetected_marker_number, marker_id in enumerate(undetected_markers):
                v_offset = undetected_marker_number * 20
                cv2.putText(
                    annotated_image,
                    f"ID: {marker_id}",
                    (image_width - 200, 40 + v_offset),
                    self.config.text_font,
                    self.config.text_size,
                    self.config.text_color,
                    self.config.text_thickness,
                )
        return annotated_image
