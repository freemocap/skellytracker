import cv2
import numpy as np
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker import BaseImageAnnotatorConfig
from skellytracker.trackers.charuco_tracker.charuco_observations import CharucoObservations


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


class CharucoImageAnnotator(BaseModel):
    config: CharucoAnnotatorConfig

    @classmethod
    def create(cls, config: CharucoAnnotatorConfig):
        return cls(config=config)

    def annotate_image(
            self,
            image: np.ndarray,
            observations: CharucoObservations,

    ) -> np.ndarray:
        # Copy the original image for annotation
        annotated_image = image.copy()
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)
        if self.config.show_tracks is not None:
            show_tracks = min(self.config.show_tracks, len(observations))
        else:
            show_tracks = len(observations)

        # Reverse the observations list so that the most recent observations are drawn last (on top)
        observations = observations[::-1]

        # Draw a marker for each tracked corner
        for obs_count, observation in enumerate(observations[:show_tracks]):
            obs_count_scale = 1 - (obs_count / show_tracks)
            marker_color = tuple(int(element * obs_count_scale) for element in self.config.marker_color)
            marker_thickness = max(1, int(self.config.marker_thickness * obs_count_scale))
            marker_size = max(1, int(self.config.marker_size * obs_count_scale))

            for corner_id, corner in observation.charuco_corners.items():
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
        for key, value in observations[0].charuco_corners.items():
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
