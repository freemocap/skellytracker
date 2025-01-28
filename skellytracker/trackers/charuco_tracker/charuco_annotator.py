import cv2
import numpy as np

from skellytracker.trackers.base_tracker.base_tracker import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation


class CharucoAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    corner_marker_type: int = cv2.MARKER_DIAMOND
    corner_marker_size: int = 10
    corner_marker_thickness: int = 2
    corner_marker_color: tuple[int, int, int] = (0, 0, 255)

    aruco_lines_thickness: int = 2
    aruco_lines_color: tuple[int, int, int] = (0, 255, 0)

    text_color: tuple[int, int, int] = (215, 115, 40)
    text_size: float = .5
    text_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX


class CharucoImageAnnotator(BaseImageAnnotator):
    config: CharucoAnnotatorConfig
    observations: list[CharucoObservation]

    @classmethod
    def create(cls, config: CharucoAnnotatorConfig):
        return cls(config=config ,
                     observations=[])

    def annotate_image(
            self,
            image: np.ndarray,
            latest_observation: CharucoObservation | None = None,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        text_offset = int(image_height * 0.01)

        if latest_observation is None:
            return image.copy()
        # Copy the original image for annotation
        annotated_image = image.copy()

        self.observations.append(latest_observation)

        if self.config.show_tracks is None or self.config.show_tracks < 1:
            self.observations = [latest_observation]
        elif len(self.observations) > self.config.show_tracks:
            self.observations = self.observations[-self.config.show_tracks:]

        # Draw a marker for each tracked corner
        for obs_count, observation in enumerate(self.observations[::-1]):
            if latest_observation.charuco_empty:
                continue
            obs_count_scale = 1 - (obs_count / len(self.observations))

            marker_color = tuple(int(element * obs_count_scale) for element in self.config.corner_marker_color)
            marker_thickness = max(1, int(self.config.corner_marker_thickness * obs_count_scale))
            marker_size = max(1, int(self.config.corner_marker_size * obs_count_scale))

            for corner_id, corner in observation.charuco_corners_dict.items():
                if corner is not None:
                    cv2.drawMarker(
                        annotated_image,
                        (int(corner[0]), int(corner[1])),
                        marker_color,
                        markerType=self.config.corner_marker_type,
                        markerSize=marker_size,
                        thickness=marker_thickness,
                    )
                    if obs_count == 0:
                        self.draw_doubled_text(image=annotated_image,
                                               text=f"Corner#{corner_id}",
                                               x=int(corner[0]) + text_offset,
                                               y=int(corner[1]) + text_offset,
                                               font_scale=self.config.text_size,
                                               color=self.config.text_color,
                                               thickness=self.config.text_thickness)
                        for aruco_id, aruco_bounding_box in observation.aruco_corners_dict.items():
                            if aruco_bounding_box is not None:
                                cv2.polylines(
                                    annotated_image,
                                    [np.array(aruco_bounding_box, dtype=np.int32)],
                                    isClosed=True,
                                    color=self.config.aruco_lines_color,
                                    thickness=self.config.aruco_lines_thickness,
                                )
                                self.draw_doubled_text(
                                    image=annotated_image,
                                    text=f"Aruco#{aruco_id}",
                                    x=int(aruco_bounding_box[0][0]) + text_offset,
                                    y=int(aruco_bounding_box[0][1]) + text_offset,
                                    font_scale=self.config.text_size,
                                    color=(255, 125, 0),
                                    thickness=1)

        # List undetected markers
        undetected_corners = latest_observation.all_charuco_ids.copy()
        if not latest_observation.charuco_empty:
            for charuco_id in latest_observation.detected_charuco_corner_ids:
                undetected_corners.remove(charuco_id[0])

        if len(undetected_corners) > 0:
            self.draw_doubled_text(
                image=annotated_image,
                text="Undetected Corners:",
                x=image_width - 200,
                y=20,
                font_scale=self.config.text_size,
                color=self.config.text_color,
                thickness=self.config.text_thickness)

            for undetected_marker_number, marker_id in enumerate(undetected_corners):
                v_offset = undetected_marker_number * 20
                self.draw_doubled_text(image=annotated_image,
                                       text=f" - {marker_id}",
                                       x=image_width - 200,
                                       y=40 + v_offset,
                                       font_scale=self.config.text_size,
                                       color=self.config.text_color,
                                       thickness=self.config.text_thickness,
                                       )

        return annotated_image
