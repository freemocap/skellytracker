import cv2
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotator, BaseImageAnnotatorConfig


class DeepLabCutAnnotatorConfig(BaseImageAnnotatorConfig):
    show_tracks: int | None = 15
    show_overlay: bool = True
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


class DeepLabCutImageAnnotator(BaseImageAnnotator):
    config: DeepLabCutAnnotatorConfig