import logging
import time
from collections import deque
from enum import Enum
from typing import Optional, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker

logger = logging.getLogger(__name__)

# Constants for key actions
KEY_USE_CHARUCO_TRACKER = ord("c")
KEY_USE_MEDIAPIPE_TRACKER = ord("m")

KEY_SHOW_CONTROLS = ord("h")
KEY_SHOW_OVERLAY = ord("o")
KEY_SHOW_INFO = ord("i")
KEY_SET_AUTO_EXPOSURE = ord("a")
KEY_INCREASE_EXPOSURE = ord("w")
KEY_DECREASE_EXPOSURE = ord("s")
KEY_RESET_EXPOSURE = ord("r")
KEY_PAUSE_SPACE = ord(" ")
KEY_PAUSE_P = ord("p")
KEY_QUIT_Q = ord("q")
KEY_QUIT_ESC = 27


class ExposureModes(float, Enum):
    AUTO = 0.75  # Default value to activate auto exposure mode
    MANUAL = 0.25  # Default value to activate manual exposure mode


class WebcamDemoViewer:
    DEFAULT_EXPOSURE = -7
    MAX_EXPOSURE = -12
    MIN_EXPOSURE = -4

    def __init__(
            self,
            tracker: 'BaseTracker' = None,
            window_title: Optional[str] = None,
            default_exposure: int = DEFAULT_EXPOSURE,
    ):
        """
        Initialize with a tracker and optional window title and default exposure.
        """
        self.tracker:BaseTracker|None = tracker
        self.default_exposure = default_exposure
        if window_title is None:
            window_title = f"SkellyTracker - {tracker.__class__.__name__}"
        self.window_title = window_title

    def set_tracker(self, tracker: 'BaseTracker'):
        """
        Set the tracker for the viewer.
        """
        self.tracker = tracker
    def _set_auto_exposure_mode(self, cap):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ExposureModes.AUTO.value)

    def _set_manual_exposure_mode(self, cap):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ExposureModes.MANUAL.value)

    def _set_exposure(self, cap, exposure):
        """
        Set the exposure of the camera.
        """
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def _show_overlay(self, image, text):
        """
        Overlay text on the image.
        """

        y0, dy = 30, 25  # y0 - initial y value, dy - offset between lines
        x0 = 6
        number_of_lines = text.count("\n") + 1
        longest_line = max(text.split("\n"), key=len)
        rect_horizontal_edge_length = len(longest_line) * 13
        rect_vertical_edge_length = dy * number_of_lines + 10
        rect_upper_left_coordinates = (int(x0 / 4), int(y0 / 4))
        rect_lower_right_coordinates = (
            int(x0 / 2) + rect_horizontal_edge_length, int(x0 / 2) + rect_vertical_edge_length)
        overlay = image.copy()
        rect_color = (0, 0, 0)
        cv2.rectangle(overlay, rect_upper_left_coordinates, rect_lower_right_coordinates, rect_color, -1)

        alpha = 0.6  # Transparency factor
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for i, line in enumerate(text.split("\n")):
            y = y0 + i * dy
            self.draw_doubled_text(image, line, x0, y, 0.7, (255, 25, 210), 2)

    def draw_doubled_text(self, image, text, x, y, font_scale, color, thickness):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness * 4)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def run(self, tracker: 'BaseTracker' = None):
        """
        Run the camera viewer.
        """

        if tracker is not None:
            self.set_tracker(tracker)
        if self.tracker is None:
            raise RuntimeError("Error: No tracker set! use `set_tracker(tracker)` to set a tracker.")

        port_number = 0
        frame_number = 0
        cap: cv2.VideoCapture | None = None
        while port_number < 10:
            cap = cv2.VideoCapture(port_number)
            if cap.isOpened():
                break
            port_number += 1
        if cap is None:
            raise RuntimeError("Error: Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            logger.error("Error: Could not open camera.")
            return

        exposure = self.default_exposure
        auto_exposure = True
        show_controls = False
        paused = False
        show_overlay = True
        show_info = True
        show_watermark = True
        self._set_exposure(cap, exposure)
        self._set_auto_exposure_mode(cap)

        cv2.namedWindow(self.window_title)
        frame_durations = deque(maxlen=30)

        tracker_durations = deque(maxlen=30)
        annotation_durations = deque(maxlen=30)
        tik = time.perf_counter()
        success, image = cap.read()
        while True:
            if not paused:
                success, image = cap.read()
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tok = time.perf_counter()
                frame_durations.append(tok - tik)
                tik = tok

                if not success:
                    logger.error("Error: Failed to read image.")
                    break
                frame_number += 1

                tracker_tik = time.perf_counter()
                observation = self.tracker.process_image(frame_number=frame_number,
                                                                      image=image)
                tracker_tok = time.perf_counter()
                tracker_durations.append(tracker_tok - tracker_tik)

                annotation_tik = time.perf_counter()
                annotated_image = self.tracker.annotate_image(image, observation)
                annotation_tok = time.perf_counter()
                annotation_durations.append(annotation_tok - annotation_tik)

                # # Get the window size
                # _, _, window_width, window_height = cv2.getWindowImageRect(self.window_title)
                # # Resize the image to fit the window
                # annotated_image = cv2.resize(annotated_image, (window_width, window_height))

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT_Q or key == KEY_QUIT_ESC:
                break
            elif key == KEY_PAUSE_SPACE or key == KEY_PAUSE_P:
                paused = not paused
            elif key == KEY_USE_CHARUCO_TRACKER:
                if "charuco" not in self.tracker.__class__.__name__.lower():
                    logger.info("Switching to CharucoTracker")
                    from skellytracker.trackers.charuco_tracker import CharucoTracker
                    self.set_tracker(CharucoTracker.create())
            elif key == KEY_USE_MEDIAPIPE_TRACKER:
                if "mediapipe" not in self.tracker.__class__.__name__.lower():
                    logger.info("Switching to MediaPipeTracker")
                    from skellytracker.trackers.mediapipe_tracker import MediapipeTracker
                    self.set_tracker(MediapipeTracker.create())
            elif key == KEY_SHOW_OVERLAY:
                show_overlay = not show_overlay
                if hasattr(self.tracker.config.annotator_config, "show_overlay"):
                    self.tracker.config.annotator_config.show_overlay = show_overlay
            elif key == KEY_SHOW_INFO: 
                show_info = not show_info
            elif key == KEY_SET_AUTO_EXPOSURE:
                auto_exposure = True
                self._set_auto_exposure_mode(cap)
            elif key == KEY_INCREASE_EXPOSURE:
                exposure += 1
                auto_exposure = False
                self._set_manual_exposure_mode(cap)
                self._set_exposure(cap, exposure)
            elif key == KEY_DECREASE_EXPOSURE:
                exposure -= 1
                auto_exposure = False
                self._set_manual_exposure_mode(cap)
                self._set_exposure(cap, exposure)
            elif key == KEY_RESET_EXPOSURE:
                exposure = self.default_exposure
                self._set_exposure(cap, exposure)
            elif key == KEY_SHOW_CONTROLS:
                show_controls = not show_controls
            mean_luminance = image.mean() / 3
            mean_frame_duration = sum(frame_durations) / len(frame_durations)
            mean_frames_per_second = 1 / mean_frame_duration
            mean_tracker_duration = sum(tracker_durations) / len(tracker_durations)
            mean_annotation_duration = sum(annotation_durations) / len(annotation_durations)
            overlay_string = ""
            exposure_string = f"Exposure: {exposure if not auto_exposure else 'AUTO'}"
            exposure_string += f"({(2 ** exposure) * 1000:.2f}ms)" if not auto_exposure else "\n"
            exposure_string += f"(~ {(2 ** exposure) * 1000:.2f}ms)\n\n" if not auto_exposure else "\n\n"
            info_string = exposure_string
            info_string += f"Mean Luminance: {mean_luminance / 255:.2f}\n"
            info_string += f"Mean FPS: {mean_frames_per_second:.2f}\n"
            info_string += f"Mean Frame Duration: {mean_frame_duration * 1000:.2f} ms\n"
            info_string += f"Mean Tracker Processing Duration: {mean_tracker_duration * 1000:.2f} ms\n"
            info_string += f"Mean Annotation Duration: {mean_annotation_duration * 1000:.2f} ms\n"
            if show_info:
                overlay_string += info_string
            if show_controls:
                overlay_string += (
                    "Controls:\n"
                    f"'SPACE'/'{chr(KEY_PAUSE_P)}': pause\n"
                    f"'Current Tracker: {self.tracker.__class__.__name__}\n"
                    f"'{chr(KEY_USE_CHARUCO_TRACKER)})': Use CharucoTracker\n"
                    f"'{chr(KEY_USE_MEDIAPIPE_TRACKER)})': Use MediaPipeTracker\n"
                    f"'{chr(KEY_SHOW_INFO)}': {'show info' if not show_info else 'hide info'}\n"
                    f"'{chr(KEY_SHOW_OVERLAY)}': show overlay\n"
                    f"'{chr(KEY_SET_AUTO_EXPOSURE)}': auto-exposure\n"
                    f"'{chr(KEY_INCREASE_EXPOSURE)}'/'{chr(KEY_DECREASE_EXPOSURE)}': exposure +/-\n"
                    f"'{chr(KEY_RESET_EXPOSURE)}': reset\n"
                    f"'ESC/{chr(KEY_QUIT_Q)}': quit\n"
                    f"'{chr(KEY_SHOW_CONTROLS)}': hide controls"
                )
            else:
                overlay_string += f"'{chr(KEY_SHOW_CONTROLS)}': show controls"

            self._show_overlay(
                annotated_image,
                overlay_string,
            )
            cv2.imshow(self.window_title, annotated_image)

        cap.release()
        cv2.destroyAllWindows()
