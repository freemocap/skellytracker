import logging
import time
from collections import deque
from enum import Enum
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

# Constants for key actions
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
            tracker,
            window_title: Optional[str] = None,
            default_exposure: int = DEFAULT_EXPOSURE,
    ):
        """
        Initialize with a tracker and optional window title and default exposure.
        """
        self.tracker = tracker
        self.default_exposure = default_exposure
        if window_title is None:
            window_title = f"SkellyTracker - {tracker.__class__.__name__}"
        self.window_title = window_title

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
        rect_horizontal_edge_length = len(longest_line) * 10
        rect_vertical_edge_length = dy * number_of_lines + 10
        rect_upper_left_coordinates = (int(x0 / 2), int(y0 / 2))
        rect_lower_right_coordinates = (
        int(x0 / 2) + rect_vertical_edge_length, int(x0 / 2) + rect_horizontal_edge_length)
        rect_color_and_transparency = (25, 25, 25, .2)
        # cv2.rectangle(image, rect_upper_left_coordinates, rect_lower_right_coordinates, rect_color_and_transparency, -1)

        for i, line in enumerate(text.split("\n")):
            y = y0 + i * dy
            self.draw_doubled_text(image, line, x0, y, 0.7, (255, 15, 210), 1)

    def draw_doubled_text(self, image, text, x, y, font_scale, color, thickness):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def run(self):
        """
        Run the camera viewer.
        """
        cap = cv2.VideoCapture(0)
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

                tracker_tik = time.perf_counter()
                observation, raw_results = self.tracker.process_image(image, annotate_image=False)
                tracker_tok = time.perf_counter()
                tracker_durations.append(tracker_tok - tracker_tik)

                annotation_tik = time.perf_counter()
                if show_overlay:
                    # JSM - Hacky nonsense to view the mediapipe segmentation mask. should figure out a way to extract this without sending the whole segmentation mask image
                    if hasattr(raw_results, "segmentation_mask") and raw_results.segmentation_mask is not None:
                        image[:, :, 2] += (raw_results.segmentation_mask * 50).astype('uint8')

                    annotated_image = self.tracker.annotator.annotate_image(image, observation)
                else:
                    annotated_image = image
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
            elif key == KEY_SHOW_OVERLAY:
                show_overlay = not show_overlay
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
            mean_brightness = image.mean() / 3
            mean_frame_duration = sum(frame_durations) / len(frame_durations)
            mean_frames_per_second = 1 / mean_frame_duration
            mean_tracker_duration = sum(tracker_durations) / len(tracker_durations)
            mean_annotation_duration = sum(annotation_durations) / len(annotation_durations)
            overlay_string = ""
            info_string = f"Exposure: {exposure if not auto_exposure else 'auto'}\n"
            info_string += f"Mean Brightness: {mean_brightness:.2f}\n\n"
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
