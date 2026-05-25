import gc
import logging
import time
from collections import deque
from enum import Enum
from typing import Callable, Optional

import cv2

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker

logger = logging.getLogger(__name__)

# Key bindings — single source of truth for key→action mapping.
KEY_USE_BRIGHTEST_POINT_TRACKER = ord("b")
KEY_USE_CHARUCO_TRACKER = ord("c")
KEY_USE_MEDIAPIPE_TRACKER = ord("m")
KEY_TOGGLE_RUST_BACKEND = ord("r")
KEY_SHOW_CONTROLS = ord("h")
KEY_SHOW_OVERLAY = ord("o")
KEY_SHOW_INFO = ord("i")
KEY_SET_AUTO_EXPOSURE = ord("a")
KEY_INCREASE_EXPOSURE = ord("w")
KEY_DECREASE_EXPOSURE = ord("s")
KEY_RESET_EXPOSURE = ord("0")  # "0" = reset to default (was ord("r"), conflicted with Rust toggle)
KEY_PAUSE_SPACE = ord(" ")
KEY_PAUSE_P = ord("p")
KEY_QUIT_Q = ord("q")
KEY_QUIT_ESC = 27


class ExposureModes(float, Enum):
    AUTO = 0.75
    MANUAL = 0.25


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
        self.tracker: BaseTracker | None = tracker
        self.use_rust_backend: bool = True
        self.default_exposure = default_exposure
        if window_title is None:
            window_title = f"SkellyTracker - {tracker.__class__.__name__}"
        self.window_title = window_title

        # Display toggles — session state promoted to instance attrs so key
        # handlers don't need nonlocal closures.
        self.paused = False
        self.show_controls = False
        self.show_overlay = True
        self.show_info = True

    # ── Tracker switching helpers ──────────────────────────────────────

    def _is_using_tracker(self, name_fragment: str) -> bool:
        return name_fragment in self.tracker.__class__.__name__.lower()

    def _create_brightest_point_tracker(self) -> BaseTracker:
        """Create a BrightestPointTracker in the currently selected backend."""
        if self.use_rust_backend:
            logger.info("Switching to BrightestPointTracker (Rust)")
            from skellytracker.trackers.brightest_point_tracker.rust_bridge import RustBrightestPointTracker
            return RustBrightestPointTracker.create()
        else:
            logger.info("Switching to BrightestPointTracker (Python)")
            from skellytracker.trackers.brightest_point_tracker import BrightestPointTracker
            return BrightestPointTracker.create()

    def _switch_to_brightest_point(self) -> None:
        if not self._is_using_tracker("brightestpoint"):
            self.tracker = self._create_brightest_point_tracker()

    def _create_charuco_tracker(self) -> BaseTracker:
        """Create a CharucoTracker in the currently selected backend."""
        if self.use_rust_backend:
            logger.info("Switching to CharucoTracker (Rust)")
            from skellytracker.trackers.charuco_tracker.rust_bridge import RustCharucoTracker
            return RustCharucoTracker.create()
        else:
            logger.info("Switching to CharucoTracker (Python)")
            from skellytracker.trackers.charuco_tracker import CharucoTracker
            return CharucoTracker.create()

    def _toggle_rust_backend(self) -> None:
        self.use_rust_backend = not self.use_rust_backend
        tracker_name = self.tracker.__class__.__name__

        if "brightestpoint" in tracker_name.lower():
            self.tracker = self._create_brightest_point_tracker()
        elif "charuco" in tracker_name.lower():
            self.tracker = self._create_charuco_tracker()
        else:
            backend = "Rust" if self.use_rust_backend else "Python"
            logger.warning(
                f"NOT IMPLEMENTED: {tracker_name} has no {backend} backend "
                f"— only BrightestPointTracker and CharucoTracker support Rust/Python hot-swap"
            )

    def _switch_to_charuco(self) -> None:
        if not self._is_using_tracker("charuco"):
            self.tracker = self._create_charuco_tracker()

    def _switch_to_mediapipe(self) -> None:
        if not self._is_using_tracker("mediapipe"):
            logger.info("Switching to MediaPipeTracker")
            from skellytracker.trackers.mediapipe_tracker import MediapipeTracker
            self.tracker = MediapipeTracker.create()

    # ── Camera helpers ─────────────────────────────────────────────────


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

    # ── Key handler dispatch table builder ─────────────────────────────
    # Camera-scoped state (exposure, auto_exposure, cap) is closed over.
    # Display toggles live on `self` so methods can access them directly.

    def _build_key_handlers(
            self, cap, exposure_container, auto_exposure_container
    ) -> dict[int, Callable[[], bool | None]]:
        """
        Build a dispatch dict mapping key codes to callables.

        Each handler returns True to break the main loop, or None to continue.
        A mutable container (list) holds exposure/auto_exposure so handlers
        can mutate them without nonlocal declarations.
        """

        def _quit() -> bool:
            return True

        def _pause() -> None:
            self.paused = not self.paused

        def _brightest_point() -> None:
            self._switch_to_brightest_point()

        def _toggle_rust() -> None:
            self._toggle_rust_backend()

        def _charuco() -> None:
            self._switch_to_charuco()

        def _mediapipe() -> None:
            self._switch_to_mediapipe()

        def _show_overlay() -> None:
            self.show_overlay = not self.show_overlay
            if hasattr(self.tracker.config.annotator_config, "show_overlay"):
                self.tracker.config.annotator_config.show_overlay = self.show_overlay

        def _show_info() -> None:
            self.show_info = not self.show_info

        def _auto_exposure() -> None:
            auto_exposure_container[0] = True
            self._set_auto_exposure_mode(cap)

        def _increase_exposure() -> None:
            exposure_container[0] += 1
            auto_exposure_container[0] = False
            self._set_manual_exposure_mode(cap)
            self._set_exposure(cap, exposure_container[0])

        def _decrease_exposure() -> None:
            exposure_container[0] -= 1
            auto_exposure_container[0] = False
            self._set_manual_exposure_mode(cap)
            self._set_exposure(cap, exposure_container[0])

        def _reset_exposure() -> None:
            exposure_container[0] = self.default_exposure
            self._set_exposure(cap, exposure_container[0])

        def _show_controls() -> None:
            self.show_controls = not self.show_controls

        return {
            KEY_QUIT_Q: _quit,
            KEY_QUIT_ESC: _quit,
            KEY_PAUSE_SPACE: _pause,
            KEY_PAUSE_P: _pause,
            KEY_USE_BRIGHTEST_POINT_TRACKER: _brightest_point,
            KEY_TOGGLE_RUST_BACKEND: _toggle_rust,
            KEY_USE_CHARUCO_TRACKER: _charuco,
            KEY_USE_MEDIAPIPE_TRACKER: _mediapipe,
            KEY_SHOW_OVERLAY: _show_overlay,
            KEY_SHOW_INFO: _show_info,
            KEY_SET_AUTO_EXPOSURE: _auto_exposure,
            KEY_INCREASE_EXPOSURE: _increase_exposure,
            KEY_DECREASE_EXPOSURE: _decrease_exposure,
            KEY_RESET_EXPOSURE: _reset_exposure,
            KEY_SHOW_CONTROLS: _show_controls,
        }

    def _build_overlay_string(self, exposure, auto_exposure) -> str:
        """Build the overlay text for the current frame."""
        mean_luminance = self._image.mean() / 3
        mean_frame_duration = (
            sum(self._frame_durations) / len(self._frame_durations)
        )
        mean_frames_per_second = 1 / mean_frame_duration
        mean_tracker_duration = (
            sum(self._tracker_durations) / len(self._tracker_durations)
        )
        mean_annotation_duration = (
            sum(self._annotation_durations) / len(self._annotation_durations)
        )

        info_string = f"Exposure: {exposure if not auto_exposure else 'AUTO'}"
        if not auto_exposure:
            info_string += f"({(2 ** exposure) * 1000:.2f}ms)\n\n"
        else:
            info_string += "\n\n"

        info_string += f"Backend: {'Rust' if 'Rust' in self.tracker.__class__.__name__ else 'Python'}\n"
        info_string += f"Mean Luminance: {mean_luminance / 255:.2f}\n"
        info_string += f"Mean FPS: {mean_frames_per_second:.2f}\n"
        info_string += f"Mean Frame Duration: {mean_frame_duration * 1000:.2f} ms\n"
        info_string += f"Mean Tracker Processing Duration: {mean_tracker_duration * 1000:.2f} ms\n"
        info_string += f"Mean Annotation Duration: {mean_annotation_duration * 1000:.2f} ms\n"

        overlay = ""
        if self.show_info:
            overlay += info_string
        if self.show_controls:
            overlay += (
                "Controls:\n"
                f"'SPACE'/'{chr(KEY_PAUSE_P)}': pause\n"
                f"'Current Tracker: {self.tracker.__class__.__name__}\n"
                f"'{chr(KEY_USE_BRIGHTEST_POINT_TRACKER)})': Use BrightestPointTracker\n"
                f"'{chr(KEY_USE_CHARUCO_TRACKER)})': Use CharucoTracker\n"
                f"'{chr(KEY_USE_MEDIAPIPE_TRACKER)})': Use MediaPipeTracker\n"
                f"'{chr(KEY_TOGGLE_RUST_BACKEND)}': Toggle Rust/Python backend (currently "
                f"{'Rust' if self.use_rust_backend else 'Python'})\n"
                f"'{chr(KEY_SHOW_INFO)}': {'show info' if not self.show_info else 'hide info'}\n"
                f"'{chr(KEY_SHOW_OVERLAY)}': show overlay\n"
                f"'{chr(KEY_SET_AUTO_EXPOSURE)}': auto-exposure\n"
                f"'{chr(KEY_INCREASE_EXPOSURE)}'/'{chr(KEY_DECREASE_EXPOSURE)}': exposure +/-\n"
                f"'{chr(KEY_RESET_EXPOSURE)}': reset\n"
                f"'ESC/{chr(KEY_QUIT_Q)}': quit\n"
                f"'{chr(KEY_SHOW_CONTROLS)}': hide controls"
            )
        else:
            overlay += f"'{chr(KEY_SHOW_CONTROLS)}': show controls"
        return overlay

    def run(self):
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

        # Mutable containers closed over by key handlers so they can mutate
        # camera-scoped state without nonlocal declarations.
        exposure_container = [self.default_exposure]
        auto_exposure_container = [True]

        self._set_exposure(cap, exposure_container[0])
        self._set_auto_exposure_mode(cap)

        key_handlers = self._build_key_handlers(
            cap, exposure_container, auto_exposure_container
        )

        cv2.namedWindow(self.window_title)
        self._frame_durations = deque(maxlen=30)
        self._tracker_durations = deque(maxlen=30)
        self._annotation_durations = deque(maxlen=30)
        tik = time.perf_counter()
        success, self._image = cap.read()

        while True:
            if not self.paused:
                success, self._image = cap.read()
                tok = time.perf_counter()
                self._frame_durations.append(tok - tik)
                tik = tok

                if not success:
                    logger.error("Error: Failed to read image.")
                    break
                frame_number += 1

                tracker_tik = time.perf_counter()
                observation = self.tracker.process_image(
                    frame_number=frame_number,
                    image=self._image,
                    record_observation=False,
                )
                tracker_tok = time.perf_counter()
                self._tracker_durations.append(tracker_tok - tracker_tik)

                annotation_tik = time.perf_counter()
                if observation is not None:
                    annotated_image = self.tracker.annotate_image(
                        self._image, observation
                    )
                    annotation_tok = time.perf_counter()
                    self._annotation_durations.append(
                        annotation_tok - annotation_tik
                    )

            # ── Key dispatch ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            handler = key_handlers.get(key)
            if handler is not None:
                if handler():
                    break
            # ───────────────────────────────────────────────────────────

            overlay_string = self._build_overlay_string(
                exposure_container[0], auto_exposure_container[0]
            )

            self._show_overlay(annotated_image, overlay_string)
            cv2.imshow(self.window_title, annotated_image)

            # Periodic GC prevents OpenCV internal buffer accumulation
            # on long-running Windows sessions.
            if not self.paused and frame_number % 60 == 0:
                gc.collect()

        cap.release()
        cv2.destroyAllWindows()
