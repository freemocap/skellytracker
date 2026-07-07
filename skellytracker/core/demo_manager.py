from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.annotator import Annotator
from skellytracker.core.tracker.tracker import Tracker
from skellytracker.core.tracker.tracker_state import TrackerState

logger = logging.getLogger(__name__)

_KEY_QUIT_Q = ord("q")
_KEY_QUIT_ESC = 27
_KEY_PAUSE = ord(" ")
_KEY_HELP = ord("h")


class _CaptureThread:
    """Reads frames from a VideoCapture in a background thread.

    Keeps the most recent frame so the inference loop never blocks on camera I/O.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: NDArray[np.uint8] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


@dataclass
class DemoManager:
    """Runs a Tracker in a live loop against a webcam or video file.

    Intentionally thin: wires together Tracker, Annotator, and an OpenCV window.
    TrackerState is maintained across frames; session cleanup happens on exit.
    """

    tracker: Tracker
    annotator: Annotator
    window_title: str = "SkellyTracker Demo"
    _fps_history: deque = field(default_factory=lambda: deque(maxlen=30), init=False, repr=False)
    _inference_history: deque = field(default_factory=lambda: deque(maxlen=30), init=False, repr=False)

    def run_webcam(self, camera_index: int = 0) -> None:
        cap = self._open_camera(camera_index)
        capture_thread = _CaptureThread(cap)
        self._wait_for_first_frame(capture_thread)
        try:
            self._run_loop(capture_thread)
        finally:
            capture_thread.stop()
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()

    def run_video(self, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            self._run_loop_video(cap)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {camera_index}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _wait_for_first_frame(self, capture_thread: _CaptureThread) -> None:
        while True:
            ok, _ = capture_thread.read()
            if ok:
                return
            time.sleep(0.01)

    def _run_loop(self, capture_thread: _CaptureThread) -> None:
        state = TrackerState()
        frame_number = 0
        paused = False
        show_help = False
        annotated: NDArray[np.uint8] | None = None

        cv2.namedWindow(self.window_title)
        t_prev = time.perf_counter()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (_KEY_QUIT_Q, _KEY_QUIT_ESC):
                break
            if key == _KEY_PAUSE:
                paused = not paused
            if key == _KEY_HELP:
                show_help = not show_help

            if not paused:
                ok, frame = capture_thread.read()
                if not ok or frame is None:
                    continue

                t_now = time.perf_counter()
                self._fps_history.append(t_now - t_prev)
                t_prev = t_now
                timestamp_ms = int(t_now * 1000)

                t_inf = time.perf_counter()
                observation, state = self.tracker.process_image(frame, frame_number, state, timestamp_ms)
                self._inference_history.append(time.perf_counter() - t_inf)

                annotated = self.annotator.annotate(frame, observation)
                frame_number += 1

            if annotated is not None:
                display = annotated.copy()
                self._draw_hud(display, show_help)
                cv2.imshow(self.window_title, display)

    def _run_loop_video(self, cap: cv2.VideoCapture) -> None:
        state = TrackerState()
        frame_number = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cv2.namedWindow(self.window_title)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(frame_number / fps * 1000)
            observation, state = self.tracker.process_image(frame, frame_number, state, timestamp_ms)
            annotated = self.annotator.annotate(frame, observation)
            frame_number += 1

            cv2.imshow(self.window_title, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (_KEY_QUIT_Q, _KEY_QUIT_ESC):
                break

    def _draw_hud(self, image: NDArray[np.uint8], show_help: bool) -> None:
        fps = (1.0 / (sum(self._fps_history) / len(self._fps_history))) if self._fps_history else 0.0
        inf_ms = (sum(self._inference_history) / len(self._inference_history) * 1000) if self._inference_history else 0.0

        lines = [f"FPS: {fps:.1f}  |  inference: {inf_ms:.1f} ms"]
        if show_help:
            lines += ["SPACE: pause  |  h: toggle help  |  q/ESC: quit"]
        else:
            lines.append("h: help")

        y = 24
        for line in lines:
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 22
