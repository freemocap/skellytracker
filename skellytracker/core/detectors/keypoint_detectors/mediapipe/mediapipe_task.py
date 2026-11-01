"""Lazy native-task ownership shared by MediaPipe detectors."""
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


@dataclass
class MediaPipeTask:
    create_task: Callable[[Any], Any]
    options: Any
    _native_task: Any = field(default=None, init=False, repr=False)

    def _get_task(self) -> Any:
        if self._native_task is None:
            self._native_task = self.create_task(self.options)
        return self._native_task

    def detect(self, image: Any) -> Any:
        return self._get_task().detect(image)

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        return self._get_task().detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        if self._native_task is not None:
            native_task, self._native_task = self._native_task, None
            native_task.close()
