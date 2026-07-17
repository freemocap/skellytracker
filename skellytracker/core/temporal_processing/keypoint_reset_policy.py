from __future__ import annotations

from dataclasses import dataclass

from skellytracker.core.temporal_processing.temporal_processing_config import (
    KeypointResetPolicyConfig,
)


@dataclass
class KeypointResetPolicy:
    """Decides when a keypoint detector's consecutive-miss streak warrants a reset.

    See KeypointResetPolicyConfig for the motivating failure mode (MediaPipe
    VIDEO-mode tracking getting silently stuck).
    """

    max_consecutive_misses: int | None = None

    def should_reset(self, consecutive_misses: int) -> bool:
        if self.max_consecutive_misses is None:
            return False
        return consecutive_misses >= self.max_consecutive_misses

    @classmethod
    def from_config(cls, config: KeypointResetPolicyConfig) -> KeypointResetPolicy:
        return cls(max_consecutive_misses=config.max_consecutive_misses)
