from __future__ import annotations

from dataclasses import dataclass

from skellytracker.core.temporal_processing.temporal_processing_config import (
    KeypointResetPolicyConfig,
)


@dataclass
class KeypointResetPolicy:
    """Decides when a keypoint detector's consecutive-miss streak warrants a reset.

    See KeypointResetPolicyConfig for the motivating failure mode (MediaPipe
    VIDEO-mode tracking getting silently stuck) and for why backoff exists: a
    bare threshold re-fires every max_consecutive_misses frames for as long as
    the subject stays out of frame, since a successful reset zeroes the miss
    counter and the empty streak immediately starts climbing again.
    """

    max_consecutive_misses: int | None = None
    backoff_multiplier: float = 2.0
    max_backoff_misses: int | None = None

    def should_reset(self, consecutive_misses: int, consecutive_resets: int = 0) -> bool:
        if self.max_consecutive_misses is None:
            return False
        threshold = self.max_consecutive_misses * (self.backoff_multiplier**consecutive_resets)
        if self.max_backoff_misses is not None:
            threshold = min(threshold, self.max_backoff_misses)
        return consecutive_misses >= threshold

    @classmethod
    def from_config(cls, config: KeypointResetPolicyConfig) -> KeypointResetPolicy:
        return cls(
            max_consecutive_misses=config.max_consecutive_misses,
            backoff_multiplier=config.backoff_multiplier,
            max_backoff_misses=config.max_backoff_misses,
        )
