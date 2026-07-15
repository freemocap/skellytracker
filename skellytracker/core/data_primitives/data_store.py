from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives.observation import Observation


@dataclass
class DataStore:
    """Accumulates Observations across frames and serializes them.

    Primary output is a (frames, points, 3) numpy array for freemocap
    triangulation. JSON output provides a human-readable alternative.
    """

    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def to_array(self) -> NDArray[np.float64]:
        """Return (num_frames, num_points, 3) array of all keypoints."""
        if not self.observations:
            raise ValueError("DataStore is empty")
        clouds = [obs.to_keypoints() for obs in self.observations]
        return np.stack([c.xyz for c in clouds], axis=0)

    def to_json(self) -> str:
        frames = {}
        for obs in self.observations:
            kpts = obs.to_keypoints()
            frames[obs.frame_number] = kpts.to_named_dict(dimensions=3)
        return json.dumps(
            {k: {name: v.tolist() for name, v in d.items()} for k, d in frames.items()}
        )

    def save(
        self,
        path: Path,
        fmt: Literal["npy", "json"] = "npy",
    ) -> None:
        path = Path(path)
        if fmt == "npy":
            np.save(path, self.to_array())
        else:
            path.write_text(self.to_json())
