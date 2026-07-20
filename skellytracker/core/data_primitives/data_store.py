from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.data_primitives.multi_person_observation import MultiPersonObservation
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


@dataclass
class MultiPersonDataStore:
    """Accumulates MultiPersonObservations, split into one DataStore per track_id.

    Lazily creates a DataStore the first time a track_id is seen and forwards
    that person's Observation to it every frame it appears — no changes to
    DataStore itself are needed.
    """

    stores: dict[int, DataStore] = field(default_factory=dict)

    def add(self, observation: MultiPersonObservation) -> None:
        for track_id, person_obs in observation.people.items():
            if track_id not in self.stores:
                self.stores[track_id] = DataStore()
            self.stores[track_id].add(person_obs)

    def to_arrays(self) -> dict[int, NDArray[np.float64]]:
        return {track_id: store.to_array() for track_id, store in self.stores.items()}

    def save(
        self,
        directory: Path,
        fmt: Literal["npy", "json"] = "npy",
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "npy" if fmt == "npy" else "json"
        for track_id, store in self.stores.items():
            store.save(directory / f"track_{track_id}.{suffix}", fmt=fmt)
