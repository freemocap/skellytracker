"""
Three-form tracker→canonical landmark mapping.

A ``TrackerMapping`` loads a YAML file that defines, for each canonical
landmark, how to produce its position from the tracker's keypoints.
Three forms are supported::

    string  →  1:1 passthrough  ``left_elbow: "left_elbow"``
    list    →  unweighted mean  ``hips_center: ["left_hip", "right_hip"]``
    dict    →  weighted sum     ``head_center: {left_ear: 0.5, right_ear: 0.5}``

Virtual markers as a separate concept disappear — a "virtual marker" is
simply a canonical landmark whose mapping happens to be a list or dict.

Usage::

    mapping = TrackerMapping.from_yaml(Path("rtmpose_body_to_canonical_mapping.yaml"))
    canonical = mapping.apply(tracker_positions)
    # canonical uses canonical landmark names, ready for FABRIK/CoM/etc.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Mapping forms
# ---------------------------------------------------------------------------

# A single mapping entry: str=passthrough, list=mean, dict=weighted sum
MappingEntry = Union[str, List[str], Dict[str, float]]


# ---------------------------------------------------------------------------
# TrackerMapping
# ---------------------------------------------------------------------------


class TrackerMapping:
    """Load and apply a tracker→canonical landmark mapping.

    Parameters
    ----------
    entries : dict
        Mapping from canonical landmark name to its definition
        (string, list of strings, or dict of name→weight pairs).
    prefix : str or None
        If set, strip this prefix from tracker keypoint names before
        looking them up.  For example ``prefix="right_hand_"`` strips
        the prefix so that ``right_hand_root`` becomes ``root`` which
        matches the unprefixed mapping entry.
    """

    def __init__(
        self,
        entries: Dict[str, MappingEntry],
        prefix: Optional[str] = None,
    ) -> None:
        self._entries: Dict[str, MappingEntry] = {}
        self._prefix = prefix or ""

        for canonical_name, entry in entries.items():
            if isinstance(entry, str):
                self._entries[canonical_name] = entry
            elif isinstance(entry, list):
                if len(entry) == 0:
                    raise ValueError(
                        f"List mapping for '{canonical_name}' is empty"
                    )
                self._entries[canonical_name] = tuple(entry)
            elif isinstance(entry, dict):
                if len(entry) == 0:
                    raise ValueError(
                        f"Dict mapping for '{canonical_name}' is empty"
                    )
                self._entries[canonical_name] = dict(entry)
            else:
                raise TypeError(
                    f"Mapping entry for '{canonical_name}' must be str, list, or dict, "
                    f"got {type(entry).__name__}"
                )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        *,
        prefix: Optional[str] = None,
    ) -> "TrackerMapping":
        """Load a mapping from a YAML file.

        Parameters
        ----------
        yaml_path : Path
            Path to a YAML file whose top-level keys are canonical
            landmark names and values are string/list/dict entries.
        prefix : str or None
            Strip this prefix from tracker keypoint names during
            :meth:`apply`.  Useful for hand mappings where the tracker
            produces ``right_hand_root`` but the mapping defines
            ``root`` → ``wrist``.
        """
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError(
                f"Mapping YAML must be a dict at top level, got {type(data).__name__}"
            )
        return cls(entries=data, prefix=prefix)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        tracker_positions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Produce canonical landmark positions from tracker positions.

        Parameters
        ----------
        tracker_positions : dict of str → (3,) ndarray
            Raw keypoint positions with tracker-specific names.

        Returns
        -------
        dict of str → (3,) ndarray
            Positions keyed by canonical landmark names.  Landmarks
            whose tracker source is missing are silently omitted.
        """
        result: Dict[str, np.ndarray] = {}
        prefix = self._prefix

        for canonical_name, entry in self._entries.items():
            if isinstance(entry, str):
                # ---- 1:1 passthrough ----
                tracker_name = prefix + entry
                pos = tracker_positions.get(tracker_name)
                if pos is not None:
                    result[canonical_name] = np.asarray(pos, dtype=np.float64)

            elif isinstance(entry, tuple):
                # ---- unweighted mean ----
                positions: List[np.ndarray] = []
                for name in entry:
                    pos = tracker_positions.get(prefix + name)
                    if pos is not None:
                        positions.append(np.asarray(pos, dtype=np.float64))
                if positions:
                    result[canonical_name] = np.mean(
                        np.column_stack(positions), axis=1
                    )

            elif isinstance(entry, dict):
                # ---- weighted sum ----
                weighted: List[np.ndarray] = []
                total_weight = 0.0
                for name, weight in entry.items():
                    pos = tracker_positions.get(prefix + name)
                    if pos is not None:
                        weighted.append(np.asarray(pos, dtype=np.float64) * weight)
                        total_weight += weight
                if weighted and total_weight > 0.0:
                    result[canonical_name] = sum(weighted) / total_weight

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def canonical_names(self) -> List[str]:
        """Ordered list of canonical landmark names this mapping produces."""
        return list(self._entries.keys())

    @property
    def tracker_names(self) -> List[str]:
        """Tracker keypoint names referenced by this mapping (with prefix)."""
        names: List[str] = []
        prefix = self._prefix
        for entry in self._entries.values():
            if isinstance(entry, str):
                names.append(prefix + entry)
            elif isinstance(entry, tuple):
                names.extend(prefix + n for n in entry)
            elif isinstance(entry, dict):
                names.extend(prefix + n for n in entry)
        return names

    def __repr__(self) -> str:
        return (
            f"TrackerMapping({len(self._entries)} entries"
            + (f", prefix='{self._prefix}'" if self._prefix else "")
            + ")"
        )
