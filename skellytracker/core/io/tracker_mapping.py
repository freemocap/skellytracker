"""Four-form tracker→canonical landmark mapping.

A ``TrackerMapping`` loads a YAML file that defines, for each canonical
landmark, how to produce its position from the tracker's keypoints.
Four forms are supported::

    string    →  1:1 passthrough  ``left_elbow: "left_elbow"``
    list      →  unweighted mean  ``hips_center: ["left_hip", "right_hip"]``
    dict      →  weighted sum     ``head_center: {left_ear: 0.5, right_ear: 0.5}``
    dict      →  anatomical_offset  off-surface joint center via local frame
                  (detected by ``form: anatomical_offset`` key)

Every canonical landmark is produced this way — including computed ones
like ``neck_center`` and ``hips_center``, whose mapping is a list or dict,
and off-surface joint centers like the sternoclavicular and glenohumeral
joints, whose mapping is an ``anatomical_offset``.

Usage::

    mapping = TrackerMapping.from_yaml(Path("rtmpose_body_to_canonical_mapping.yaml"))
    canonical = mapping.apply(tracker_positions)
    # canonical uses canonical landmark names, ready for FABRIK/CoM/etc.

anatomical_offset form
----------------------
Places a joint center that sits *off* the marked surface — the anterior
sternoclavicular joint (clavicle base), glenohumeral joint, hip joint
centers.  These are real and required; the three convex forms (string,
list, dict) cannot produce a point outside the keypoints' convex hull.

The form is deterministic and subject-scaled — no runtime fitting.
It is a dict with these keys::

    form: anatomical_offset
    origin: [keypoint, ...]              # mean → frame origin
    frame:
      <axis_name>:
        from: keypoint | [keypoint, ...]
        to:   keypoint | [keypoint, ...]
        kind: exact | approximate
      <axis_name>:
        ...
    offset: { <axis_name>: ratio, ... }  # ratios of reference_length
    reference_length: <named_length>     # or {from: ..., to: ...}

Exactly TWO frame axes must be defined: one ``exact`` and one
``approximate``.  The third axis is computed via right-handed cross
product.  The offset vector is assembled in the frame's basis and
scaled by ``reference_length`` (computed from the current frame's
keypoints, so it scales with the subject).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Mapping forms
# ---------------------------------------------------------------------------

MappingEntry = Union[str, List[str], Dict[str, float], Dict[str, Any]]


# ---------------------------------------------------------------------------
# anatomical_offset internal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameAxisDef:
    """Definition of one axis in an anatomical frame."""

    name: str  # axis name from the YAML (e.g. "up", "lateral", "anterior")
    from_keypoints: list[str]
    to_keypoints: list[str]
    kind: str  # "exact" | "approximate"


@dataclass(frozen=True)
class _AnatomicalOffsetDef:
    """Parsed anatomical_offset mapping entry."""

    canonical_name: str
    origin_keypoints: list[str]
    axes: list[_FrameAxisDef]  # exactly 2: one exact, one approximate
    offset_ratios: dict[str, float]  # axis_name → ratio
    reference_length_from: list[str]
    reference_length_to: list[str]  # may be empty (named reference)


# ---------------------------------------------------------------------------
# TrackerMapping
# ---------------------------------------------------------------------------


class TrackerMapping:
    """Load and apply a tracker→canonical landmark mapping.

    Parameters
    ----------
    entries : dict
        Mapping from canonical landmark name to its definition
        (string, list of strings, dict of name→weight pairs, or
        dict with ``form: anatomical_offset``).
    prefix : str or None
        If set, strip this prefix from tracker keypoint names before
        looking them up.
    """

    def __init__(
        self,
        entries: Dict[str, MappingEntry],
        prefix: Optional[str] = None,
    ) -> None:
        self._entries: Dict[str, MappingEntry] = {}
        self._anatomical_offsets: Dict[str, _AnatomicalOffsetDef] = {}
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
                if entry.get("form") == "anatomical_offset":
                    self._anatomical_offsets[canonical_name] = (
                        _parse_anatomical_offset(canonical_name, entry)
                    )
                else:
                    if len(entry) == 0:
                        raise ValueError(
                            f"Dict mapping for '{canonical_name}' is empty"
                        )
                    self._entries[canonical_name] = dict(entry)
            else:
                raise TypeError(
                    f"Mapping entry for '{canonical_name}' must be str, "
                    f"list, or dict, got {type(entry).__name__}"
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
        """Load a mapping from a YAML file."""
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError(
                f"Mapping YAML must be a dict at top level, "
                f"got {type(data).__name__}"
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

        # ── Pass 1: string / list / weighted-sum forms ──────────
        for canonical_name, entry in self._entries.items():
            if isinstance(entry, str):
                tracker_name = prefix + entry
                pos = tracker_positions.get(tracker_name)
                if pos is not None:
                    result[canonical_name] = np.asarray(
                        pos, dtype=np.float64
                    )
            elif isinstance(entry, tuple):
                positions: List[np.ndarray] = []
                for name in entry:
                    pos = tracker_positions.get(prefix + name)
                    if pos is not None:
                        positions.append(
                            np.asarray(pos, dtype=np.float64)
                        )
                if positions:
                    result[canonical_name] = np.mean(
                        np.column_stack(positions), axis=1
                    )
            elif isinstance(entry, dict):
                weighted: List[np.ndarray] = []
                total_weight = 0.0
                for name, weight in entry.items():
                    pos = tracker_positions.get(prefix + name)
                    if pos is not None:
                        weighted.append(
                            np.asarray(pos, dtype=np.float64) * weight
                        )
                        total_weight += weight
                if weighted and total_weight > 0.0:
                    result[canonical_name] = sum(weighted) / total_weight

        # ── Pass 2: anatomical_offset form ───────────────────────
        # anatomical_offsets may reference other canonical landmarks
        # (e.g. hips_center, neck_center) that were computed in pass 1.
        # We merge raw tracker positions + pass-1 results so the offset
        # resolver can find both.
        combined_positions = {**tracker_positions, **result}
        for canonical_name, offset_def in self._anatomical_offsets.items():
            pos = _apply_anatomical_offset(
                offset_def, combined_positions, prefix
            )
            if pos is not None:
                result[canonical_name] = pos

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def canonical_names(self) -> List[str]:
        """Ordered list of canonical landmark names this mapping produces."""
        names: List[str] = list(self._entries.keys())
        names.extend(self._anatomical_offsets.keys())
        return names

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
        for offset_def in self._anatomical_offsets.values():
            names.extend(
                prefix + n for n in offset_def.origin_keypoints
            )
            for axis in offset_def.axes:
                names.extend(prefix + n for n in axis.from_keypoints)
                names.extend(prefix + n for n in axis.to_keypoints)
        return names

    def __repr__(self) -> str:
        n_entries = len(self._entries)
        n_offsets = len(self._anatomical_offsets)
        parts = [f"{n_entries} basic entries"]
        if n_offsets:
            parts.append(f"{n_offsets} anatomical_offsets")
        if self._prefix:
            parts.append(f"prefix='{self._prefix}'")
        return f"TrackerMapping({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════
# anatomical_offset: parsing
# ═══════════════════════════════════════════════════════════════════════


def _parse_anatomical_offset(
    canonical_name: str, entry: dict[str, Any]
) -> _AnatomicalOffsetDef:
    """Parse and validate an ``anatomical_offset`` mapping entry."""
    _err = lambda msg: _mapping_error(canonical_name, msg)

    origin = entry.get("origin")
    if not isinstance(origin, list) or len(origin) == 0:
        raise _err("'origin' must be a non-empty list of keypoint names")

    frame = entry.get("frame")
    if not isinstance(frame, dict):
        raise _err("'frame' must be a dict of axis definitions")

    axes: list[_FrameAxisDef] = []
    for axis_name, axis_def in frame.items():
        if not isinstance(axis_def, dict):
            raise _err(f"frame axis '{axis_name}' must be a dict")
        kind = axis_def.get("kind")
        if kind not in ("exact", "approximate"):
            raise _err(
                f"axis '{axis_name}' kind must be 'exact' or "
                f"'approximate', got {kind!r}"
            )
        from_kps = _normalize_keypoint_list(axis_def.get("from"))
        to_kps = _normalize_keypoint_list(axis_def.get("to"))
        if not from_kps or not to_kps:
            raise _err(
                f"axis '{axis_name}' must have non-empty 'from' and 'to'"
            )
        axes.append(
            _FrameAxisDef(
                name=axis_name,
                from_keypoints=from_kps,
                to_keypoints=to_kps,
                kind=kind,
            )
        )

    if len(axes) != 2:
        raise _err(f"exactly 2 frame axes required, got {len(axes)}")
    kinds = {ax.kind for ax in axes}
    if kinds != {"exact", "approximate"}:
        raise _err(
            f"must have one 'exact' and one 'approximate' axis, "
            f"got {sorted(kinds)}"
        )

    offset = entry.get("offset")
    if not isinstance(offset, dict) or len(offset) == 0:
        raise _err(
            "'offset' must be a non-empty dict of axis_name -> ratio"
        )
    offset_ratios: dict[str, float] = {}
    for axis_name, ratio in offset.items():
        if not isinstance(ratio, (int, float)):
            raise _err(
                f"offset ratio '{axis_name}' must be a number, "
                f"got {type(ratio).__name__}"
            )
        offset_ratios[axis_name] = float(ratio)

    ref_len = entry.get("reference_length")
    ref_from: list[str]
    ref_to: list[str]
    if isinstance(ref_len, str):
        ref_from, ref_to = _resolve_named_length(ref_len)
    elif isinstance(ref_len, dict):
        ref_from = _normalize_keypoint_list(ref_len.get("from"))
        ref_to = _normalize_keypoint_list(ref_len.get("to"))
        if not ref_from or not ref_to:
            raise _err(
                "reference_length dict must have 'from' and 'to' "
                "keypoint lists"
            )
    else:
        raise _err(
            f"'reference_length' must be a string or dict, "
            f"got {type(ref_len).__name__}"
        )

    return _AnatomicalOffsetDef(
        canonical_name=canonical_name,
        origin_keypoints=origin,
        axes=axes,
        offset_ratios=offset_ratios,
        reference_length_from=ref_from,
        reference_length_to=ref_to,
    )


def _mapping_error(canonical_name: str, detail: str) -> ValueError:
    return ValueError(
        f"anatomical_offset '{canonical_name}': {detail}"
    )


# ═══════════════════════════════════════════════════════════════════════
# anatomical_offset: application
# ═══════════════════════════════════════════════════════════════════════


def _apply_anatomical_offset(
    offset_def: _AnatomicalOffsetDef,
    tracker_positions: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray | None:
    """Compute a landmark position via anatomical offset.

    Returns ``None`` if any required keypoint is missing or degenerate.
    """
    # ── 1. Resolve origin ───────────────────────────────────────
    origin = _mean_position(
        offset_def.origin_keypoints, tracker_positions, prefix
    )
    if origin is None:
        return None

    # ── 2. Compute reference length ─────────────────────────────
    ref_length = _compute_distance(
        offset_def.reference_length_from,
        offset_def.reference_length_to,
        tracker_positions,
        prefix,
    )
    if ref_length is None or ref_length < 1e-10:
        return None

    # ── 3. Build frame axis vectors ─────────────────────────────
    # axis_name → unit vector (from mean → to mean)
    axis_vecs: dict[str, np.ndarray] = {}
    axis_kinds: dict[str, str] = {}  # axis_name → "exact"|"approximate"
    for axis in offset_def.axes:
        vec = _direction_vector(
            axis.from_keypoints, axis.to_keypoints,
            tracker_positions, prefix,
        )
        if vec is None:
            return None
        axis_vecs[axis.name] = vec
        axis_kinds[axis.name] = axis.kind

    # Find which axis is exact and which is approximate
    exact_name = next(
        n for n, k in axis_kinds.items() if k == "exact"
    )
    approx_name = next(
        n for n, k in axis_kinds.items() if k == "approximate"
    )
    exact_vec = axis_vecs[exact_name]
    approx_vec = axis_vecs[approx_name]

    # Gram-Schmidt orthogonalization
    dot = float(np.dot(exact_vec, approx_vec))
    if abs(dot) > 0.9998:
        return None  # nearly parallel — can't build a valid frame
    approx_orth = approx_vec - dot * exact_vec
    approx_orth = approx_orth / np.linalg.norm(approx_orth)

    # Third axis via right-handed cross product
    third_vec = np.cross(exact_vec, approx_orth)
    third_vec = third_vec / np.linalg.norm(third_vec)

    # Determine the name of the third axis.  The frame has axis names
    # (e.g. "up", "lateral", "anterior").  Two of those are defined
    # in the YAML; the third is implicit.  We find it by looking at
    # which axis names appear in the offset_ratios dict.
    defined_names = {exact_name, approx_name}
    offset_names = set(offset_def.offset_ratios.keys())
    third_name_candidates = offset_names - defined_names
    # Also check: the third axis name might not be in the offset
    # (if the offset doesn't use it).  In that case we just pick a
    # synthetic name.
    third_name = (
        third_name_candidates.pop()
        if len(third_name_candidates) == 1
        else "_third"
    )

    # ── 4. Assemble basis: axis_name → unit vector ──────────────
    basis: dict[str, np.ndarray] = {
        exact_name: exact_vec,
        approx_name: approx_orth,
        third_name: third_vec,
    }

    # ── 5. Apply offset ─────────────────────────────────────────
    offset_vec = np.zeros(3, dtype=np.float64)
    for axis_name, ratio in offset_def.offset_ratios.items():
        direction = basis.get(axis_name)
        if direction is None:
            # The ratio references an axis not in our basis — skip
            continue
        offset_vec = offset_vec + ratio * ref_length * direction

    return origin + offset_vec


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _mean_position(
    keypoint_names: list[str],
    positions: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray | None:
    """Compute the mean position of named keypoints, or None if any missing."""
    pts: list[np.ndarray] = []
    for name in keypoint_names:
        pos = positions.get(prefix + name)
        if pos is not None:
            pts.append(np.asarray(pos, dtype=np.float64))
    if not pts:
        return None
    return np.mean(np.column_stack(pts), axis=1)


def _direction_vector(
    from_names: list[str],
    to_names: list[str],
    positions: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray | None:
    """Compute a unit direction vector (from_mean → to_mean), or None."""
    frm = _mean_position(from_names, positions, prefix)
    to = _mean_position(to_names, positions, prefix)
    if frm is None or to is None:
        return None
    vec = to - frm
    norm = float(np.linalg.norm(vec))
    if norm < 1e-10:
        return None
    return vec / norm


def _compute_distance(
    from_names: list[str],
    to_names: list[str],
    positions: dict[str, np.ndarray],
    prefix: str,
) -> float | None:
    """Compute the Euclidean distance between two mean positions."""
    frm = _mean_position(from_names, positions, prefix)
    to = _mean_position(to_names, positions, prefix)
    if frm is None or to is None:
        return None
    return float(np.linalg.norm(to - frm))


def _resolve_named_length(
    name: str,
) -> tuple[list[str], list[str]]:
    """Resolve a named reference length to from/to keypoint pairs."""
    _NAMED_LENGTHS: dict[str, tuple[list[str], list[str]]] = {
        "shoulder_width": (
            ["left_shoulder"],
            ["right_shoulder"],
        ),
        "hip_width": (
            ["left_hip"],
            ["right_hip"],
        ),
    }
    if name not in _NAMED_LENGTHS:
        raise ValueError(
            f"Unknown named reference_length {name!r}. "
            f"Known: {sorted(_NAMED_LENGTHS.keys())}"
        )
    return _NAMED_LENGTHS[name]


def _normalize_keypoint_list(value: Any) -> list[str]:
    """Normalize a keypoint or list of keypoints to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]
