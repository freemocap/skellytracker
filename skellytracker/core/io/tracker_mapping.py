"""Four-form tracker→standard-human landmark mapping.

A ``TrackerMapping`` loads a YAML file that defines, for each standard-human
landmark, how to hydrate its position from the tracker's keypoints.
Four forms are supported::

    string    →  1:1 passthrough  ``left_elbow: "left_elbow"``
    list      →  unweighted mean  ``pelvis_origin: ["left_hip", "right_hip"]``
    dict      →  weighted sum     ``head_center: {left_ear: 0.5, right_ear: 0.5}``
    dict      →  anatomical_offset  off-surface joint center via local frame
                  (detected by ``form: anatomical_offset`` key)

Every standard-human landmark is produced this way — including computed ones
like ``head_center`` and ``pelvis_origin``, whose mapping is a list or dict,
and off-surface joint centers like the sternoclavicular and glenohumeral
joints, whose mapping is an ``anatomical_offset``.

Usage::

    mapping = TrackerMapping.from_yaml(Path("rtmpose_body_to_standard_human_mapping.yaml"))
    landmarks = mapping.apply(tracker_positions)
    # landmarks use the standard-human landmark names, ready for FABRIK/CoM/etc.

anatomical_offset form
----------------------
Places a joint center that sits *off* the marked surface — the anterior
sternoclavicular joint (clavicle base), glenohumeral joint, hip joint
centers.  These are real and required; the three convex forms (string,
list, dict) cannot produce a point outside the tracker keypoints' convex hull.

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
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Mapping forms
# ---------------------------------------------------------------------------

MappingEntry = str | list[str] | dict[str, float] | dict[str, Any]

PASSTHROUGH_KEY = "passthrough_keypoints_as_landmarks"
"""The whole file, for an object whose markers ARE its landmarks.

A charuco board has no anatomy to map onto: corner 7 is corner 7. Authoring a line per
marker would be duplication with a chance of typos, and it would have to be regenerated
every time a board size changed. The flag says "every keypoint is a landmark of the same
name" once, and is the reusable answer for any simple tracked object.
"""


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

    landmark_name: str
    origin_keypoints: list[str]
    axes: list[_FrameAxisDef]  # exactly 2: one exact, one approximate
    offset_ratios: dict[str, float]  # axis_name → ratio
    reference_length_from: list[str]
    reference_length_to: list[str]  # may be empty (named reference)


# ---------------------------------------------------------------------------
# TrackerMapping
# ---------------------------------------------------------------------------


class TrackerMapping:
    """Load and apply a tracker→standard-human landmark mapping.

    Parameters
    ----------
    entries : dict
        Mapping from standard-human landmark name to its definition
        (string, list of strings, dict of name→weight pairs, or
        dict with ``form: anatomical_offset``).  Keys are landmark names;
        values name the tracker keypoints used to hydrate them.
    prefix : str or None
        If set, strip this prefix from tracker keypoint names before
        looking them up.
    known_tracker_keypoints : set of str or None
        The set of tracker keypoint names the tracker actually produces.
        When provided, every tracker-side name this mapping references is
        checked against it at load time and a name the tracker NEVER
        produces raises.  A keypoint missing THIS frame (occlusion) is
        still a silent skip at apply time.
    """

    def __init__(
        self,
        entries: dict[str, MappingEntry],
        prefix: str | None = None,
        known_tracker_keypoints: set[str] | None = None,
        passthrough_keypoints_as_landmarks: bool = False,
    ) -> None:
        self._entries: dict[str, MappingEntry] = {}
        self._anatomical_offsets: dict[str, _AnatomicalOffsetDef] = {}
        self._prefix = prefix or ""
        self._passthrough = passthrough_keypoints_as_landmarks

        if self._passthrough:
            if entries:
                raise ValueError(
                    "a pass-through mapping declares that every keypoint IS a landmark, so "
                    "it cannot also list entries - got "
                    f"{sorted(entries)[:8]}. Drop either the flag or the entries."
                )
            if not known_tracker_keypoints:
                raise ValueError(
                    "a pass-through mapping produces exactly the keypoints its tracker "
                    "emits, so it needs `known_tracker_keypoints` to be able to say what "
                    "it produces. Without them it could only answer at apply time, and "
                    "callers that must decide up front (which landmarks are measured, "
                    "which segments may set the model scale) would have nothing to read."
                )
            self._passthrough_landmark_names = frozenset(
                self._strip_prefix(name) for name in known_tracker_keypoints
            )

        for landmark_name, entry in entries.items():
            if isinstance(entry, str):
                self._entries[landmark_name] = entry
            elif isinstance(entry, list):
                if len(entry) == 0:
                    raise ValueError(
                        f"List mapping for '{landmark_name}' is empty"
                    )
                self._entries[landmark_name] = tuple(entry)
            elif isinstance(entry, dict):
                if entry.get("form") == "anatomical_offset":
                    self._anatomical_offsets[landmark_name] = (
                        _parse_anatomical_offset(landmark_name, entry)
                    )
                else:
                    if len(entry) == 0:
                        raise ValueError(
                            f"Dict mapping for '{landmark_name}' is empty"
                        )
                    self._entries[landmark_name] = dict(entry)
            else:
                raise TypeError(
                    f"Mapping entry for '{landmark_name}' must be str, "
                    f"list, or dict, got {type(entry).__name__}"
                )

        if known_tracker_keypoints is not None:
            offenders = sorted(
                self._referenced_tracker_names() - known_tracker_keypoints
            )
            if offenders:
                raise ValueError(
                    "mapping references tracker keypoints the tracker "
                    "never produces: " + ", ".join(offenders)
                )

    def _strip_prefix(self, name: str) -> str:
        """A tracker-side name with this mapping's prefix removed, if it carries one."""
        if self._prefix and name.startswith(self._prefix):
            return name[len(self._prefix):]
        return name

    @property
    def is_passthrough(self) -> bool:
        """Whether every tracked keypoint is a landmark of the same name."""
        return self._passthrough

    @property
    def directly_measured_landmark_names(self) -> frozenset[str]:
        """The landmarks this mapping MEASURES, as opposed to constructs.

        Every non-offset form — a passthrough, a mean of keypoints, a weighted sum — is an
        affine combination of measured keypoints with constant coefficients, so it carries
        the subject's real geometry: the distance between two of them is a distance on the
        subject.  An ``anatomical_offset`` is not.  It places a landmark at
        ``ratio x reference_length`` along an authored direction, so the distance between
        two of them is an authored ratio times a span already measured elsewhere — the
        template quoting itself back.

        The distinction matters to anything inferring the subject's SIZE from the mapped
        landmarks (SkellyForge's body-scale fit): constructed landmarks are near
        noise-free, so a consistency-weighted estimator would rank them as its best
        evidence, which is exactly backwards.  They are perfectly good POSITIONS; they are
        not independent evidence about scale.

        A pass-through mapping measures everything: its landmarks ARE its keypoints, so
        there is nothing constructed to exclude.
        """
        if self._passthrough:
            return self._passthrough_landmark_names
        return frozenset(self._entries)

    def _referenced_tracker_names(self) -> set[str]:
        """Every tracker-side name referenced by this mapping (prefix applied).

        Named reference lengths like ``shoulder_width`` are NOT tracker names
        (they resolve to keypoint pairs at apply time) and are excluded.
        """
        names: set[str] = set()
        prefix = self._prefix
        for entry in self._entries.values():
            if isinstance(entry, str):
                names.add(prefix + entry)
            elif isinstance(entry, tuple):
                names.update(prefix + n for n in entry)
            elif isinstance(entry, dict):
                names.update(prefix + n for n in entry)
        for offset_def in self._anatomical_offsets.values():
            names.update(
                prefix + n for n in offset_def.origin_keypoints
            )
            for axis in offset_def.axes:
                names.update(prefix + n for n in axis.from_keypoints)
                names.update(prefix + n for n in axis.to_keypoints)
            names.update(
                prefix + n for n in offset_def.reference_length_from
            )
            names.update(
                prefix + n for n in offset_def.reference_length_to
            )
        return names

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path,
        *,
        prefix: str | None = None,
        known_tracker_keypoints: set[str] | None = None,
    ) -> "TrackerMapping":
        """Load a mapping from a YAML file."""
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise TypeError(
                f"Mapping YAML must be a dict at top level, "
                f"got {type(data).__name__}"
            )
        entries = dict(data)
        passthrough = bool(entries.pop(PASSTHROUGH_KEY, False))
        return cls(
            entries=entries,
            prefix=prefix,
            known_tracker_keypoints=known_tracker_keypoints,
            passthrough_keypoints_as_landmarks=passthrough,
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        tracker_positions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Produce standard-human landmark positions from tracker positions.

        Parameters
        ----------
        tracker_positions : dict of str → (3,) ndarray
            Raw keypoint positions with tracker-specific names.

        Returns
        -------
        dict of str → (3,) ndarray
            Positions keyed by standard-human landmark name.  Landmarks whose
            tracker source is missing are silently omitted.
        """
        if self._passthrough:
            # Every keypoint is a landmark of the same name. Restricted to the declared
            # set so an unexpected name is dropped here rather than becoming a landmark
            # nothing downstream has ever heard of.
            return {
                self._strip_prefix(name): np.asarray(position, dtype=np.float64)
                for name, position in tracker_positions.items()
                if self._strip_prefix(name) in self._passthrough_landmark_names
            }

        result: dict[str, np.ndarray] = {}
        prefix = self._prefix

        # ── Pass 1: string / list / weighted-sum forms ──────────
        for landmark_name, entry in self._entries.items():
            if isinstance(entry, str):
                tracker_name = prefix + entry
                pos = tracker_positions.get(tracker_name)
                if pos is not None:
                    result[landmark_name] = np.asarray(
                        pos, dtype=np.float64
                    )
            elif isinstance(entry, tuple):
                positions: list[np.ndarray] = []
                complete = True
                for name in entry:
                    pos = tracker_positions.get(prefix + name)
                    if pos is None:
                        complete = False
                        break
                    positions.append(np.asarray(pos, dtype=np.float64))
                # A mean requires ALL its inputs: a partial mean would silently
                # relabel a different point. Missing any -> omit (occlusion is data).
                if complete and positions:
                    result[landmark_name] = np.mean(
                        np.column_stack(positions), axis=1
                    )
            elif isinstance(entry, dict):
                weighted: list[np.ndarray] = []
                total_weight = 0.0
                complete = True
                for name, weight in entry.items():
                    pos = tracker_positions.get(prefix + name)
                    if pos is None:
                        complete = False
                        break
                    weighted.append(np.asarray(pos, dtype=np.float64) * weight)
                    total_weight += weight
                # All weighted inputs required: renormalizing over survivors would
                # silently change what the landmark means. Missing any -> omit.
                if complete and weighted and total_weight > 0.0:
                    result[landmark_name] = sum(weighted) / total_weight

        # ── Pass 2: anatomical_offset form ───────────────────────
        # anatomical_offsets may reference other standard-human landmarks
        # (e.g. pelvis_origin, head_center) that were computed in pass 1.
        # We merge raw tracker positions + pass-1 results so the offset
        # resolver can find both.
        combined_positions = {**tracker_positions, **result}
        for landmark_name, offset_def in self._anatomical_offsets.items():
            pos = _apply_anatomical_offset(
                offset_def, combined_positions, prefix
            )
            if pos is not None:
                result[landmark_name] = pos

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def landmark_names(self) -> list[str]:
        """The standard-human landmark names this mapping produces, in order."""
        names: list[str] = list(self._entries.keys())
        names.extend(self._anatomical_offsets.keys())
        return names

    @property
    def tracker_names(self) -> list[str]:
        """Tracker keypoint names referenced by this mapping (with prefix)."""
        names: list[str] = []
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
    landmark_name: str, entry: dict[str, Any]
) -> _AnatomicalOffsetDef:
    """Parse and validate an ``anatomical_offset`` mapping entry."""
    _err = lambda msg: _mapping_error(landmark_name, msg)

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
        landmark_name=landmark_name,
        origin_keypoints=origin,
        axes=axes,
        offset_ratios=offset_ratios,
        reference_length_from=ref_from,
        reference_length_to=ref_to,
    )


def _mapping_error(landmark_name: str, detail: str) -> ValueError:
    return ValueError(
        f"anatomical_offset '{landmark_name}': {detail}"
    )


# ═══════════════════════════════════════════════════════════════════════
# anatomical_offset: application
# ═══════════════════════════════════════════════════════════════════════


def _apply_anatomical_offset(
    offset_def: _AnatomicalOffsetDef,
    tracker_positions: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray | None:
    """Compute a derived landmark position via anatomical offset.

    Returns ``None`` if any required tracker keypoint is missing or degenerate.
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

    # Third axis via right-handed cross product.
    #
    # The frame has three axis names (e.g. "up", "lateral", "anterior"); the YAML
    # names two of them and the third is implicit, identified as the offset name
    # that is not one of the two declared axes.
    defined_names = {exact_name, approx_name}
    offset_names = set(offset_def.offset_ratios.keys())
    undeclared_offset_names = offset_names - defined_names

    if len(undeclared_offset_names) > 1:
        raise ValueError(
            f"Anatomical offset for {offset_def.landmark_name!r} references "
            f"{sorted(undeclared_offset_names)} beyond the declared frame axes "
            f"{sorted(defined_names)}. A frame has exactly three axes, so at most "
            f"one offset component may name the implicit third axis."
        )

    third_name = (
        undeclared_offset_names.pop() if undeclared_offset_names else "_implicit_third"
    )

    # ── 4. Assemble basis: axis_name → unit vector ──────────────
    basis: dict[str, np.ndarray] = {
        exact_name: exact_vec,
        approx_name: approx_orth,
        third_name: third_vec,
    }

    # ── 5. Apply offset ─────────────────────────────────────────
    # Every named component must resolve to a basis axis. Silently skipping an
    # unrecognized name would place the landmark somewhere plausible-looking but
    # wrong, with nothing to notice — a typo'd axis must fail, not degrade.
    offset_vec = np.zeros(3, dtype=np.float64)
    for axis_name, ratio in offset_def.offset_ratios.items():
        direction = basis.get(axis_name)
        if direction is None:
            raise ValueError(
                f"Anatomical offset for {offset_def.landmark_name!r} names axis "
                f"{axis_name!r}, which is not part of its frame "
                f"{sorted(basis)}."
            )
        offset_vec = offset_vec + ratio * ref_length * direction

    return origin + offset_vec


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _mean_position(
    names: list[str],
    positions: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray | None:
    """Compute the mean position of named tracker keypoints, or None if ANY is
    missing. A partial mean would silently relabel a different point (e.g. one
    hip returned as ``pelvis_origin``), so an incomplete input yields no result."""
    pts: list[np.ndarray] = []
    for name in names:
        pos = positions.get(prefix + name)
        if pos is None:
            return None
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
        "eye_width": (
            ["left_eye"],
            ["right_eye"],
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
