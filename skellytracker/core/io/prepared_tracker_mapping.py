"""Prepared named mapping plans; reusable, stateless and safe across frame calls."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from skellytracker.core.io.tracker_mapping_batch import MappedLandmarkBatch


def _fixed(values, dtype=None):
    result = np.asarray(values, dtype=dtype)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class _Reduction:
    targets: np.ndarray
    sources: np.ndarray
    weights: np.ndarray | None

    def apply(self, points, output):
        values = points[:, self.sources, :]
        if self.weights is None:
            reduced = np.mean(values, axis=2)
        else:
            reduced = (
                np.cumsum(values * self.weights[None, :, :, None], axis=2)[:, :, -1]
                / np.cumsum(self.weights, axis=1)[None, :, -1, None]
            )
        output[:, self.targets] = reduced


def _reductions(entries):
    groups = {}
    for target, sources, weights in entries:
        groups.setdefault((len(sources), weights is not None), []).append(
            (target, sources, weights)
        )
    return tuple(
        _Reduction(
            _fixed([x[0] for x in group], int),
            _fixed([x[1] for x in group], int),
            _fixed([x[2] for x in group], float) if weighted else None,
        )
        for (_, weighted), group in groups.items()
    )


@dataclass(frozen=True)
class _Component:
    names: tuple[str, ...]
    affine_count: int
    affine: tuple[_Reduction, ...]
    fallback_raw: np.ndarray
    fallback_affine: np.ndarray
    means: tuple[_Reduction, ...]
    offset_count: int
    offset_axes: np.ndarray
    offset_ratios: np.ndarray

    def apply(self, points):
        frames = len(points)
        output = np.full((frames, len(self.names), 3), np.nan)
        for reduction in self.affine:
            reduction.apply(points, output)
        if not self.offset_count:
            return output
        combined = np.concatenate((points, output[:, : self.affine_count]), axis=1)
        if len(self.fallback_raw):
            mapped = combined[:, self.fallback_affine]
            combined[:, self.fallback_affine] = np.where(
                np.isfinite(mapped).all(axis=-1)[..., None],
                mapped,
                points[:, self.fallback_raw],
            )
        means = np.full((frames, self.offset_count * 7, 3), np.nan)
        for reduction in self.means:
            reduction.apply(combined, means)
        means = means.reshape(frames, self.offset_count, 7, 3)
        origin = means[:, :, 0]
        length = np.linalg.norm(means[:, :, 2] - means[:, :, 1], axis=-1)
        axes = np.stack(
            (means[:, :, 4] - means[:, :, 3], means[:, :, 6] - means[:, :, 5]), axis=2
        )
        norms = np.linalg.norm(axes, axis=-1)
        valid = (
            np.isfinite(means).all(axis=(-1, -2))
            & (length >= 1e-10)
            & (norms >= 1e-10).all(axis=-1)
        )
        axes = axes / np.where(norms >= 1e-10, norms, 1)[..., None]
        exact, approximate = axes[:, :, 0], axes[:, :, 1]
        dot = np.einsum("foc,foc->fo", exact, approximate)
        valid &= np.abs(dot) <= 0.9998
        orthogonal = approximate - dot[..., None] * exact
        norm = np.linalg.norm(orthogonal, axis=-1)
        orthogonal /= np.where(norm > 0, norm, 1)[..., None]
        third = np.cross(exact, orthogonal)
        norm = np.linalg.norm(third, axis=-1)
        third /= np.where(norm > 0, norm, 1)[..., None]
        basis = np.stack((exact, orthogonal, third), axis=2)
        selected = basis[:, np.arange(self.offset_count)[:, None], self.offset_axes]
        contributions = (
            self.offset_ratios[None, :, :, None] * length[:, :, None, None] * selected
        )
        offset = np.cumsum(contributions, axis=2)[:, :, -1]
        output[:, self.affine_count :] = np.where(
            valid[..., None], origin + offset, np.nan
        )
        return output

    def quality(self, input_quality, mapped):
        frames = len(input_quality)
        result = np.full((frames, len(self.names)), np.nan)
        for reduction in self.affine:
            result[:, reduction.targets] = np.min(
                input_quality[:, reduction.sources], axis=-1
            )
        if self.offset_count:
            combined = np.concatenate(
                (input_quality, result[:, : self.affine_count]), axis=1
            )
            if len(self.fallback_raw):
                available = np.isfinite(
                    mapped[:, self.fallback_affine - input_quality.shape[1]]
                ).all(axis=-1)
                combined[:, self.fallback_affine] = np.where(
                    available,
                    combined[:, self.fallback_affine],
                    input_quality[:, self.fallback_raw],
                )
            means = np.full((frames, self.offset_count * 7), np.nan)
            for reduction in self.means:
                means[:, reduction.targets] = np.min(
                    combined[:, reduction.sources], axis=-1
                )
            result[:, self.affine_count :] = means.reshape(
                frames, self.offset_count, 7
            ).min(axis=-1)
        return np.where(np.isfinite(mapped).all(axis=-1), result, np.nan)


def _prepare_component(mapping, names):
    lookup = {name: i for i, name in enumerate(names)}
    entries = []
    outputs = []
    if mapping._passthrough:
        for name, index in lookup.items():
            stripped = mapping._strip_prefix(name)
            if stripped in mapping._passthrough_landmark_names:
                outputs.append(stripped)
                entries.append((len(outputs) - 1, [index], None))
    else:
        for name, entry in mapping._entries.items():
            sources = (entry,) if isinstance(entry, str) else tuple(entry)
            weights = list(entry.values()) if isinstance(entry, dict) else None
            if not all(mapping._prefix + source in lookup for source in sources) or (
                weights is not None and sum(weights) <= 0
            ):
                continue
            outputs.append(name)
            entries.append(
                (
                    len(outputs) - 1,
                    [lookup[mapping._prefix + source] for source in sources],
                    weights,
                )
            )
    affine_count = len(outputs)
    raw, mapped = [], []
    combined = dict(lookup)
    for index, name in enumerate(outputs):
        combined[name] = len(names) + index
        if name in lookup:
            raw.append(lookup[name])
            mapped.append(len(names) + index)
    mean_entries, axes, ratios = [], [], []
    for index, (name, definition) in enumerate(mapping._anatomical_offsets.items()):
        outputs.append(name)
        exact = next(axis for axis in definition.axes if axis.kind == "exact")
        approximate = next(
            axis for axis in definition.axes if axis.kind == "approximate"
        )
        groups = (
            definition.origin_keypoints,
            definition.reference_length_from,
            definition.reference_length_to,
            exact.from_keypoints,
            exact.to_keypoints,
            approximate.from_keypoints,
            approximate.to_keypoints,
        )
        for slot, group in enumerate(groups):
            if all(mapping._prefix + source in combined for source in group):
                mean_entries.append(
                    (
                        index * 7 + slot,
                        [combined[mapping._prefix + source] for source in group],
                        None,
                    )
                )
        third = set(definition.offset_ratios) - {exact.name, approximate.name}
        if len(third) > 1:
            raise ValueError(
                "Anatomical offset can name at most one implicit third axis"
            )
        basis = {
            exact.name: 0,
            approximate.name: 1,
            next(iter(third), "_implicit_third"): 2,
        }
        axes.append([basis[key] for key in definition.offset_ratios])
        ratios.append(list(definition.offset_ratios.values()))
    # Padded zero terms do not change finite offsets; every invalid row is masked.
    width = max(map(len, axes), default=1)
    axes = [row + [0] * (width - len(row)) for row in axes]
    ratios = [row + [0.0] * (width - len(row)) for row in ratios]
    return _Component(
        tuple(outputs),
        affine_count,
        _reductions(entries),
        _fixed(raw, int),
        _fixed(mapped, int),
        _reductions(mean_entries),
        len(axes),
        _fixed(axes, int),
        _fixed(ratios, float),
    )


@dataclass(frozen=True)
class PreparedTrackerMapping:
    """Prepare on configuration/axis change, reuse across coordinates and occlusions.

    Owns immutable definition arrays, not mutable per-frame buffers. Returned
    arrays belong to the caller and are never overwritten by subsequent calls.
    Fingerprint is computed once, not on the frame path. Caller must replace the
    plan when configuration changes; ordered tracker names are checked per call.
    """

    tracker_names: tuple[str, ...]
    landmark_names: tuple[str, ...]
    fingerprint: str
    components: tuple[_Component, ...]
    tracker_columns: Mapping[str, int]

    @classmethod
    def prepare(cls, mapping, *, tracker_names: tuple[str, ...]):
        if len(set(tracker_names)) != len(tracker_names):
            raise ValueError("Tracker names must be unique")
        snapshots = mapping.mapping_snapshots()
        components = tuple(
            _prepare_component(snapshot.restore(), tracker_names)
            for snapshot in snapshots
        )
        names = tuple(name for component in components for name in component.names)
        if len(set(names)) != len(names):
            raise ValueError("Composite mapping has overlapping landmark names")
        from dataclasses import asdict

        payload = json.dumps(
            (tracker_names, [asdict(snapshot) for snapshot in snapshots]),
            separators=(",", ":"),
        )
        return cls(
            tracker_names,
            names,
            hashlib.sha256(payload.encode()).hexdigest(),
            components,
            MappingProxyType({name: index for index, name in enumerate(tracker_names)}),
        )

    def apply(
        self, *, tracker_names: tuple[str, ...], positions: np.ndarray
    ) -> MappedLandmarkBatch:
        if tracker_names != self.tracker_names:
            raise ValueError("Tracker axis changed; prepare a new mapping")
        points = np.asarray(positions, dtype=float)
        if points.ndim != 3 or points.shape[1:] != (len(tracker_names), 3):
            raise ValueError("Expected (frames, keypoints, 3) positions")
        points = np.where(np.isfinite(points).all(axis=-1)[..., None], points, np.nan)
        values = np.concatenate(
            [component.apply(points) for component in self.components], axis=1
        )
        return MappedLandmarkBatch(
            self.landmark_names, values, np.isfinite(values).all(axis=-1)
        )

    def apply_frame(
        self, *, tracker_positions: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Existing named-point interface; omitted keys are absent, unknown keys reject."""
        unknown = tracker_positions.keys() - self.tracker_columns.keys()
        if unknown:
            raise ValueError(
                f"Tracker axis changed; prepare a new mapping: {sorted(unknown)}"
            )
        points = np.full((1, len(self.tracker_names), 3), np.nan)
        if tracker_positions:
            values = np.asarray(list(tracker_positions.values()), dtype=float)
            if values.shape != (len(tracker_positions), 3):
                raise ValueError("Each position must have three coordinates")
            points[0, [self.tracker_columns[name] for name in tracker_positions]] = (
                values
            )
        mapped = self.apply(tracker_names=self.tracker_names, positions=points)
        return {
            name: mapped.positions[0, index]
            for index, name in enumerate(mapped.landmark_names)
            if mapped.present[0, index]
        }

    def apply_with_quality(
        self,
        *,
        tracker_names: tuple[str, ...],
        positions: np.ndarray,
        quality: np.ndarray,
    ) -> "MappedLandmarkQualityBatch":
        """Conservative minimum source confidence; absent points carry NaN scores."""
        points = np.asarray(positions, dtype=float)
        scores = np.asarray(quality, dtype=float)
        mapped = self.apply(tracker_names=tracker_names, positions=points)
        if scores.shape != points.shape[:2]:
            raise ValueError("Quality must match frame and keypoint axes")
        present = np.isfinite(points).all(axis=-1)
        if np.any(present & (~np.isfinite(scores) | (scores < 0) | (scores > 1))):
            raise ValueError("Present source quality must be finite and in [0, 1]")
        scores = np.where(present, scores, np.nan)
        parts = []
        column = 0
        for component in self.components:
            parts.append(
                component.quality(
                    scores, mapped.positions[:, column : column + len(component.names)]
                )
            )
            column += len(component.names)
        return MappedLandmarkQualityBatch(
            mapped.landmark_names,
            mapped.positions,
            mapped.present,
            np.concatenate(parts, axis=1),
        )


@dataclass(frozen=True)
class MappedLandmarkQualityBatch:
    landmark_names: tuple[str, ...]
    positions: np.ndarray
    present: np.ndarray
    quality: np.ndarray
