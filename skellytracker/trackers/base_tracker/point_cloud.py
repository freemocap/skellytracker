"""
PointCloud: a labeled array of 3D points with visibility scores.

Names and coordinates are structurally coupled — they cannot desync.
This is the canonical data primitive for tracked landmarks throughout
the pipeline: detection → triangulation → filtering → visualization.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PointCloud:
    """
    A fixed-size labeled point array.

    Stores N named points, each with xyz coordinates and a visibility score.
    Names are immutable and set at construction — the i-th name always
    corresponds to the i-th row of the coordinate array.

    Points that are not detected have NaN coordinates and zero visibility.
    """

    names: tuple[str, ...]
    xyz: NDArray[np.floating]        # (N, 3) — mutable coordinates
    visibility: NDArray[np.floating] # (N,)   — mutable confidence scores
    _name_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.names)
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz shape {self.xyz.shape} does not match {n} names (expected ({n}, 3))")
        if self.visibility.shape != (n,):
            raise ValueError(f"visibility shape {self.visibility.shape} does not match {n} names (expected ({n},))")
        self._name_to_idx = {name: i for i, name in enumerate(self.names)}

    # =========================================================================
    # Properties — zero-copy views
    # =========================================================================

    @property
    def n_points(self) -> int:
        return len(self.names)

    @property
    def xy(self) -> NDArray[np.floating]:
        """(N, 2) view of just x, y coordinates. Zero-copy."""
        return self.xyz[:, :2]

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where point has non-NaN coordinates."""
        return ~np.isnan(self.xyz).any(axis=1)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    # =========================================================================
    # Lookup
    # =========================================================================

    def index_of(self, name: str) -> int:
        """Get the row index for a point name. Raises KeyError if not found."""
        return self._name_to_idx[name]

    def has_name(self, name: str) -> bool:
        return name in self._name_to_idx

    def xyz_by_name(self, name: str) -> NDArray[np.floating]:
        """Get (3,) coordinate array for a single point by name."""
        return self.xyz[self._name_to_idx[name]]

    def xy_by_name(self, name: str) -> NDArray[np.floating]:
        """Get (2,) coordinate array for a single point by name."""
        return self.xyz[self._name_to_idx[name], :2]

    # =========================================================================
    # Slicing — returns views, not copies
    # =========================================================================

    def slice(self, start: int, stop: int) -> "PointCloud":
        """
        Return a PointCloud backed by views into this cloud's arrays.

        The returned cloud shares memory with this one — mutations
        to one affect the other.
        """
        return PointCloud(
            names=self.names[start:stop],
            xyz=self.xyz[start:stop],
            visibility=self.visibility[start:stop],
        )

    def slice_by_names(self, names: tuple[str, ...]) -> "PointCloud":
        """Return a PointCloud with only the named points (copies data)."""
        indices = np.array([self._name_to_idx[n] for n in names])
        return PointCloud(
            names=names,
            xyz=self.xyz[indices].copy(),
            visibility=self.visibility[indices].copy(),
        )

    # =========================================================================
    # Construction helpers
    # =========================================================================

    @staticmethod
    def empty(names: tuple[str, ...]) -> "PointCloud":
        """Create a PointCloud with all NaN coordinates and zero visibility."""
        n = len(names)
        return PointCloud(
            names=names,
            xyz=np.full((n, 3), np.nan),
            visibility=np.zeros(n),
        )

    @staticmethod
    def concatenate(clouds: list["PointCloud"]) -> "PointCloud":
        """
        Concatenate multiple PointClouds into one.

        Copies data — the result is independent of the inputs.
        """
        if not clouds:
            raise ValueError("Cannot concatenate empty list of PointClouds")

        all_names: list[str] = []
        all_xyz: list[NDArray] = []
        all_vis: list[NDArray] = []

        for cloud in clouds:
            all_names.extend(cloud.names)
            all_xyz.append(cloud.xyz)
            all_vis.append(cloud.visibility)

        return PointCloud(
            names=tuple(all_names),
            xyz=np.concatenate(all_xyz, axis=0).copy(),
            visibility=np.concatenate(all_vis, axis=0).copy(),
        )

    # =========================================================================
    # Conversion — for interfaces that expect dicts or flat arrays
    # =========================================================================

    def to_2d_array(self) -> NDArray[np.floating]:
        """
        Return (N, 2) array of xy coordinates.

        Always returns N rows in name order. NaN for undetected points.
        Copy — safe to mutate without affecting the cloud.
        """
        return self.xy.copy()

    def to_named_dict(self, dimensions: int = 2) -> dict[str, NDArray[np.floating]]:
        """
        Return dict of {name: coordinate_array} for ALL points.

        Always includes all N names, preserving structural consistency
        with to_2d_array(). NaN-valued points are included.
        """
        d = dimensions
        return {name: self.xyz[i, :d].copy() for i, name in enumerate(self.names)}

    def to_valid_dict(self, dimensions: int = 2) -> dict[str, NDArray[np.floating]]:
        """
        Return dict of {name: coordinate_array} for only valid (non-NaN) points.

        Use this for display/overlay purposes where you want to skip missing points.
        Do NOT use this for triangulation — use to_named_dict() instead.
        """
        mask = self.valid_mask
        d = dimensions
        return {
            name: self.xyz[i, :d].copy()
            for i, name in enumerate(self.names)
            if mask[i]
        }

    def to_scaled_tuples(self, dimensions: int, scale_by: float = 1.0) -> dict[str, tuple]:
        """
        Return dict of {name: (x, y[, z])} for only valid points, scaled.

        Matches the legacy MediapipeObservation.all_points() interface.
        """
        mask = self.valid_mask
        d = dimensions
        scaled = self.xyz * scale_by
        return {
            name: tuple(scaled[i, :d])
            for i, name in enumerate(self.names)
            if mask[i]
        }

    # =========================================================================
    # Filtering
    # =========================================================================

    def filtered_by_confidence(self, threshold: float, fill_with_nans: bool = True) -> "PointCloud":
        """
        Return a new PointCloud with low-confidence points masked.

        If fill_with_nans is True, returns same-sized cloud with NaN for
        low-confidence points. If False, returns a smaller cloud with
        only high-confidence points (names change!).
        """
        if fill_with_nans:
            xyz = self.xyz.copy()
            vis = self.visibility.copy()
            mask = vis < threshold
            xyz[mask] = np.nan
            return PointCloud(names=self.names, xyz=xyz, visibility=vis)
        else:
            mask = self.visibility >= threshold
            indices = np.where(mask)[0]
            return PointCloud(
                names=tuple(self.names[i] for i in indices),
                xyz=self.xyz[indices].copy(),
                visibility=self.visibility[indices].copy(),
            )

    # =========================================================================
    # Repr
    # =========================================================================

    def __len__(self) -> int:
        return self.n_points
