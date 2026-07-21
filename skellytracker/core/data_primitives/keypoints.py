from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Keypoints:
    """Named array of 3D points with visibility scores.

    Structural invariant: names, xyz rows, and visibility entries are always
    the same length. Undetected points have NaN coordinates and 0.0 visibility.
    """

    names: tuple[str, ...]
    xyz: NDArray[np.float64]         # (N, 3) — x,y in pixels; z filled by triangulation
    visibility: NDArray[np.float64]  # (N,)   — confidence scores 0.0–1.0
    _name_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.names)
        if self.xyz.shape != (n, 3):
            raise ValueError(
                f"xyz shape {self.xyz.shape} does not match {n} names (expected ({n}, 3))"
            )
        if self.visibility.shape != (n,):
            raise ValueError(
                f"visibility shape {self.visibility.shape} does not match {n} names "
                f"(expected ({n},))"
            )
        object.__setattr__(self, "_name_to_idx", {name: i for i, name in enumerate(self.names)})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.names)

    @property
    def xy(self) -> NDArray[np.float64]:
        """(N, 2) zero-copy view of x, y coordinates."""
        return self.xyz[:, :2]

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        return ~np.isnan(self.xyz).any(axis=1)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def index_of(self, name: str) -> int:
        return self._name_to_idx[name]

    def has_name(self, name: str) -> bool:
        return name in self._name_to_idx

    def xyz_by_name(self, name: str) -> NDArray[np.float64]:
        return self.xyz[self._name_to_idx[name]]

    def xy_by_name(self, name: str) -> NDArray[np.float64]:
        return self.xyz[self._name_to_idx[name], :2]

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    def slice(self, start: int, stop: int) -> Keypoints:
        """Range slice; shares memory with this instance."""
        return Keypoints(
            names=self.names[start:stop],
            xyz=self.xyz[start:stop],
            visibility=self.visibility[start:stop],
        )

    def slice_by_names(self, names: tuple[str, ...]) -> Keypoints:
        """Name-based subset; copies data."""
        indices = np.array([self._name_to_idx[n] for n in names])
        return Keypoints(
            names=names,
            xyz=self.xyz[indices].copy(),
            visibility=self.visibility[indices].copy(),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def empty(names: tuple[str, ...]) -> Keypoints:
        n = len(names)
        return Keypoints(
            names=names,
            xyz=np.full((n, 3), np.nan, dtype=np.float64),
            visibility=np.zeros(n, dtype=np.float64),
        )

    @staticmethod
    def concatenate(clouds: list[Keypoints]) -> Keypoints:
        if not clouds:
            raise ValueError("Cannot concatenate an empty list of Keypoints")
        all_names: list[str] = []
        all_xyz: list[NDArray[np.float64]] = []
        all_vis: list[NDArray[np.float64]] = []
        for c in clouds:
            all_names.extend(c.names)
            all_xyz.append(c.xyz)
            all_vis.append(c.visibility)
        return Keypoints(
            names=tuple(all_names),
            xyz=np.concatenate(all_xyz, axis=0),
            visibility=np.concatenate(all_vis, axis=0),
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filtered_by_confidence(
        self, threshold: float, *, fill_with_nans: bool = True
    ) -> Keypoints:
        """Return a new Keypoints with low-confidence points masked.

        fill_with_nans=True: same-sized result, low-confidence coords set to NaN.
        fill_with_nans=False: smaller result with only high-confidence points.
        """
        if fill_with_nans:
            xyz = self.xyz.copy()
            xyz[self.visibility < threshold] = np.nan
            return Keypoints(names=self.names, xyz=xyz, visibility=self.visibility.copy())
        mask = self.visibility >= threshold
        indices = np.where(mask)[0]
        return Keypoints(
            names=tuple(self.names[i] for i in indices),
            xyz=self.xyz[indices].copy(),
            visibility=self.visibility[indices].copy(),
        )

    def translated(self, dx: float, dy: float) -> Keypoints:
        """Return a new Keypoints with (dx, dy) added to all x, y coordinates.

        NaN coordinates remain NaN (nan + offset = nan).
        """
        xyz = self.xyz.copy()
        xyz[:, 0] += dx
        xyz[:, 1] += dy
        return Keypoints(names=self.names, xyz=xyz, visibility=self.visibility.copy())

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_named_dict(self, dimensions: int = 2) -> dict[str, NDArray[np.float64]]:
        """All points keyed by name. NaN entries included."""
        return {name: self.xyz[i, :dimensions].copy() for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return self.n_points
