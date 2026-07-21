from __future__ import annotations

import numpy as np
import pytest

from skellytracker.core.data_primitives.keypoints import Keypoints


def _make(names: tuple[str, ...], *, nan: bool = False) -> Keypoints:
    n = len(names)
    xyz = np.full((n, 3), np.nan) if nan else np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    vis = np.zeros(n) if nan else np.linspace(0.1, 1.0, n)
    return Keypoints(names=names, xyz=xyz, visibility=vis)


NAMES = ("a", "b", "c", "d")


class TestConstruction:
    def test_basic(self):
        kpts = _make(NAMES)
        assert kpts.n_points == 4
        assert kpts.xyz.shape == (4, 3)
        assert kpts.visibility.shape == (4,)

    def test_shape_mismatch_xyz_raises(self):
        with pytest.raises(ValueError, match="xyz shape"):
            Keypoints(
                names=("a", "b"),
                xyz=np.zeros((3, 3)),
                visibility=np.zeros(2),
            )

    def test_shape_mismatch_visibility_raises(self):
        with pytest.raises(ValueError, match="visibility shape"):
            Keypoints(
                names=("a", "b"),
                xyz=np.zeros((2, 3)),
                visibility=np.zeros(3),
            )

    def test_empty_factory(self):
        kpts = Keypoints.empty(("x", "y"))
        assert np.all(np.isnan(kpts.xyz))
        assert np.all(kpts.visibility == 0.0)
        assert kpts.n_points == 2


class TestProperties:
    def test_n_valid_all_detected(self):
        kpts = _make(NAMES)
        assert kpts.n_valid == len(NAMES)

    def test_n_valid_all_nan(self):
        kpts = _make(NAMES, nan=True)
        assert kpts.n_valid == 0

    def test_n_valid_partial(self):
        kpts = _make(NAMES)
        kpts.xyz[1] = np.nan
        kpts.xyz[3] = np.nan
        assert kpts.n_valid == 2

    def test_valid_mask(self):
        kpts = _make(NAMES)
        kpts.xyz[2] = np.nan
        mask = kpts.valid_mask
        assert mask.tolist() == [True, True, False, True]

    def test_xy_view(self):
        kpts = _make(NAMES)
        assert kpts.xy.shape == (4, 2)
        np.testing.assert_array_equal(kpts.xy, kpts.xyz[:, :2])

    def test_len(self):
        assert len(_make(NAMES)) == 4


class TestLookup:
    def test_index_of(self):
        kpts = _make(NAMES)
        assert kpts.index_of("a") == 0
        assert kpts.index_of("d") == 3

    def test_has_name_true(self):
        assert _make(NAMES).has_name("b")

    def test_has_name_false(self):
        assert not _make(NAMES).has_name("z")

    def test_xyz_by_name(self):
        kpts = _make(NAMES)
        np.testing.assert_array_equal(kpts.xyz_by_name("a"), kpts.xyz[0])

    def test_xy_by_name(self):
        kpts = _make(NAMES)
        np.testing.assert_array_equal(kpts.xy_by_name("c"), kpts.xyz[2, :2])

    def test_index_of_missing_raises(self):
        with pytest.raises(KeyError):
            _make(NAMES).index_of("z")


class TestSlicing:
    def test_slice_range(self):
        kpts = _make(NAMES)
        sliced = kpts.slice(1, 3)
        assert sliced.names == ("b", "c")
        assert sliced.xyz.shape == (2, 3)

    def test_slice_shares_memory(self):
        kpts = _make(NAMES)
        sliced = kpts.slice(0, 2)
        assert np.shares_memory(sliced.xyz, kpts.xyz)

    def test_slice_by_names(self):
        kpts = _make(NAMES)
        sub = kpts.slice_by_names(("d", "a"))
        assert sub.names == ("d", "a")
        np.testing.assert_array_equal(sub.xyz[0], kpts.xyz[3])
        np.testing.assert_array_equal(sub.xyz[1], kpts.xyz[0])

    def test_slice_by_names_copies(self):
        kpts = _make(NAMES)
        sub = kpts.slice_by_names(("a", "b"))
        assert sub.xyz.base is None or sub.xyz.base is not kpts.xyz


class TestConcatenate:
    def test_two_clouds(self):
        a = _make(("p", "q"))
        b = _make(("r", "s"))
        result = Keypoints.concatenate([a, b])
        assert result.names == ("p", "q", "r", "s")
        assert result.xyz.shape == (4, 3)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            Keypoints.concatenate([])

    def test_single_element(self):
        kpts = _make(("x",))
        result = Keypoints.concatenate([kpts])
        assert result.names == ("x",)


class TestFilteredByConfidence:
    def test_fill_with_nans(self):
        kpts = _make(NAMES)
        kpts.visibility[:] = [0.1, 0.5, 0.9, 0.2]
        filtered = kpts.filtered_by_confidence(0.4)
        assert filtered.names == NAMES
        assert np.isnan(filtered.xyz[0]).all()
        assert np.isnan(filtered.xyz[3]).all()
        assert not np.isnan(filtered.xyz[1]).any()
        assert not np.isnan(filtered.xyz[2]).any()

    def test_no_fill_reduces_size(self):
        kpts = _make(NAMES)
        kpts.visibility[:] = [0.1, 0.5, 0.9, 0.2]
        filtered = kpts.filtered_by_confidence(0.4, fill_with_nans=False)
        assert filtered.names == ("b", "c")
        assert filtered.xyz.shape == (2, 3)

    def test_does_not_mutate_original(self):
        kpts = _make(NAMES)
        original_xyz = kpts.xyz.copy()
        kpts.filtered_by_confidence(0.5)
        np.testing.assert_array_equal(kpts.xyz, original_xyz)


class TestTranslated:
    def test_offsets_xy(self):
        kpts = _make(("p",))
        kpts.xyz[0] = [10.0, 20.0, 5.0]
        moved = kpts.translated(3.0, -7.0)
        assert moved.xyz[0, 0] == pytest.approx(13.0)
        assert moved.xyz[0, 1] == pytest.approx(13.0)
        assert moved.xyz[0, 2] == pytest.approx(5.0)

    def test_nan_stays_nan(self):
        kpts = Keypoints.empty(("a",))
        moved = kpts.translated(100.0, 100.0)
        assert np.isnan(moved.xyz[0]).all()

    def test_does_not_mutate_original(self):
        kpts = _make(("a",))
        original = kpts.xyz.copy()
        kpts.translated(1.0, 2.0)
        np.testing.assert_array_equal(kpts.xyz, original)


class TestToNamedDict:
    def test_2d(self):
        kpts = _make(("a", "b"))
        d = kpts.to_named_dict(dimensions=2)
        assert set(d.keys()) == {"a", "b"}
        assert d["a"].shape == (2,)

    def test_3d(self):
        kpts = _make(("a",))
        d = kpts.to_named_dict(dimensions=3)
        assert d["a"].shape == (3,)
