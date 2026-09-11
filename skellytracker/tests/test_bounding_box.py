from __future__ import annotations

from skellytracker.core.data_primitives.bounding_box import BoundingBox


class TestBoundingBoxClipped:
    def test_partially_out_of_bounds_box_is_clamped(self):
        bbox = BoundingBox(x1=-10.0, y1=-10.0, x2=50.0, y2=50.0)
        clipped = bbox.clipped(image_height=40, image_width=40)
        assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0.0, 0.0, 40.0, 40.0)

    def test_box_entirely_past_bottom_right_edge_collapses_without_raising(self):
        # Reproduces a real crash: a keypoint-derived box centered near the
        # frame edge can lie entirely outside [0, w] x [0, h] on one axis.
        # x1/y1 must not stay unclamped-from-above while x2/y2 clamp down,
        # or the x1<=x2/y1<=y2 invariant breaks and BoundingBox() raises.
        bbox = BoundingBox(x1=948.0, y1=755.0, x2=1068.0, y2=875.0)
        clipped = bbox.clipped(image_height=720, image_width=1280)
        assert clipped.x1 <= clipped.x2
        assert clipped.y1 <= clipped.y2
        assert (clipped.y1, clipped.y2) == (720.0, 720.0)

    def test_box_entirely_past_top_left_edge_collapses_without_raising(self):
        bbox = BoundingBox(x1=-100.0, y1=-100.0, x2=-20.0, y2=-20.0)
        clipped = bbox.clipped(image_height=480, image_width=640)
        assert (clipped.x1, clipped.x2) == (0.0, 0.0)
        assert (clipped.y1, clipped.y2) == (0.0, 0.0)

    def test_fully_in_bounds_box_is_unchanged(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        clipped = bbox.clipped(image_height=480, image_width=640)
        assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (10.0, 20.0, 100.0, 200.0)

    def test_clipped_never_raises_for_any_out_of_bounds_direction(self):
        image_height, image_width = 100, 100
        for x1, y1, x2, y2 in [
            (150.0, 150.0, 200.0, 200.0),  # past bottom-right
            (-200.0, -200.0, -150.0, -150.0),  # past top-left
            (150.0, -200.0, 200.0, -150.0),  # past right, above top
            (-200.0, 150.0, -150.0, 200.0),  # past left, below bottom
        ]:
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            clipped = bbox.clipped(image_height=image_height, image_width=image_width)
            assert clipped.x1 <= clipped.x2
            assert clipped.y1 <= clipped.y2
