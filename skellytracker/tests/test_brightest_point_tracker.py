import cv2
import pytest
import numpy as np


from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import (
    BrightestPointTracker,
)


@pytest.fixture()
def sample_image():
    """
    Create a sample image with bright spots for testing.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (30, 30), 10, (255, 255, 255), -1)  # Bright spot
    cv2.circle(image, (70, 70), 15, (255, 255, 255), -1)  # Brighter spot
    return image


def test_process_image_with_one_brightest_point(sample_image):
    tracker = BrightestPointTracker(num_points=1, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)

    assert len(tracked_objects) == 1
    assert tracked_objects["brightest_point_0"].pixel_x == 70
    assert tracked_objects["brightest_point_0"].pixel_y == 70


def test_process_image_with_two_brightest_points(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)

    assert len(tracked_objects) == 2
    assert tracked_objects["brightest_point_0"].pixel_x == 70
    assert tracked_objects["brightest_point_0"].pixel_y == 70
    assert tracked_objects["brightest_point_1"].pixel_x == 30
    assert tracked_objects["brightest_point_1"].pixel_y == 30


def test_process_image_with_five_brightest_points(sample_image):
    tracker = BrightestPointTracker(num_points=5, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)

    assert len(tracked_objects) == 5
    assert tracked_objects["brightest_point_0"].pixel_x == 70
    assert tracked_objects["brightest_point_0"].pixel_y == 70
    assert tracked_objects["brightest_point_1"].pixel_x == 30
    assert tracked_objects["brightest_point_1"].pixel_y == 30
    assert tracked_objects["brightest_point_2"].pixel_x is None
    assert tracked_objects["brightest_point_2"].pixel_y is None
    assert tracked_objects["brightest_point_3"].pixel_x is None
    assert tracked_objects["brightest_point_3"].pixel_y is None
    assert tracked_objects["brightest_point_4"].pixel_x is None
    assert tracked_objects["brightest_point_4"].pixel_y is None


def test_annotate_image(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)

    assert tracker.annotated_image is not None

    # Check if the bright spots have markers
    bright_point_0 = tracked_objects["brightest_point_0"]
    bright_point_1 = tracked_objects["brightest_point_1"]

    assert bright_point_0.pixel_x is not None
    assert bright_point_0.pixel_y is not None
    assert bright_point_1.pixel_x is not None
    assert bright_point_1.pixel_y is not None

    assert tracker.annotated_image[
        int(bright_point_0.pixel_y), int(bright_point_0.pixel_x)
    ].tolist() == [0, 0, 255]
    assert tracker.annotated_image[
        int(bright_point_1.pixel_y), int(bright_point_1.pixel_x)
    ].tolist() == [0, 0, 255]


def test_record(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)
    tracker.recorder.record(tracked_objects=tracked_objects)

    assert len(tracker.recorder.recorded_objects) == 1

    assert len(tracker.recorder.recorded_objects[0]) == 2
