import cv2
import pytest
import numpy as np


from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import BrightestPointTracker


@pytest.fixture
def sample_image():
    """
    Create a sample image with bright spots for testing.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (30, 30), 10, (255, 255, 255), -1)  # Bright spot
    cv2.circle(image, (70, 70), 15, (255, 255, 255), -1)  # Brighter spot
    return image

def test_process_image(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)
    
    assert len(tracked_objects) == 2
    assert tracked_objects["brightest_point_0"].pixel_x == 70
    assert tracked_objects["brightest_point_0"].pixel_y == 70
    assert tracked_objects["brightest_point_1"].pixel_x == 30
    assert tracked_objects["brightest_point_1"].pixel_y == 30

def test_annotate_image(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)

    assert tracker.annotated_image is not None

    # Check if the bright spots have markers
    bright_point_0 = tracked_objects["brightest_point_0"]
    bright_point_1 = tracked_objects["brightest_point_1"]

    assert tracker.annotated_image[bright_point_0.pixel_y, bright_point_0.pixel_x].tolist() == [0, 0, 255]
    assert tracker.annotated_image[bright_point_1.pixel_y, bright_point_1.pixel_x].tolist() == [0, 0, 255]

def test_record(sample_image):
    tracker = BrightestPointTracker(num_points=2, luminance_threshold=200)
    tracked_objects = tracker.process_image(sample_image)
    tracker.recorder.record(tracked_objects)

    assert len(tracker.recorder.recorded_objects) == 2

    print(tracked_objects)
    print(tracker.recorder.recorded_objects)