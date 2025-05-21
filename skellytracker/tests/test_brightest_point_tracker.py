from typing import List

import cv2
import numpy as np
import pytest

from skellytracker.trackers.bright_point_tracker.__brightest_point_tracker import (
    BrightestPointTracker,
    BrightestPointTrackerConfig,
)
from skellytracker.trackers.bright_point_tracker.brightest_point_detector import BrightestPointDetectorConfig
from skellytracker.trackers.bright_point_tracker.brightest_point_observation import BrightestPointObservation


@pytest.fixture()
def sample_image():
    """
    Create a sample image with bright spots for testing.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (30, 20), 10, (255, 255, 255), -1)  # Bright spot
    cv2.circle(image, (70, 80), 15, (255, 255, 255), -1)  # Brighter spot
    return image

def test_image_with_one_brightest_point(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=1, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation: BrightestPointObservation = tracker.process_image(frame_number=0, image=sample_image)
    bright_patches = observation.bright_patches

    assert len(bright_patches) == 1
    assert bright_patches[0].centroid_x == 70
    assert bright_patches[0].centroid_y == 80

def test_image_with_one_brightest_point_array(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=1, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation = tracker.process_image(frame_number=0, image=sample_image)
    observation_array = observation.to_array()

    assert observation_array.shape == (1, 2)
    assert observation_array[0, 0] == 70
    assert observation_array[0, 1] == 80

def test_image_with_two_brightest_points(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=2, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation = tracker.process_image(frame_number=0, image=sample_image)
    observation_array = observation.to_array()

    assert observation_array.shape == (2, 2)
    assert observation_array[0, 0] == 70
    assert observation_array[0, 1] == 80
    assert observation_array[1, 0] == 30
    assert observation_array[1, 1] == 20


def test_image_with_two_brightest_points_array(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=2, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation = tracker.process_image(frame_number=0, image=sample_image)
    observation_array = observation.to_array()

    assert observation_array.shape == (2, 2)
    assert observation_array[0, 0] == 70
    assert observation_array[0, 1] == 80
    assert observation_array[1, 0] == 30
    assert observation_array[1, 1] == 20


def test_process_image_with_five_brightest_points(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=5, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation = tracker.process_image(frame_number=0, image=sample_image)
    observation_array = observation.to_array()

    assert observation_array.shape == (5, 2)
    assert observation_array[0, 0] == 70
    assert observation_array[0, 1] == 80
    assert observation_array[1, 0] == 30
    assert observation_array[1, 1] == 20 
    assert np.isnan(observation_array[2, 0])
    assert np.isnan(observation_array[2, 1])
    assert np.isnan(observation_array[3, 0])
    assert np.isnan(observation_array[3, 1])
    assert np.isnan(observation_array[4, 0])
    assert np.isnan(observation_array[4, 1])


def test_annotate_image(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=5, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    observation: BrightestPointObservation = tracker.process_image(frame_number=0, image=sample_image)
    annotated_image = tracker.annotate_image(image=sample_image, latest_observation=observation)

    assert annotated_image is not None
    assert annotated_image.shape == sample_image.shape

    # Check if the bright spots have markers
    bright_point_0 = observation.bright_patches[0]
    assert bright_point_0 is not None
    bright_point_1 = observation.bright_patches[1]
    assert bright_point_1 is not None

    assert bright_point_0.centroid_x is not None
    assert bright_point_0.centroid_y is not None
    assert bright_point_1.centroid_x is not None
    assert bright_point_1.centroid_y is not None

    assert annotated_image[
        int(bright_point_0.centroid_y), int(bright_point_0.centroid_y)
    ].tolist() == [0, 0, 255]
    assert annotated_image[
        int(bright_point_1.centroid_y), int(bright_point_1.centroid_x)
    ].tolist() == [0, 0, 255]


def test_record(sample_image):
    config = BrightestPointTrackerConfig(
        detector_config=BrightestPointDetectorConfig(num_tracked_points=5, luminance_threshold=200)
    )
    tracker = BrightestPointTracker.create(config=config)
    tracker.process_image(frame_number=0, image=sample_image)
    recorded_observations: List[BrightestPointObservation] = tracker.recorder.observations

    assert len(recorded_observations) == 1

    assert len(recorded_observations[0].bright_patches) == 5
