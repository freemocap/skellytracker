import math
import pytest
import numpy as np


from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo
from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker


@pytest.mark.usefixtures("test_image")
def test_process_image(test_image):
    tracker = YOLOPoseTracker(model_size="nano")
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == 1
    tracked_person = tracked_objects["tracked_person"]
    assert tracked_person.pixel_x is not None
    assert tracked_person.pixel_y is not None
    assert math.isclose(tracked_person.pixel_x, 266.48523, rel_tol=1e-2)
    assert math.isclose(tracked_person.pixel_y, 273.92798, rel_tol=1e-2)

    landmarks = tracked_person.extra["landmarks"]
    assert landmarks is not None
    assert landmarks.shape == (1, YOLOModelInfo.num_tracked_points, 2)

    expected_results = np.array(
        [
            [
                [392.56927490234375, 140.40118408203125],
                [414.94940185546875, 132.90655517578125],
                [386.4353332519531, 125.51483154296875],
                [446.2061767578125, 157.98883056640625],
                [373.6619873046875, 138.93646240234375],
                [453.78662109375, 265.081787109375],
                [317.9375305175781, 231.9653778076172],
                [465.893310546875, 396.12274169921875],
                [220.12176513671875, 325.96636962890625],
                [465.23358154296875, 499.0487365722656],
                [142.17066955566406, 407.7397155761719],
                [352.325439453125, 468.44671630859375],
                [268.8867492675781, 448.7227783203125],
                [310.899658203125, 630.5478515625],
                [227.7810821533203, 617.9011840820312],
                [269.08587646484375, 733.4285888671875],
                [213.1557159423828, 741.54541015625],
            ]
        ]
    )

    assert np.allclose(landmarks, expected_results)


@pytest.mark.usefixtures("test_image")
def test_annotate_image(test_image):
    tracker = YOLOPoseTracker(model_size="nano")
    tracker.process_image(test_image)

    assert tracker.annotated_image is not None


@pytest.mark.usefixtures("test_image")
def test_record(test_image):
    tracker = YOLOPoseTracker(model_size="nano")
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1

    processed_results = tracker.recorder.process_tracked_objects()
    assert processed_results is not None
    assert processed_results.shape == (1, YOLOModelInfo.num_tracked_points, 3)

    expected_results = np.array(
        [
            [
                [392.56927490234375, 140.40118408203125, np.nan],
                [414.94940185546875, 132.90655517578125, np.nan],
                [386.4353332519531, 125.51483154296875, np.nan],
                [446.2061767578125, 157.98883056640625, np.nan],
                [373.6619873046875, 138.93646240234375, np.nan],
                [453.78662109375, 265.081787109375, np.nan],
                [317.9375305175781, 231.9653778076172, np.nan],
                [465.893310546875, 396.12274169921875, np.nan],
                [220.12176513671875, 325.96636962890625, np.nan],
                [465.23358154296875, 499.0487365722656, np.nan],
                [142.17066955566406, 407.7397155761719, np.nan],
                [352.325439453125, 468.44671630859375, np.nan],
                [268.8867492675781, 448.7227783203125, np.nan],
                [310.899658203125, 630.5478515625, np.nan],
                [227.7810821533203, 617.9011840820312, np.nan],
                [269.08587646484375, 733.4285888671875, np.nan],
                [213.1557159423828, 741.54541015625, np.nan],
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :, :2], expected_results[:, :, :2], atol=1e-2
    )
    assert np.isnan(processed_results[:, :, 2]).all()


class MockKeypoints:
    xy: list = []


class MockResults:
    keypoints: MockKeypoints = MockKeypoints()


def test_unpack_empty_results():
    tracker = YOLOPoseTracker(model_size="nano")
    results = [MockResults()]
    tracker.unpack_results(results=results)

    tracked_person = tracker.tracked_objects["tracked_person"]
    assert tracked_person.pixel_x is None
    assert tracked_person.pixel_y is None
    assert tracked_person.extra["landmarks"].shape == (
        1,
        YOLOModelInfo.num_tracked_points,
        2,
    )
    assert np.isnan(tracked_person.extra["landmarks"][0, :, 0]).all()
    assert np.isnan(tracked_person.extra["landmarks"][0, :, 1]).all()
