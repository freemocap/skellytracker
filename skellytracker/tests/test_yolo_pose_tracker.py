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
    tracked_person = tracked_objects["tracked_person_0"]
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
    assert processed_results.shape == (1, YOLOModelInfo.num_tracked_points, 2)

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
    assert np.allclose(processed_results, expected_results, atol=1e-2)


# @pytest.mark.usefixtures("test_image")
import cv2

bus_image = cv2.imread("/Users/philipqueen/Downloads/bus.jpg")
bus_image = cv2.cvtColor(bus_image, cv2.COLOR_BGR2RGB)


def test_record_multiperson(test_image=bus_image):
    max_det = 2
    tracker = YOLOPoseTracker(model_size="nano", max_det=max_det)
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1

    processed_results = tracker.recorder.process_tracked_objects()
    assert processed_results is not None
    assert processed_results.shape == (
        1,
        (YOLOModelInfo.num_tracked_points * max_det),
        2,
    )

    expected_results = np.array(
        [
            [
                [142.94776916503906, 441.1304626464844],
                [149.22552490234375, 432.66583251953125],
                [132.95570373535156, 432.6253356933594],
                [154.91513061523438, 438.90625],
                [108.84307861328125, 438.0549621582031],
                [169.82933044433594, 493.0794982910156],
                [90.12742614746094, 494.21734619140625],
                [193.97451782226562, 565.8185424804688],
                [113.39808654785156, 563.470458984375],
                [150.80001831054688, 564.606689453125],
                [156.97048950195312, 556.6004028320312],
                [159.6195831298828, 635.1295166015625],
                [97.7007064819336, 636.2892456054688],
                [182.3831024169922, 747.2435913085938],
                [85.82090759277344, 754.9800415039062],
                [187.85714721679688, 855.2320556640625],
                [69.11174774169922, 866.8134765625],
                [142.94776916503906, 441.1304626464844],
                [149.22552490234375, 432.66583251953125],
                [132.95570373535156, 432.6253356933594],
                [154.91513061523438, 438.90625],
                [108.84307861328125, 438.0549621582031],
                [169.82933044433594, 493.0794982910156],
                [90.12742614746094, 494.21734619140625],
                [193.97451782226562, 565.8185424804688],
                [113.39808654785156, 563.470458984375],
                [150.80001831054688, 564.606689453125],
                [156.97048950195312, 556.6004028320312],
                [159.6195831298828, 635.1295166015625],
                [97.7007064819336, 636.2892456054688],
                [182.3831024169922, 747.2435913085938],
                [85.82090759277344, 754.9800415039062],
                [187.85714721679688, 855.2320556640625],
                [69.11174774169922, 866.8134765625],
            ]
        ]
    )
    print(processed_results.tolist())
    print(processed_results.shape)
    assert np.allclose(processed_results, expected_results, atol=1e-2)


class MockKeypoints:
    xy: list = []


class MockResults:
    keypoints: MockKeypoints = MockKeypoints()


def test_unpack_empty_results():
    tracker = YOLOPoseTracker(model_size="nano")
    results = [MockResults()]
    tracker.unpack_results(results=results)

    tracked_person = tracker.tracked_objects["tracked_person_0"]
    assert tracked_person.pixel_x is None
    assert tracked_person.pixel_y is None
    assert tracked_person.extra["landmarks"].shape == (
        1,
        YOLOModelInfo.num_tracked_points,
        2,
    )
    assert np.isnan(tracked_person.extra["landmarks"][0, :, 0]).all()
    assert np.isnan(tracked_person.extra["landmarks"][0, :, 1]).all()
