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
    assert math.isclose(tracked_person.pixel_x,  337.6132, rel_tol=1e-2)
    assert math.isclose(tracked_person.pixel_y, 377.9709, rel_tol=1e-2)

    landmarks = tracked_person.extra["landmarks"]
    assert landmarks is not None
    assert landmarks.shape == (1, YOLOModelInfo.num_tracked_points, 3)

    expected_results = np.array(
        [
            [
                [393.5962219238281, 142.34054565429688],
                [417.33905029296875, 133.9437255859375],
                [389.22515869140625, 125.23101806640625],
                [448.60614013671875, 157.5833740234375],
                [382.77239990234375, 136.32574462890625],
                [452.36279296875, 263.24786376953125],
                [313.58905029296875, 225.33258056640625],
                [454.72259521484375, 390.9169921875],
                [228.13490295410156, 311.61346435546875],
                [469.9539794921875, 495.96563720703125],
                [145.72940063476562, 404.16790771484375],
                [354.826904296875, 470.28216552734375],
                [269.1026306152344, 448.4520263671875],
                [313.2445068359375, 632.5315551757812],
                [223.3149871826172, 607.5588989257812],
                [263.5947265625, 755.4601440429688],
                [219.31011962890625, 724.552978515625],
            ]
        ]
    )   
    assert np.allclose(landmarks[:,:,:2], expected_results)


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
                [393.5962219238281, 142.34054565429688, np.nan],
                [417.33905029296875, 133.9437255859375, np.nan],
                [389.22515869140625, 125.23101806640625, np.nan],
                [448.60614013671875, 157.5833740234375, np.nan],
                [382.77239990234375, 136.32574462890625, np.nan],
                [452.36279296875, 263.24786376953125, np.nan],
                [313.58905029296875, 225.33258056640625, np.nan],
                [454.72259521484375, 390.9169921875, np.nan],
                [228.13490295410156, 311.61346435546875, np.nan],
                [469.9539794921875, 495.96563720703125, np.nan],
                [145.72940063476562, 404.16790771484375, np.nan],
                [354.826904296875, 470.28216552734375, np.nan],
                [269.1026306152344, 448.4520263671875, np.nan],
                [313.2445068359375, 632.5315551757812, np.nan],
                [223.3149871826172, 607.5588989257812, np.nan],
                [263.5947265625, 755.4601440429688, np.nan],
                [219.31011962890625, 724.552978515625, np.nan],
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :, :2], expected_results[:, :, :2], atol=1e-2
    )
    assert np.isnan(processed_results[:, :, 2]).all()


class MockKeypoints:
    data = None  
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
