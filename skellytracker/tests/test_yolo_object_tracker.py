import pytest
import numpy as np


from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import (
    YOLOObjectTracker,
)


@pytest.mark.usefixtures("test_image")
def test_process_image_person_only(test_image):
    tracker = YOLOObjectTracker(model_size="nano", person_only=True)
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == 1
    assert tracked_objects["object"] is not None
    assert tracked_objects["object"].extra["boxes_xyxy"] is not None
    assert np.allclose(
        tracked_objects["object"].extra["boxes_xyxy"],
        [90.676, 96.981, 493.54, 812.03],
        atol=1e-2,
    )
    assert tracked_objects["object"].extra["original_image_shape"] == (1280, 720)


@pytest.mark.usefixtures("test_image")
def test_annotate_image(test_image):
    tracker = YOLOObjectTracker()
    tracker.process_image(test_image)

    assert tracker.annotated_image is not None


@pytest.mark.usefixtures("test_image")
def test_record(test_image):
    tracker = YOLOObjectTracker(model_size="nano", person_only=True)
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1

    processed_results = tracker.recorder.process_tracked_objects()
    assert processed_results is not None
    assert processed_results.shape == (1, 4)

    assert np.allclose(processed_results, [90.676, 96.981, 493.54, 812.03], atol=1e-2)
