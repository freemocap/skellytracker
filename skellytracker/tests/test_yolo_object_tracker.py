import pytest
import numpy as np


from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import (
    YOLOObjectTracker,
)
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_info import (
    yolo_object_model_dictionary,
)


@pytest.mark.usefixtures("test_image")
def test_process_image_person_only(test_image):
    tracker = YOLOObjectTracker(model_size="nano", person_only=True)
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == 1
    assert tracked_objects["object"] is not None
    assert tracked_objects["object"].extra["boxes_xyxy"] is not None
    assert np.allclose(tracked_objects["object"].extra["boxes_xyxy"], [89.506,95.665,492.3,816.64], atol=1e-2)
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
    
    assert np.allclose(
        processed_results, [89.506,95.665,492.3,816.64], atol=1e-2
    )

def test_yolo_object_model_dictionary():
    for key in yolo_object_model_dictionary.keys():
        try:
            YOLOObjectTracker(model_size=key)
        except KeyError:
            print(f"Invalid model size: {key}")
        except FileNotFoundError:
            print(f"Invalid model size (.pt file not found): {key}")

    # try invalid model size
    with pytest.raises(KeyError):
        YOLOObjectTracker(model_size="invalid")
