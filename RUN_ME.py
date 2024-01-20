from pathlib import Path
import cv2

from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import (
    BrightestPointTracker,
)
from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker
from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import (
    MediapipeHolisticTracker,
)
from skellytracker.trackers.segment_anything_tracker.segment_anything_tracker import (
    SAMTracker,
)
from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker
from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import (
    YOLOObjectTracker,
)


class DemoTypes:
    WEBCAM = "webcam"
    IMAGE = "image"


if __name__ == "__main__":
    demo_tracker = "yolo_object_tracker"
    demo_type = DemoTypes.WEBCAM
    image_path = Path("/Path/To/Your/Image.jpg")

    if demo_tracker == "brightest_point_tracker":
        tracker_object = BrightestPointTracker()

    elif demo_tracker == "charuco_tracker":
        tracker_object = CharucoTracker(
            squaresX=7,
            squaresY=5,
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        )

    elif demo_tracker == "mediapipe_holistic_tracker":
        tracker_object = MediapipeHolisticTracker(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            smooth_landmarks=True,
        )

    elif demo_tracker == "yolo_tracker":
        tracker_object = YOLOPoseTracker(model_size="high_res")

    elif demo_tracker == "SAM_tracker":
        tracker_object = SAMTracker().demo()

    elif demo_tracker == "yolo_object_tracker":
        tracker_object = YOLOObjectTracker(model_size="medium").demo()

    else:
        raise ValueError("Invalid demo tracker selection")

    if demo_type == DemoTypes.WEBCAM:
        tracker_object.demo()
    elif demo_type == DemoTypes.IMAGE:
        tracker_object.image_demo(image_path=image_path)
    else:
        tracker_object.demo()
