import cv2
from pathlib import Path

from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import (
    BrightestPointTracker,
)
from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker

try:
    from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import (
        MediapipeHolisticTracker,
    )
except ModuleNotFoundError:
    print("\n\nTo use mediapipe_holistic_tracker, install skellytracker[mediapipe]\n\n")
try:
    from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker
except ModuleNotFoundError:
    print("\n\nTo use yolo_tracker, install skellytracker[yolo]\n\n")


if __name__ == "__main__":
    demo_tracker = "brightest_point_tracker"
    image_path = Path("/Path/To/Your/Image.jpg")

    if demo_tracker == "brightest_point_tracker":
        BrightestPointTracker(num_points=2).image_demo(image_path=image_path)

    elif demo_tracker == "charuco_tracker":
        charuco_squares_x = 7
        charuco_squares_y = 5
        number_of_charuco_markers = (charuco_squares_x - 1) * (charuco_squares_y - 1)
        charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

        CharucoTracker(
            tracked_object_names=charuco_ids,
            squares_x=charuco_squares_x,
            squares_y=charuco_squares_y,
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        ).image_demo(image_path=image_path)

    elif demo_tracker == "mediapipe_holistic_tracker":
        MediapipeHolisticTracker(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            smooth_landmarks=True,
        ).image_demo(image_path=image_path)

    elif demo_tracker == "yolo_tracker":
        YOLOPoseTracker(model_size="high_res").image_demo(image_path=image_path)
