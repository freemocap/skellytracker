import cv2


from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import (
    BrightestPointTracker,
)
from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker

try:
    from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import (
        MediapipeHolisticTracker,
    )
except ModuleNotFoundError:
    print("To use mediapipe_holistic_tracker, install skellytracker[mediapipe]")
try:
    from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker
    from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import (
        YOLOObjectTracker,
    )
    from skellytracker.trackers.segment_anything_tracker.segment_anything_tracker import (
        SAMTracker,
    )
except ModuleNotFoundError:
    print("To use yolo_tracker, install skellytracker[yolo]")


def main(demo_tracker: str = "mediapipe_holistic_tracker"):

    if demo_tracker == "brightest_point_tracker":
        BrightestPointTracker(num_points=2).demo()

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
        ).demo()

    elif demo_tracker == "mediapipe_holistic_tracker":
        MediapipeHolisticTracker(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            smooth_landmarks=True,
        ).demo()

    elif demo_tracker == "yolo_tracker":
        YOLOPoseTracker(model_size="nano").demo()
    elif demo_tracker == "SAM_tracker":
        SAMTracker().demo()
    elif demo_tracker == "yolo_object_tracker":
        YOLOObjectTracker(model_size="medium").demo()


if __name__ == "__main__":
    main(demo_tracker="mediapipe_holistic_tracker")
