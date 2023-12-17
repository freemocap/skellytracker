import cv2

from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import BrightestPointTracker
from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker
from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import MediapipeHolisticTracker
from skellytracker.trackers.segment_anything_tracker.segment_anything_tracker import SAMTracker
from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker
from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import YOLOObjectTracker

if __name__ == "__main__":

    demo_tracker = "yolo_object_tracker"


    if demo_tracker == "brightest_point_tracker":
        BrightestPointTracker().demo()

    elif demo_tracker == "charuco_tracker":
        CharucoTracker(squaresX=7,
                       squaresY=5,
                       dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)).demo()

    elif demo_tracker == "mediapipe_holistic_tracker":
        MediapipeHolisticTracker(model_complexity=2,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5,
                                 static_image_mode=False,
                                 smooth_landmarks=True).demo()

    elif demo_tracker == "yolo_tracker":
        YOLOPoseTracker(model_size="high_res").demo()
    elif demo_tracker == "SAM_tracker":
        SAMTracker().demo()
    elif demo_tracker == "yolo_object_tracker":
        YOLOObjectTracker(model_size="medium").demo()
