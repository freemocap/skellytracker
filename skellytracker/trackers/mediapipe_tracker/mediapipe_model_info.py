from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams

mediapipe_body_landmark_names = [
    landmark.name.lower() for landmark in mp_holistic.PoseLandmark
]
mediapipe_hand_landmark_names = [
    landmark.name.lower() for landmark in mp_holistic.HandLandmark
]


class MediapipeModelInfo:
    num_tracked_points_body = len(mediapipe_body_landmark_names)
    num_tracked_points_face = FACEMESH_NUM_LANDMARKS_WITH_IRISES
    num_tracked_points_left_hand = len(mediapipe_hand_landmark_names)
    num_tracked_points_right_hand = len(mediapipe_hand_landmark_names)
    num_tracked_points_total = (
        len(mediapipe_body_landmark_names)
        + 2 * len(mediapipe_hand_landmark_names)
        + FACEMESH_NUM_LANDMARKS_WITH_IRISES
    )
    mediapipe_tracked_object_names = [
        "pose_landmarks",
        "face_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",
    ]


class MediapipeTrackingParams(BaseTrackingParams):
    use_yolo_crop_method: bool = False
    mediapipe_model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = True
