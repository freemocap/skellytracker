from enum import Enum


class MediapipeModelInfo(Enum):
    num_tracked_points_total = 533
    num_tracked_points_body = 33
    num_tracked_points_face = 468
    num_tracked_points_left_hand = 21
    num_tracked_points_right_hand = 21
    mediapipe_tracked_object_names = ["pose_landmarks", "face_landmarks", "left_hand_landmarks", "right_hand_landmarks"]