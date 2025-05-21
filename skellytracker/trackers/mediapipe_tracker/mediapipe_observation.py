from numpydantic import NDArray, Shape
from pydantic import ConfigDict
from typing import NamedTuple

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, LandmarkList # linter sees an error here, but it runs fine
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.mediapipe_tracker.get_mediapipe_face_info import MEDIAPIPE_FACE_CONTOURS_INDICIES, \
    MEDIAPIPE_FACE_CONTOURS_NAMES
from typing import NamedTuple

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, \
    LandmarkList  # linter sees an error here, but it runs fine
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.mediapipe_tracker.get_mediapipe_face_info import MEDIAPIPE_FACE_CONTOURS_INDICIES, \
    MEDIAPIPE_FACE_CONTOURS_NAMES

MediapipeResults = NamedTuple

# TODO: use numpydantic to fix numpy type hints for this
class MediapipeObservation(BaseObservation):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_number: int  # the frame number of the image in which this observation was made
    pose_landmarks: NormalizedLandmarkList | None
    pose_world_landmarks: LandmarkList | None
    right_hand_landmarks: NormalizedLandmarkList | None
    left_hand_landmarks: NormalizedLandmarkList | None
    face_landmarks: NormalizedLandmarkList | None
    segmentation_mask: np.ndarray | None
    image_size: tuple[int, int]

    @classmethod
    def from_detection_results(cls,
                               frame_number: int,
                               mediapipe_results: MediapipeResults,
                               image_size: tuple[int, int],
                               include_segmentation_mask: bool = True):
        if include_segmentation_mask:  # TODO: make sure we don't get a missing attribute error
            segmentation_mask = mediapipe_results.segmentation_mask
        else:
            segmentation_mask = None
        return cls(
            frame_number=frame_number,
            pose_landmarks=mediapipe_results.pose_landmarks,
            pose_world_landmarks=mediapipe_results.pose_world_landmarks,
            right_hand_landmarks=mediapipe_results.right_hand_landmarks,
            left_hand_landmarks=mediapipe_results.left_hand_landmarks,
            face_landmarks=mediapipe_results.face_landmarks,
            segmentation_mask=segmentation_mask,
            image_size=image_size
        )

    @property
    def body_landmark_names(self) -> list[str]:
        return [f"body.{landmark.name.lower()}" for landmark in mp_holistic.PoseLandmark]

    @property
    def hand_landmark_names(self) -> list[str]:
        return [landmark.name.lower() for landmark in mp_holistic.HandLandmark]

    @property
    def right_hand_landmark_names(self) -> list[str]:
        return [f"right_hand.{landmark}" for landmark in self.hand_landmark_names]

    @property
    def left_hand_landmark_names(self) -> list[str]:
        return [f"left_hand.{landmark}" for landmark in self.hand_landmark_names]

    @property
    def face_contour_landmark_names(self) -> list[str]:
        return MEDIAPIPE_FACE_CONTOURS_NAMES

    @property
    def num_body_points(self) -> int:
        return len(self.body_landmark_names)

    @property
    def num_single_hand_points(self) -> int:
        return len(self.hand_landmark_names)

    @property
    def num_face_tesselation_points(self) -> int:
        return FACEMESH_NUM_LANDMARKS_WITH_IRISES

    @property
    def num_face_contour_points(self) -> int:
        return len(MEDIAPIPE_FACE_CONTOURS_INDICIES)

    @property
    def num_total_points(self) -> int:
        return self.num_body_points + (2 * self.num_single_hand_points) + self.num_face_tesselation_points

    @property
    def body_points_xyz(self) -> NDArray[Shape["* body points, 3"], float]:
        if self.pose_landmarks is None:
            return np.full((self.num_body_points, 3), np.nan)

        return self._landmarks_to_array(self.pose_landmarks)

    @property
    def right_hand_points_xyz(self) -> NDArray[Shape["* right hand points, 3"], float]:
        if self.right_hand_landmarks is None:
            return np.full((self.num_single_hand_points, 3), np.nan)

        return self._landmarks_to_array(self.right_hand_landmarks)

    @property
    def left_hand_points_xyz(self) -> NDArray[Shape["* left hand points, 3"], float]:
        if self.left_hand_landmarks is None:
            return np.full((self.num_single_hand_points, 3), np.nan)

        return self._landmarks_to_array(self.left_hand_landmarks)

    @property
    def face_tesselation_points_xyz(self) -> NDArray[Shape["* face tessellation points, 3"], float]:
        if self.face_landmarks is None:
            return np.full((self.num_face_tesselation_points, 3), np.nan)

        return self._landmarks_to_array(self.face_landmarks)

    @property
    def face_contour_points_xyz(self) -> NDArray[Shape["* face contour points, 3"], float]:
        all_face_landmarks = self.face_tesselation_points_xyz
        xyz = all_face_landmarks[list(MEDIAPIPE_FACE_CONTOURS_INDICIES)]
        if len(xyz) != self.num_face_contour_points:
            raise ValueError(f"Expected {self.num_face_contour_points} face contour points, got {len(xyz)}")
        return xyz

    def _landmarks_to_array(self, landmarks: NormalizedLandmarkList) -> NDArray[Shape["* all points, 3"], float]:
        landmark_array = np.array(
            [
                (landmark.x, landmark.y, landmark.z)
                for landmark in landmarks.landmark
            ]
        )

        # convert from normalized image coordinates to pixel coordinates
        landmark_array *= np.array([self.image_size[0], self.image_size[1],
                                    self.image_size[0]])  # multiply z by image width per mediapipe docs

        return landmark_array

    def all_points(self, dimensions:int, face_type: str = "contour",  scale_by:float=1.0) -> dict[str, tuple]:
        if not dimensions in [2, 3]:
            raise ValueError(f"Invalid dimensions: {dimensions}")

        all_points_by_name = {}
        body_xyz = self.body_points_xyz.copy()* scale_by
        right_hand_xyz = self.right_hand_points_xyz.copy() * scale_by
        left_hand_xyz = self.left_hand_points_xyz.copy()* scale_by
        if face_type == "tesselation":
            face_xyz = self.face_tesselation_points_xyz.copy()* scale_by
        elif face_type == "contour":
            face_xyz = self.face_contour_points_xyz.copy()* scale_by
        else:
            raise ValueError(f"Invalid face type: {face_type}")

        for index, point_name in enumerate(self.body_landmark_names):
            all_points_by_name[point_name] = tuple(body_xyz[index, :dimensions])

        for index, point_name in enumerate(self.right_hand_landmark_names):
            all_points_by_name[point_name] = right_hand_xyz[index, :dimensions]

        for index, point_name in enumerate(self.left_hand_landmark_names):
            all_points_by_name[point_name] = left_hand_xyz[index, :dimensions]

        for index, point_name in enumerate(self.face_contour_landmark_names):
            all_points_by_name[point_name] = face_xyz[index, :dimensions]

        return all_points_by_name

    def to_array(self) -> NDArray[Shape["533, 3"], float]:
        return np.concatenate(
            # this order matters, do not change
            (
                self.body_points_xyz,
                self.right_hand_points_xyz,
                self.left_hand_points_xyz,
                self.face_tesselation_points_xyz,
            ),
            axis=0,
        )
    

MediapipeObservations = list[MediapipeObservation]
