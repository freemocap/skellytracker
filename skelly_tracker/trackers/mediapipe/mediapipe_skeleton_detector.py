import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import mediapipe as mp
import numpy as np

from skelly_tracker.trackers.mediapipe.mediapipe_parameters_model import (
    MediapipeParametersModel,
)

from skelly_tracker.trackers.mediapipe.mediapipe_skeleton_names_and_connections import (
    mediapipe_tracked_point_names_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class Mediapipe2dNumpyArrays:
    body_frameNumber_trackedPointNumber_XYZ: np.ndarray = None
    body_world_frameNumber_trackedPointNumber_XYZ: np.ndarray = None
    rightHand_frameNumber_trackedPointNumber_XYZ: np.ndarray = None
    leftHand_frameNumber_trackedPointNumber_XYZ: np.ndarray = None
    face_frameNumber_trackedPointNumber_XYZ: np.ndarray = None

    body_frameNumber_trackedPointNumber_confidence: np.ndarray = None

    @property
    def has_data(self):
        return not np.isnan(self.body_frameNumber_trackedPointNumber_XYZ).all()

    @property
    def all_data2d_nFrames_nTrackedPts_XY(self):
        """dimensions will be [number_of_frames , number_of_markers, XY]"""

        if self.body_frameNumber_trackedPointNumber_XYZ is None:
            # if there's no body data, there's no hand or face data either
            return

        if (
            len(self.body_frameNumber_trackedPointNumber_XYZ.shape) == 3
        ):  # multiple frames
            return np.hstack(
                [
                    self.body_frameNumber_trackedPointNumber_XYZ,
                    self.rightHand_frameNumber_trackedPointNumber_XYZ,
                    self.leftHand_frameNumber_trackedPointNumber_XYZ,
                    self.face_frameNumber_trackedPointNumber_XYZ,
                ]
            )
        elif (
            len(self.body_frameNumber_trackedPointNumber_XYZ.shape) == 2
        ):  # single frame
            return np.vstack(
                [
                    self.body_frameNumber_trackedPointNumber_XYZ,
                    self.rightHand_frameNumber_trackedPointNumber_XYZ,
                    self.leftHand_frameNumber_trackedPointNumber_XYZ,
                    self.face_frameNumber_trackedPointNumber_XYZ,
                ]
            )
        else:
            logger.error("data should have either 2 or 3 dimensions")


@dataclass
class Mediapipe2dDataPayload:
    mediapipe_results: Any = None
    annotated_image: np.ndarray = None
    pixel_data_numpy_arrays: Mediapipe2dNumpyArrays = None


class MediaPipeSkeletonDetector:
    def __init__(
        self,
        parameter_model=MediapipeParametersModel(),
    ):
        self._mediapipe_payload_list = []

        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_holistic = mp.solutions.holistic

        self._body_drawing_spec = self._mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1
        )
        self._hand_drawing_spec = self._mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1
        )
        self._face_drawing_spec = self._mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1
        )

        self._holistic_tracker = self._mp_holistic.Holistic(
            model_complexity=parameter_model.model_complexity,
            min_detection_confidence=parameter_model.min_detection_confidence,
            min_tracking_confidence=parameter_model.min_tracking_confidence,
        )
        self._mediapipe_tracked_point_names_dict = mediapipe_tracked_point_names_dict

        self.body_names_list = self._mediapipe_tracked_point_names_dict["body"]
        self.right_hand_names_list = self._mediapipe_tracked_point_names_dict[
            "right_hand"
        ]
        self.left_hand_names_list = self._mediapipe_tracked_point_names_dict[
            "left_hand"
        ]
        self.face_names_list = self._mediapipe_tracked_point_names_dict["face"]

        # TODO - build a better iterator and list of `face_marker_names` that will only pull out the face_counters & iris edges (mp.python.solutions.face_mesh_connections.FACEMESH_CONTOURS, FACE_MESH_IRISES)

        self.number_of_body_tracked_points = len(self.body_names_list)
        self.number_of_right_hand_tracked_points = len(self.right_hand_names_list)
        self.number_of_left_hand_tracked_points = len(self.left_hand_names_list)
        self.number_of_face_tracked_points = (
            mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES
        )

        self.number_of_tracked_points_total = (
            self.number_of_body_tracked_points
            + self.number_of_left_hand_tracked_points
            + self.number_of_right_hand_tracked_points
            + self.number_of_face_tracked_points
        )

    def detect_skeleton_in_image(
        self,
        raw_image: np.ndarray = None,
        annotated_image: np.ndarray = None,
    ) -> Mediapipe2dDataPayload:
        mediapipe_results = self._holistic_tracker.process(
            raw_image
        )  # <-this is where the magic happens, i.e. where the raw image is processed by a convolutional neural network to provide an estimate of joint position in pixel coordinates. Please don't forget that this is insane and should not be possible lol

        annotated_image = self._annotate_image(image=raw_image, mediapipe_results=mediapipe_results)

        mediapipe_single_frame_npy_data = self._list_of_mediapipe_results_to_npy_arrays(
            [mediapipe_results],
            image_width=annotated_image.shape[0],
            image_height=annotated_image.shape[1],
        )
        mediapipe_single_frame_npy_data.body_frameNumber_trackedPointNumber_XYZ = self._threshold_by_confidence(
            mediapipe_single_frame_npy_data.body_frameNumber_trackedPointNumber_XYZ,
            mediapipe_single_frame_npy_data.body_frameNumber_trackedPointNumber_confidence,
            confidence_threshold=0.5,
        )

        return Mediapipe2dDataPayload(
            mediapipe_results=mediapipe_results,
            annotated_image=annotated_image,
            pixel_data_numpy_arrays=mediapipe_single_frame_npy_data,
        )

    def _annotate_image(self, image, mediapipe_results):
        self._mp_drawing.draw_landmarks(
            image=image,
            landmark_list=mediapipe_results.face_landmarks,
            connections=self._mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        self._mp_drawing.draw_landmarks(
            image=image,
            landmark_list=mediapipe_results.face_landmarks,
            connections=self._mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self._mp_drawing.draw_landmarks(
            image=image,
            landmark_list=mediapipe_results.pose_landmarks,
            connections=self._mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        self._mp_drawing.draw_landmarks(
            image=image,
            landmark_list=mediapipe_results.left_hand_landmarks,
            connections=self._mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_hand_connections_style(),
        )

        self._mp_drawing.draw_landmarks(
            image=image,
            landmark_list=mediapipe_results.right_hand_landmarks,
            connections=self._mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_hand_connections_style(),
        )
        return image

    def _list_of_mediapipe_results_to_npy_arrays(
        self,
        mediapipe_results_list: List,
        image_width: Union[int, float],
        image_height: Union[int, float],
    ) -> Mediapipe2dNumpyArrays:
        number_of_frames = len(mediapipe_results_list)
        number_of_spatial_dimensions = (
            3  # this will be 2d XY pixel data, with mediapipe's estimate of Z
        )

        body_frameNumber_trackedPointNumber_XYZ = np.zeros(
            (
                number_of_frames,
                self.number_of_body_tracked_points,
                number_of_spatial_dimensions,
            )
        )
        body_frameNumber_trackedPointNumber_XYZ[:] = np.nan

        body_world_frameNumber_trackedPointNumber_XYZ = np.zeros(
            (
                number_of_frames,
                self.number_of_body_tracked_points,
                number_of_spatial_dimensions,
            )
        )
        body_world_frameNumber_trackedPointNumber_XYZ[:] = np.nan

        body_frameNumber_trackedPointNumber_confidence = np.zeros(
            (number_of_frames, self.number_of_body_tracked_points)
        )
        body_frameNumber_trackedPointNumber_confidence[
            :
        ] = np.nan  # only body markers get a 'confidence' value

        rightHand_frameNumber_trackedPointNumber_XYZ = np.zeros(
            (
                number_of_frames,
                self.number_of_right_hand_tracked_points,
                number_of_spatial_dimensions,
            )
        )
        rightHand_frameNumber_trackedPointNumber_XYZ[:] = np.nan

        leftHand_frameNumber_trackedPointNumber_XYZ = np.zeros(
            (
                number_of_frames,
                self.number_of_left_hand_tracked_points,
                number_of_spatial_dimensions,
            )
        )
        leftHand_frameNumber_trackedPointNumber_XYZ[:] = np.nan

        face_frameNumber_trackedPointNumber_XYZ = np.zeros(
            (
                number_of_frames,
                self.number_of_face_tracked_points,
                number_of_spatial_dimensions,
            )
        )
        face_frameNumber_trackedPointNumber_XYZ[:] = np.nan

        all_body_tracked_points_visible_on_frame_bool_list = []
        all_right_hand_points_visible_on_frame_bool_list = []
        all_left_hand_points_visible_on_frame_bool_list = []
        all_face_points_visible_on_frame_bool_list = []
        all_tracked_points_visible_on_frame_list = []

        for frame_number, frame_results in enumerate(mediapipe_results_list):
            # get the Body data (aka 'pose')
            if frame_results.pose_landmarks is not None:
                for landmark_number, landmark_data in enumerate(
                    frame_results.pose_landmarks.landmark
                ):
                    body_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 0
                    ] = (landmark_data.x * image_width)
                    body_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 1
                    ] = (landmark_data.y * image_height)
                    body_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 2
                    ] = (
                        landmark_data.z
                        * image_width  # z is on roughly the same scale as x, according to mediapipe docs
                    )
                    body_frameNumber_trackedPointNumber_confidence[
                        frame_number, landmark_number
                    ] = (
                        landmark_data.visibility
                    )  # mediapipe calls their 'confidence' score 'visibility'

                for landmark_number, landmark_data in enumerate(
                    frame_results.pose_world_landmarks.landmark
                ):
                    body_world_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 0
                    ] = (landmark_data.x * image_width)
                    body_world_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 1
                    ] = (landmark_data.y * image_height)
                    body_world_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 2
                    ] = (landmark_data.z * image_width)

            # get Right Hand data
            if frame_results.right_hand_landmarks is not None:
                for landmark_number, landmark_data in enumerate(
                    frame_results.right_hand_landmarks.landmark
                ):
                    rightHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 0
                    ] = (landmark_data.x * image_width)
                    rightHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 1
                    ] = (landmark_data.y * image_height)
                    rightHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 2
                    ] = (landmark_data.z * image_width)

            # get Left Hand data
            if frame_results.left_hand_landmarks is not None:
                for landmark_number, landmark_data in enumerate(
                    frame_results.left_hand_landmarks.landmark
                ):
                    leftHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 0
                    ] = (landmark_data.x * image_width)
                    leftHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 1
                    ] = (landmark_data.y * image_height)
                    leftHand_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 2
                    ] = (landmark_data.z * image_width)

            # get Face data
            if frame_results.face_landmarks is not None:
                for landmark_number, landmark_data in enumerate(
                    frame_results.face_landmarks.landmark
                ):
                    face_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 0
                    ] = (landmark_data.x * image_width)
                    face_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 1
                    ] = (landmark_data.y * image_height)
                    face_frameNumber_trackedPointNumber_XYZ[
                        frame_number, landmark_number, 2
                    ] = (landmark_data.z * image_width)

            # check if all tracked points are visible on this frame
            all_body_visible = all(
                sum(
                    np.isnan(
                        body_frameNumber_trackedPointNumber_XYZ[frame_number, :, :]
                    )
                )
                == 0
            )
            all_body_tracked_points_visible_on_frame_bool_list.append(all_body_visible)

            all_right_hand_visible = all(
                sum(
                    np.isnan(
                        rightHand_frameNumber_trackedPointNumber_XYZ[frame_number, :, :]
                    )
                )
                == 0
            )
            all_right_hand_points_visible_on_frame_bool_list.append(
                all_right_hand_visible
            )

            all_left_hand_visible = all(
                sum(
                    np.isnan(
                        leftHand_frameNumber_trackedPointNumber_XYZ[frame_number, :, :]
                    )
                )
                == 0
            )
            all_left_hand_points_visible_on_frame_bool_list.append(
                all_left_hand_visible
            )

            all_face_visible = all(
                sum(
                    np.isnan(
                        face_frameNumber_trackedPointNumber_XYZ[frame_number, :, :]
                    )
                )
                == 0
            )
            all_face_points_visible_on_frame_bool_list.append(all_face_visible)

            all_points_visible = all(
                [
                    all_body_visible,
                    all_right_hand_visible,
                    all_left_hand_visible,
                    all_face_visible,
                ],
            )

            all_tracked_points_visible_on_frame_list.append(all_points_visible)

        return Mediapipe2dNumpyArrays(
            body_frameNumber_trackedPointNumber_XYZ=np.squeeze(
                body_frameNumber_trackedPointNumber_XYZ
            ),
            body_world_frameNumber_trackedPointNumber_XYZ=np.squeeze(
                body_world_frameNumber_trackedPointNumber_XYZ
            ),
            rightHand_frameNumber_trackedPointNumber_XYZ=np.squeeze(
                rightHand_frameNumber_trackedPointNumber_XYZ
            ),
            leftHand_frameNumber_trackedPointNumber_XYZ=np.squeeze(
                leftHand_frameNumber_trackedPointNumber_XYZ
            ),
            face_frameNumber_trackedPointNumber_XYZ=np.squeeze(
                face_frameNumber_trackedPointNumber_XYZ
            ),
            body_frameNumber_trackedPointNumber_confidence=np.squeeze(
                body_frameNumber_trackedPointNumber_confidence
            ),
        )

    def _threshold_by_confidence(
        self,
        data2d_trackedPoint_dim: np.ndarray,
        data2d_trackedPoint_confidence: np.ndarray,
        confidence_threshold: float,
    ):
        threshold_mask = data2d_trackedPoint_confidence < confidence_threshold
        data2d_trackedPoint_dim[threshold_mask, :] = np.nan
        return data2d_trackedPoint_dim
