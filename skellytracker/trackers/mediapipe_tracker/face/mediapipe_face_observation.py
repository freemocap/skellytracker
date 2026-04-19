from dataclasses import dataclass, field

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.names_and_connections import (
    MEDIAPIPE_FACE_CONTOUR_DEFINITION,
    MEDIAPIPE_FACE_TESSELATED_DEFINITION,
)

_FACE_NAMES: tuple[str, ...] = MEDIAPIPE_FACE_TESSELATED_DEFINITION.tracked_points
NUM_FACE_LANDMARKS: int = MEDIAPIPE_FACE_TESSELATED_DEFINITION.num_tracked_points

# Row indices into the 478-point tesselation for the contour subset.
# Precomputed once so extracting the contour view is a plain slice of the
# full PointCloud.
_CONTOUR_INDICES: tuple[int, ...] = tuple(
    MEDIAPIPE_FACE_TESSELATED_DEFINITION.index_of(name)
    for name in MEDIAPIPE_FACE_CONTOUR_DEFINITION.tracked_points
)


@dataclass(slots=True)
class MediapipeFaceObservation(BaseObservation):
    """
    Face observation storing the full 478-point tessellation in a PointCloud.

    The contour subset is derived on access via index lookup.
    """

    tracker_type: str = field(default="mediapipe_face", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)

    points: PointCloud = field(default_factory=MEDIAPIPE_FACE_TESSELATED_DEFINITION.empty_point_cloud)

    face_blendshapes: dict[str, float] | None = None

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        face_landmarker_result: FaceLandmarkerResult,
        image_size: tuple[int, int],
    ) -> "MediapipeFaceObservation":
        """Convert a FaceLandmarkerResult into a MediapipeFaceObservation."""
        height, width = image_size

        if len(face_landmarker_result.face_landmarks) == 0:
            return cls(frame_number=frame_number, image_size=image_size)

        landmarks = face_landmarker_result.face_landmarks[0]

        face_xyz = np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        face_vis = np.zeros(NUM_FACE_LANDMARKS)

        num_detected = len(landmarks)
        for i in range(min(num_detected, NUM_FACE_LANDMARKS)):
            lm = landmarks[i]
            face_xyz[i] = (lm.x * width, lm.y * height, lm.z * width)
            face_vis[i] = lm.presence if lm.presence is not None else 1.0

        blendshapes = cls._extract_blendshapes(face_landmarker_result)

        cloud = PointCloud(names=_FACE_NAMES, xyz=face_xyz, visibility=face_vis)

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            points=cloud,
            face_blendshapes=blendshapes,
        )

    @classmethod
    def from_crop_results(
        cls,
        frame_number: int,
        face_landmarker_result: FaceLandmarkerResult,
        crop_origin: tuple[int, int],
        crop_size: tuple[int, int],
        full_image_size: tuple[int, int],
    ) -> "MediapipeFaceObservation":
        """Convert a FaceLandmarkerResult from a crop back to full-image coordinates."""
        crop_h, crop_w = crop_size
        y_off, x_off = crop_origin

        if len(face_landmarker_result.face_landmarks) == 0:
            return cls(frame_number=frame_number, image_size=full_image_size)

        landmarks = face_landmarker_result.face_landmarks[0]

        face_xyz = np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        face_vis = np.zeros(NUM_FACE_LANDMARKS)

        num_detected = len(landmarks)
        for i in range(min(num_detected, NUM_FACE_LANDMARKS)):
            lm = landmarks[i]
            face_xyz[i] = (lm.x * crop_w + x_off, lm.y * crop_h + y_off, lm.z * crop_w)
            face_vis[i] = lm.presence if lm.presence is not None else 1.0

        blendshapes = cls._extract_blendshapes(face_landmarker_result)

        cloud = PointCloud(names=_FACE_NAMES, xyz=face_xyz, visibility=face_vis)

        return cls(
            frame_number=frame_number,
            image_size=full_image_size,
            points=cloud,
            face_blendshapes=blendshapes,
        )

    @staticmethod
    def _extract_blendshapes(result: FaceLandmarkerResult) -> dict[str, float] | None:
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            return {cat.category_name: cat.score for cat in result.face_blendshapes[0]}
        return None

    @property
    def has_detection(self) -> bool:
        return self.points.n_valid > 0

    @property
    def face_landmarks_xyz(self) -> NDArray:
        return self.points.xyz

    @property
    def face_visibility(self) -> NDArray:
        return self.points.visibility

    @property
    def num_face_contour_points(self) -> int:
        return len(_CONTOUR_INDICES)

    @property
    def face_contour_landmarks_xyz(self) -> NDArray:
        """Extract the face contour subset from the full tessellation."""
        if not self.has_detection:
            return np.full((self.num_face_contour_points, 3), np.nan)
        return self.points.xyz[list(_CONTOUR_INDICES)]

    @property
    def face_contour_visibility(self) -> NDArray:
        if not self.has_detection:
            return np.zeros(self.num_face_contour_points)
        return self.points.visibility[list(_CONTOUR_INDICES)]
