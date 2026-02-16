from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseObservation
from skellytracker.trackers.base_tracker.point_cloud import PointCloud
from skellytracker.trackers.mediapipe_tracker.face.get_mediapipe_face_info import (
    MEDIAPIPE_FACE_CONTOURS_INDICIES,
    MEDIAPIPE_FACE_CONTOURS_NAMES,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import NUM_FACE_LANDMARKS

_FACE_NAMES: tuple[str, ...] = tuple(f"face_{i:04d}" for i in range(NUM_FACE_LANDMARKS))
_CONTOUR_NAMES: tuple[str, ...] = tuple(MEDIAPIPE_FACE_CONTOURS_NAMES)


@dataclass(slots=True)
class MediapipeFaceObservation(BaseObservation):
    """
    Face observation storing the full 478-point tessellation in a PointCloud.

    The contour subset is derived on access via index lookup.
    """

    tracker_type: str = field(default="mediapipe_face", init=False)
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)

    # Full 478-point tessellation
    points: PointCloud = field(default_factory=lambda: PointCloud.empty(_FACE_NAMES))

    # Blendshape coefficients (52 FACS-like scores), or None
    face_blendshapes: dict[str, float] | None = None

    @classmethod
    def from_detection_results(
        cls,
        frame_number: int,
        face_landmarker_result: "mp.tasks.vision.FaceLandmarkerResult",
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

        blendshapes: dict[str, float] | None = None
        if face_landmarker_result.face_blendshapes and len(face_landmarker_result.face_blendshapes) > 0:
            blendshapes = {
                cat.category_name: cat.score
                for cat in face_landmarker_result.face_blendshapes[0]
            }

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
        face_landmarker_result: "mp.tasks.vision.FaceLandmarkerResult",
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

        blendshapes: dict[str, float] | None = None
        if face_landmarker_result.face_blendshapes and len(face_landmarker_result.face_blendshapes) > 0:
            blendshapes = {
                cat.category_name: cat.score
                for cat in face_landmarker_result.face_blendshapes[0]
            }

        cloud = PointCloud(names=_FACE_NAMES, xyz=face_xyz, visibility=face_vis)

        return cls(
            frame_number=frame_number,
            image_size=full_image_size,
            points=cloud,
            face_blendshapes=blendshapes,
        )

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
        return len(MEDIAPIPE_FACE_CONTOURS_INDICIES)

    @property
    def face_contour_landmarks_xyz(self) -> NDArray:
        """Extract the face contour subset from the full tessellation."""
        if not self.has_detection:
            return np.full((self.num_face_contour_points, 3), np.nan)
        indices = [int(name.rsplit("_", 1)[1]) for name in MEDIAPIPE_FACE_CONTOURS_NAMES]
        return self.points.xyz[indices]

    @property
    def face_contour_visibility(self) -> NDArray:
        if not self.has_detection:
            return np.zeros(self.num_face_contour_points)
        indices = [int(name.rsplit("_", 1)[1]) for name in MEDIAPIPE_FACE_CONTOURS_NAMES]
        return self.points.visibility[indices]
