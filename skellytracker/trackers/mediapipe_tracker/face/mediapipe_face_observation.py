import numpy as np
from numpydantic import NDArray, Shape
from pydantic import ConfigDict

from skellytracker.trackers.base_tracker.base_tracker_abcs import (
    BaseObservation,
    TrackedPoint2dArray,
    TrackedPointIdString,
    TrackerTypeString,
)
from skellytracker.trackers.mediapipe_tracker.face.get_mediapipe_face_info import (
    MEDIAPIPE_FACE_CONTOURS_INDICIES,
    MEDIAPIPE_FACE_CONTOURS_NAMES,
)
from skellytracker.trackers.mediapipe_tracker.mediapipe_names import NUM_FACE_LANDMARKS


class MediapipeFaceObservation(BaseObservation):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tracker_type: TrackerTypeString = "mediapipe_face"
    frame_number: int
    image_size: tuple[int, int]  # (height, width)

    face_landmarks_xyz: NDArray[Shape["478, 3"], float]  # full tessellation
    face_visibility: NDArray[Shape["478"], float]
    face_blendshapes: dict[str, float] | None  # 52 blendshape scores, or None

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
            return cls(
                frame_number=frame_number,
                image_size=image_size,
                face_landmarks_xyz=np.full((NUM_FACE_LANDMARKS, 3), np.nan),
                face_visibility=np.zeros(NUM_FACE_LANDMARKS),
                face_blendshapes=None,
            )

        landmarks = face_landmarker_result.face_landmarks[0]

        face_xyz = np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        face_vis = np.zeros(NUM_FACE_LANDMARKS)

        num_detected = len(landmarks)
        for i in range(min(num_detected, NUM_FACE_LANDMARKS)):
            lm = landmarks[i]
            face_xyz[i] = (lm.x * width, lm.y * height, lm.z * width)
            face_vis[i] = lm.presence if lm.presence is not None else 1.0

        # Blendshapes
        blendshapes: dict[str, float] | None = None
        if face_landmarker_result.face_blendshapes and len(face_landmarker_result.face_blendshapes) > 0:
            blendshapes = {
                cat.category_name: cat.score
                for cat in face_landmarker_result.face_blendshapes[0]
            }

        return cls(
            frame_number=frame_number,
            image_size=image_size,
            face_landmarks_xyz=face_xyz,
            face_visibility=face_vis,
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
        """Convert a FaceLandmarkerResult from a cropped image back to full-image coordinates."""
        crop_h, crop_w = crop_size
        y_off, x_off = crop_origin

        if len(face_landmarker_result.face_landmarks) == 0:
            return cls(
                frame_number=frame_number,
                image_size=full_image_size,
                face_landmarks_xyz=np.full((NUM_FACE_LANDMARKS, 3), np.nan),
                face_visibility=np.zeros(NUM_FACE_LANDMARKS),
                face_blendshapes=None,
            )

        landmarks = face_landmarker_result.face_landmarks[0]

        face_xyz = np.full((NUM_FACE_LANDMARKS, 3), np.nan)
        face_vis = np.zeros(NUM_FACE_LANDMARKS)

        num_detected = len(landmarks)
        for i in range(min(num_detected, NUM_FACE_LANDMARKS)):
            lm = landmarks[i]
            face_xyz[i] = (
                lm.x * crop_w + x_off,
                lm.y * crop_h + y_off,
                lm.z * crop_w,
            )
            face_vis[i] = lm.presence if lm.presence is not None else 1.0

        # Blendshapes (not affected by crop coordinates)
        blendshapes: dict[str, float] | None = None
        if face_landmarker_result.face_blendshapes and len(face_landmarker_result.face_blendshapes) > 0:
            blendshapes = {
                cat.category_name: cat.score
                for cat in face_landmarker_result.face_blendshapes[0]
            }

        return cls(
            frame_number=frame_number,
            image_size=full_image_size,
            face_landmarks_xyz=face_xyz,
            face_visibility=face_vis,
            face_blendshapes=blendshapes,
        )

    @property
    def has_detection(self) -> bool:
        return not np.isnan(self.face_landmarks_xyz).all()

    @property
    def num_face_contour_points(self) -> int:
        return len(MEDIAPIPE_FACE_CONTOURS_INDICIES)

    @property
    def face_contour_landmarks_xyz(self) -> NDArray[Shape["*, 3"], float]:
        """Extract the face contour subset from the full tessellation."""
        if not self.has_detection:
            return np.full((self.num_face_contour_points, 3), np.nan)

        indices = [int(name.rsplit("_", 1)[1]) for name in MEDIAPIPE_FACE_CONTOURS_NAMES]
        return self.face_landmarks_xyz[indices]

    @property
    def face_contour_visibility(self) -> NDArray[Shape["*"], float]:
        if not self.has_detection:
            return np.zeros(self.num_face_contour_points)
        indices = [int(name.rsplit("_", 1)[1]) for name in MEDIAPIPE_FACE_CONTOURS_NAMES]
        return self.face_visibility[indices]

    def get_confidence_scores(self) -> NDArray[Shape["478"], float]:
        return self.face_visibility

    def to_tracked_points(self, *, confidence_threshold: float | None = None) -> dict[TrackedPointIdString, TrackedPoint2dArray]:
        """Return face contour points (not full tessellation) as tracked points."""
        result: dict[TrackedPointIdString, TrackedPoint2dArray] = {}
        contour_xyz = self.face_contour_landmarks_xyz
        contour_vis = self.face_contour_visibility
        for i, name in enumerate(MEDIAPIPE_FACE_CONTOURS_NAMES):
            if np.isnan(contour_xyz[i]).any():
                continue
            if confidence_threshold is not None and contour_vis[i] < confidence_threshold:
                continue
            result[name] = np.array(contour_xyz[i, :2])
        return result

    def to_2d_array(self, *, confidence_threshold: float | None = None, fill_with_nans: bool = True) -> NDArray[Shape["*, 2"], float]:
        """Return face contour points as 2D array."""
        points_2d = self.face_contour_landmarks_xyz[:, :2].copy()
        if confidence_threshold is not None:
            points_2d = self.filter_by_confidence(
                points=points_2d,
                confidence_scores=self.face_contour_visibility,
                confidence_threshold=confidence_threshold,
                fill_with_nans=fill_with_nans,
            )
        return points_2d
