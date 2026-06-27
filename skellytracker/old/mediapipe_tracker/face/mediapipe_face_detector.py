import logging
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

from skellytracker.old.base_tracker.base_tracker_abcs import BaseDetector
from skellytracker.old.mediapipe_tracker.face.mediapipe_face_config import MediapipeFaceConfig
from skellytracker.old.mediapipe_tracker.face.mediapipe_face_observation import MediapipeFaceObservation
from skellytracker.old.mediapipe_tracker.mediapipe_model_manager import get_face_model_path
from skellytracker.old.mediapipe_tracker.names_and_connections import (
    MEDIAPIPE_FACE_TESSELATED_DEFINITION, MEDIAPIPE_FACE_CONTOUR_DEFINITION,
)

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

@dataclass
class MediapipeFaceDetector(BaseDetector):
    """Wraps MediaPipe FaceLandmarker in IMAGE mode (for use with crops)."""

    config: MediapipeFaceConfig
    landmarker: FaceLandmarker

    @classmethod
    def create(cls, config: MediapipeFaceConfig) -> "MediapipeFaceDetector":
        model_path = get_face_model_path()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=config.num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=config.output_face_blendshapes,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        return cls(config=config, landmarker=landmarker, tracked_object=MEDIAPIPE_FACE_CONTOUR_DEFINITION)

    def detect(self, frame_number: int, image: np.ndarray) -> MediapipeFaceObservation:
        """Detect face in a full image."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.landmarker.detect(image=mp_image)
        return MediapipeFaceObservation.from_detection_results(
            frame_number=frame_number,
            face_landmarker_result=result,
            image_size=(image.shape[0], image.shape[1]),
        )

    def detect_in_crop(
        self,
        frame_number: int,
        crop: np.ndarray,
        crop_origin: tuple[int, int],
        full_image_size: tuple[int, int],
    ) -> MediapipeFaceObservation:
        """Detect face in a cropped image and map coordinates back to full image space."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self.landmarker.detect(image=mp_image)
        return MediapipeFaceObservation.from_crop_results(
            frame_number=frame_number,
            face_landmarker_result=result,
            crop_origin=crop_origin,
            crop_size=(crop.shape[0], crop.shape[1]),
            full_image_size=full_image_size,
        )

    def close(self) -> None:
        self.landmarker.close()
