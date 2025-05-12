import cv2
import requests
import mediapipe as mp
import numpy as np

from typing import Dict, Union
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.mediapipe_blendshape_tracker.mediapipe_blendshape_model_info import (
    MediapipeBlendshapeModelInfo,
)
from skellytracker.trackers.mediapipe_blendshape_tracker.mediapipe_blendshape_recorder import (
    MediapipeBlendshapeRecorder,
)


class MediapipeBlendshapeTracker(BaseTracker):
    def __init__(
        self,
        model_path: Union[Path, str, None] = None,
    ):
        super().__init__(
            tracked_object_names=MediapipeBlendshapeModelInfo.tracked_object_names,
            recorder=MediapipeBlendshapeRecorder(),
        )
        self.model_info = MediapipeBlendshapeModelInfo

        # If model_path not provided, try default model path, and if that doesn't work download model
        if model_path is None:
            model_path = self.get_or_download_mediapipe_blendshape_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path)
        )  # TODO: handle this path properly
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_image(
        self, image: np.ndarray, annotate_image: bool = True, **kwargs
    ) -> Dict[str, TrackedObject]:
        rgb_image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # TODO: may need to convert this into an `mp.Image`, but can't find documentation about that

        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = self.detector.detect(mediapipe_image)

        if len(results.face_blendshapes) == 0:
            self.annotated_image = image
            return {}

        self.tracked_objects["face"].extra["blendshapes"] = [
            blendshape.score for blendshape in results.face_blendshapes[0]
        ]  # TODO: assumes we're only interested in 1 face, but docs say this works for multiple faces??

        if annotate_image:
            self.annotated_image = self.annotate_image(
                image=image,
                tracked_objects=self.tracked_objects,
                face_landmarks=results.face_landmarks[0],
            )

        return self.tracked_objects

    def annotate_image(
        self,
        image: np.ndarray,
        tracked_objects: Dict[str, TrackedObject],
        face_landmarks: list[landmark_pb2.NormalizedLandmark],
        **kwargs,
    ) -> np.ndarray:
        annotated_image = image.copy()

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        return annotated_image

    def get_or_download_mediapipe_blendshape_model(self) -> Path:
        model_path = Path(
            "face_landmarker_v2_with_blendshapes.task"
        )  # TODO: make this path absolute to skellytracker install
        if not model_path.exists():
            r = requests.get(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                stream=True,
                timeout=(5, 60),
            )
            r.raise_for_status()
            model_path.write_bytes(r.content)
        return model_path


if __name__ == "__main__":
    MediapipeBlendshapeTracker().demo()
