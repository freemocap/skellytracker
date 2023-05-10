import cv2
import logging
from pathlib import Path
from typing import Union


from skelly_tracker.trackers.mediapipe.mediapipe_parameters_model import MediapipeParametersModel
from skelly_tracker.trackers.mediapipe.mediapipe_skeleton_detector import MediaPipeSkeletonDetector

logger = logging.getLogger(__name__)


def run_me(image_path: Union[str, Path]):
    logger.info(f"Seeing if the mediapipe skeleton detector will run")

    img = cv2.imread(filename=str(image_path))

    logger.info(f"setting up the mediapipe parameter model")
    parameter_model = MediapipeParametersModel(static_image_mode=True)

    skeleton_detector = MediaPipeSkeletonDetector(parameter_model=parameter_model)
    detection_output = skeleton_detector.detect_skeleton_in_image(raw_image=img)

    logger.info(f"image processed successfully")

    annotated_image = detection_output.annotated_image
    cv2.imshow("annotated_image", annotated_image)

    cv2.waitKey(0)
  
    cv2.destroyAllWindows()
    logger.info(f"image display closed")

    return detection_output.mediapipe_results
