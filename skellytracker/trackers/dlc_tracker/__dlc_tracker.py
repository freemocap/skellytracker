from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, CumulativeBaseTracker
from skellytracker.trackers.dlc_tracker.dlc_annotator import DeepLabCutAnnotatorConfig, DeepLabCutImageAnnotator
from skellytracker.trackers.dlc_tracker.dlc_detector import DeepLabCutDetector, DeepLabCutDetectorConfig
from skellytracker.trackers.dlc_tracker.dlc_observation import DeepLabCutObservation

class DeepLabCutTrackerConfig(BaseModel):
    detector_config: DeepLabCutDetectorConfig
    annotator_config: DeepLabCutAnnotatorConfig

class DeepLabCutRecorder(BaseRecorder):
    def load_deeplabcut_csv(self, csv_path: Path, image_size: tuple[int, int] = (1280, 720)) -> list[DeepLabCutObservation]:
        df = pd.read_csv(csv_path)

        df = df.iloc[:, 1:]

        if df.shape[1] % 3 != 0:
            raise ValueError(f"csv file {csv_path} has {df.shape[1]} columns, which is not divisible by 3")

        try:
            points = df.values.reshape(df.shape[0], df.shape[1] // 3, 3)
        except ValueError as e:
            raise ValueError(f"Reshape failed for csv file {csv_path} with shape {df.shape}: {e}")
        
        observations = [
            DeepLabCutObservation(frame_number=i,
                pose_points=points[i, :, :2],
                confidence_values=points[i, :, 2],
                image_size=image_size
            ) for i in range(len(points))
        ]

        self.clear()
        self.add_observations(observations=observations)

        return observations

class DeepLabCutTracker(CumulativeBaseTracker):
    config: DeepLabCutTrackerConfig
    detector: DeepLabCutDetector
    recorder: DeepLabCutRecorder
    annotator: DeepLabCutImageAnnotator | None = None

    @classmethod
    def create(cls, config: DeepLabCutTrackerConfig):
        detector = DeepLabCutDetector.create(config.detector_config)

        return cls(
            config=config,
            detector=detector,
            annotator=DeepLabCutImageAnnotator.create(config.annotator_config),
            recorder=DeepLabCutRecorder(),
        )
    
    def process_video(self, input_video_filepath: Path, **kwargs) -> list[DeepLabCutObservation]:
        observations = self.detector.detect_video(input_video_filepath, **kwargs)

        self.recorder.add_observations(observations=observations)

        return observations
    
    def annotate_image(
        self, image: np.ndarray, latest_observation: DeepLabCutObservation
    ) -> np.ndarray:
        if self.annotator is None:
            raise ValueError("No annotator configured")
        return self.annotator.annotate_image(
            image=image, latest_observation=latest_observation
        )
    
    def annotate_video(self, input_video_filepath: Path, output_video_filepath: Path, **kwargs) -> None:
        cap = cv2.VideoCapture(str(input_video_filepath))

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(self.recorder.observations) < num_frames:
            cap.release()
            raise ValueError(f"Not enough observations to annotate video (video has {num_frames} frames, but only {len(self.recorder.observations)} observations)")

        writer = cv2.VideoWriter(
            str(output_video_filepath),
            cv2.VideoWriter.fourcc(*"AVC1"),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
        
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.annotate_image(frame, self.recorder.observations[i])

            writer.write(annotated_frame)

            i += 1

        cap.release()
        writer.release()

if __name__ == "__main__":
    tracker = DeepLabCutTracker.create(
        DeepLabCutTrackerConfig(
            detector_config=DeepLabCutDetectorConfig(dlc_config="/Users/philipqueen/clicker_testing/clicker_testing/config.yaml"),
            annotator_config=DeepLabCutAnnotatorConfig(),
        )
    )
    input_video = "/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos/sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam2.mp4"

    tracker.process_video(
        input_video_filepath=Path(input_video),
    )
    tracker.annotate_video(
        input_video_filepath=Path(input_video),
        output_video_filepath=Path("/Users/philipqueen/clicker_testing/clicker_testing/test_annotated.mp4"),
    )