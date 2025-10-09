from pathlib import Path
from pydantic import BaseModel, Field
from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseRecorder, CumulativeBaseTracker
from skellytracker.trackers.dlc_tracker.dlc_annotator import DeepLabCutAnnotatorConfig, DeepLabCutImageAnnotator
from skellytracker.trackers.dlc_tracker.dlc_detector import DeepLabCutDetector, DeepLabCutDetectorConfig
from skellytracker.trackers.dlc_tracker.dlc_observation import DeepLabCutObservation

class DeepLabCutTrackerConfig(BaseModel):
    detector_config: DeepLabCutDetectorConfig
    annotator_config: DeepLabCutAnnotatorConfig

class DeepLabCutRecorder(BaseRecorder):
    pass

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
    
    def process_video(self, input_video_filepath: Path, output_video_filepath: Path, **kwargs) -> list[DeepLabCutObservation]:
        observations = self.detector.detect_video(input_video_filepath, **kwargs)

        self.recorder.add_observations(observations=observations)
        # TODO: annotate video

        return observations