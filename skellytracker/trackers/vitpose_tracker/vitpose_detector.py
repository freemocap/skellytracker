from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector, BaseDetectorConfig
from easy_ViTPose import VitInference


class VITPoseDetectorConfig(BaseDetectorConfig):
    confidence_threshold: float = 0.5
    model: str = "base" #options are 'small', 'base', 'large', and 'huge'
    device: str|None = None 
    #later can add a bunch of other options, such as YOLO size, YOLO step size, YOLO bb size

class VITPoseDetector(BaseDetector):
    config: VITPoseDetectorConfig
    detector: VitInference

    @classmethod
    def create(cls, config: VITPoseDetectorConfig | None = None):
        if config is None:
            config = VITPoseDetectorConfig()
        return cls(config=config)
    
    @classmethod
    def create(cls, config: VITPoseDetectorConfig | None = None):
        detector = VitInference()

    def _resolve_model_type(self, model):
        mappings = {"small":"s", 
                    "base":"b", 
                    "large": "l",
                    "huge": "h"}

        if model not in mappings:
            raise ValueError(
                f"Invalid model '{model}'. "
                f"Must be one of {list(mappings.keys())}"
            )
        
        return mappings[model]

