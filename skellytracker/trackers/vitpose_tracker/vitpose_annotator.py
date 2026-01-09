from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.vitpose_tracker.vitpose_observation import VITPoseObservation
from easy_ViTPose import VitInference

class VITPoseAnnotator(BaseImageAnnotator):
    config: BaseImageAnnotatorConfig
    observations: list[VITPoseObservation]

    @classmethod
    def create(cls, config: BaseImageAnnotatorConfig | None = None):
        if config is None:
            config = BaseImageAnnotatorConfig()
        return cls(config=config, observations=[])
    
    def annotate_image(
            self,
            model: VitInference
        ):

        annotated_image = model.draw(
            show_yolo=True
        )

        return annotated_image
