from pydantic import BaseModel


class MediapipeParametersModel(BaseModel):
    model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    skip_2d_image_tracking: bool = False
