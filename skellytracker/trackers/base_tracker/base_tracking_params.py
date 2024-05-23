from pydantic import BaseModel


class BaseTrackingParams(BaseModel):
    num_processes: int = 1
    run_image_tracking: bool = True
