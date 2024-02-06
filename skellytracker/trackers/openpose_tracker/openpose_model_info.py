from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


class OpenPoseTrackingParams(BaseTrackingParams):
    openpose_exe_path: str
    output_json_path: str
    net_resolution: str = "-1x320"
    number_people_max: int = 1
    write_video: bool = True
    openpose_output_resolution: str = "-1x-1"
