from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


class BrightestPointModelInfo:
    name: str = "brightest_point"
    tracker_name: str = "BrightestPointTracker"
    num_tracked_points: int = 1

class BrightestPointTrackingParams(BaseTrackingParams):
    num_points: int = 1