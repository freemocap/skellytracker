from enum import Enum

from skellytracker.trackers.charuco_tracker import CharucoTracker


class SkellyTrackerTypes(Enum):
    DUMMY = None
    CHARUCO = CharucoTracker
