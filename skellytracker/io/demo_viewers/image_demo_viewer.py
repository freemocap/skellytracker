from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2

if TYPE_CHECKING:
    from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker


# Constants for key actions
KEY_QUIT = ord("q")


class ImageDemoViewer:
    def __init__(self, tracker: 'BaseTracker', window_title: Optional[str] = None):
        """
        Initialize with a tracker and optional window title and default exposure.
        """
        self.tracker = tracker
        if window_title is None:
            window_title = f"{tracker.__class__.__name__}"
        self.window_title = window_title

    def run(self, image_path: Path):
        """
        Display input image
        """
        image = cv2.imread(str(image_path))

        self.tracker.process_image(image)

        annotated_image = self.tracker.annotated_image

        cv2.imshow(self.window_title, annotated_image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()
