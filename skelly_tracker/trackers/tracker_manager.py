import cv2
import argparse
from typing import List

import skelly_tracker
from skelly_tracker.trackers.base_tracker.base_tracker import BaseTracker
from skelly_tracker.trackers.bright_point_tracker.brightest_point_tracker import BrightestPointTracker
from skelly_tracker.trackers.webcam_demo_viewer.webcam_demo_viewer import WebcamDemoViewer


class TrackerManager:
    def __init__(self, trackers: List[BaseTracker]):
        self.trackers = trackers

    def process_image(self, image: cv2.VideoCapture):
        for tracker in self.trackers:
            tracker.process_image(image)

    def demo(self):
        self.trackers[0].demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tracker Manager CLI.')
    parser.add_argument('--trackers', nargs='+', help='List of trackers to be used.')
    args = parser.parse_args()

    # Here, we assume that each tracker name corresponds to a class in the skelly_tracker.trackers module
    # and that each class has a no-argument constructor.
    # You will need to adjust this if your trackers need arguments to their constructors,
    # or if the trackers are not in the skelly_tracker.trackers module.
    trackers = [BrightestPointTracker()]

    manager = TrackerManager(trackers)
    manager.demo()
