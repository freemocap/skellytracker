import multiprocessing as mp
import time
from asyncio import sleep
from typing import List

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.bright_point_tracker.brightest_point_tracker import BrightestPointTracker


class TrackerManager:
    def __init__(self, trackers: List[BaseTracker]):
        self.trackers = trackers
        self.parent_connection, self.child_connection = mp.Pipe()
        self.process = mp.Process(target=self._process_images, args=(self.child_connection, self.trackers))
        self.process.start()

    @staticmethod
    def _process_images(conn, trackers):
        while True:
            time.sleep(0.001)
            image = conn.recv()
            if image is None:
                break
            for tracker in trackers:
                tracker.process_image(image)

    def add_image(self, image):
        self.parent_connection.send(image)

    def demo(self):
        self.trackers[0].demo()

    def stop(self):
        self.parent_connection.send(None)
        self.process.join()



if __name__ == "__main__":

    trackers = [BrightestPointTracker()]

    manager = TrackerManager(trackers)
    manager.demo()
