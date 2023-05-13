from typing import Tuple, List, Dict

from pydantic import BaseModel, Field


from typing import Optional

class TrackedObject(BaseModel):
    object_id: str = Field(default_factory=str,
                           description="A unique identifier for the tracked object.")
    pixel_locations: list = Field(default_factory=list,
                                  description="A list of pixel locations of the tracked object in the image.")
    bounding_boxes: Optional[list] = Field(default_factory=list,
                                           description="A list of bounding boxes of the tracked object in the image.")

    def add_data(self, pixel_location, bounding_box=None):
        self.pixel_locations.append(pixel_location)
        if bounding_box:
            self.bounding_boxes.append(bounding_box)

class FrameData(BaseModel):
    tracked_objects: Dict[int, TrackedObject] = Field(default_factory=dict)

    def add_tracked_object(self, object_id, pixel_location, bounding_box, coordinate):
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = TrackedObject(object_id=object_id)

        self.tracked_objects[object_id].add_data(pixel_location, bounding_box, coordinate)

class TrackingData(BaseModel):
    frames: Dict[int, FrameData] = Field(default_factory=dict)

    def add_frame_data(self, frame_number, frame_data):
        self.frames[frame_number] = frame_data

    def get_frame_data(self, frame_number):
        return self.frames.get(frame_number)

    def get_time_series_data(self, object_id) -> List[Tuple[int, dict, dict, tuple]]:
        time_series_data = []

        for frame_number, frame_data in self.frames.items():
            tracked_object = frame_data.tracked_objects.get(object_id)
            if tracked_object:
                time_series_data.append((frame_number, tracked_object.pixel_locations[-1],
                                         tracked_object.bounding_boxes[-1], tracked_object.coordinates[-1]))

        return time_series_data
