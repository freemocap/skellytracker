from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class TrackedObject:
    object_id: str
    pixel_locations: List[Tuple[int, int]] = field(default_factory=list)
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = field(default_factory=list)

    def add_data(self, pixel_location: Tuple[int, int], bounding_box: Optional[Tuple[int, int, int, int]] = None):
        self.pixel_locations.append(pixel_location)
        if bounding_box:
            self.bounding_boxes.append(bounding_box)


@dataclass
class FrameData:
    tracked_objects: Dict[int, TrackedObject] = field(default_factory=dict)

    def add_tracked_object(self, object_id: int, pixel_location: Tuple[int, int], bounding_box: Optional[Tuple[int, int, int, int]] = None):
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = TrackedObject(object_id=object_id)

        self.tracked_objects[object_id].add_data(pixel_location, bounding_box)


@dataclass
class TrackingData:
    frames: Dict[int, FrameData] = field(default_factory=dict)

    def add_frame_data(self, frame_number: int, frame_data: FrameData):
        self.frames[frame_number] = frame_data

    def get_frame_data(self, frame_number: int) -> Optional[FrameData]:
        return self.frames.get(frame_number)

    def get_time_series_data(self, object_id: int) -> List[Tuple[int, Tuple[int, int], Tuple[int, int, int, int]]]:
        time_series_data = []

        for frame_number, frame_data in self.frames.items():
            tracked_object = frame_data.tracked_objects.get(object_id)
            if tracked_object:
                time_series_data.append((frame_number, tracked_object.pixel_locations[-1],
                                         tracked_object.bounding_boxes[-1] if tracked_object.bounding_boxes else None))

        return time_series_data
