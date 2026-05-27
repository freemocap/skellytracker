from copy import deepcopy
from typing import Dict, List
import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class YOLOObjectRecorder(BaseRecorder):
    def __init__(self):
        super().__init__()
        self.marker_names = []  # Will be populated when recording starts
    
    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        # Store marker names on first recording
        if not self.marker_names:
            self.marker_names = list(tracked_objects.keys())
        
        frame_data = {}
        for marker_name, tracked_object in tracked_objects.items():
            frame_data[marker_name] = {
                "pixel_x": tracked_object.pixel_x,
                "pixel_y": tracked_object.pixel_y,
                "boxes_xyxy": tracked_object.extra.get("boxes_xyxy", None),
                "confidence": tracked_object.extra.get("confidence", 0),
                "class_id": tracked_object.extra.get("class_id", -1),
                "class_name": tracked_object.extra.get("class_name", ""),
                "detected": tracked_object.extra.get("detected", False),
            }
        
        self.recorded_objects.append(frame_data)

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        """
        Process recorded objects into numpy arrays.
        
        Returns:
            Array of shape (num_frames, num_markers, 6) where last dimension is:
            [pixel_x, pixel_y, confidence, class_id, box_x1, box_y1]
            Missing markers will have NaN values
        """
        num_frames = len(self.recorded_objects)
        num_markers = len(self.marker_names) if self.marker_names else 0
        
        if num_frames == 0 or num_markers == 0:
            self.recorded_objects_array = np.array([], dtype=np.float32)
            return self.recorded_objects_array
        
        # Initialize arrays with NaN for missing data
        self.recorded_objects_array = np.full(
            (num_frames, num_markers, 6), np.nan, dtype=np.float32
        )
        
        # Fill arrays
        for frame_idx, frame_data in enumerate(self.recorded_objects):
            for marker_idx, marker_name in enumerate(self.marker_names):
                if marker_name in frame_data:
                    point_data = frame_data[marker_name]
                    
                    # Position
                    self.recorded_objects_array[frame_idx, marker_idx, 0] = (
                        point_data["pixel_x"] if point_data["pixel_x"] is not None else np.nan
                    )
                    self.recorded_objects_array[frame_idx, marker_idx, 1] = (
                        point_data["pixel_y"] if point_data["pixel_y"] is not None else np.nan
                    )
                    
                    # Confidence and class ID
                    self.recorded_objects_array[frame_idx, marker_idx, 2] = point_data["confidence"]
                    self.recorded_objects_array[frame_idx, marker_idx, 3] = point_data["class_id"]
                    
                    # Box coordinates (store top-left corner)
                    if point_data["boxes_xyxy"] is not None:
                        self.recorded_objects_array[frame_idx, marker_idx, 4] = point_data["boxes_xyxy"][0]
                        self.recorded_objects_array[frame_idx, marker_idx, 5] = point_data["boxes_xyxy"][1]
        
        return self.recorded_objects_array
    
    def get_positions_array(self) -> np.ndarray:
        """
        Get just the position data (x, y coordinates).
        
        Returns:
            Array of shape (num_frames, num_markers, 2)
        """
        if not hasattr(self, 'recorded_objects_array'):
            self.process_tracked_objects()
        
        if self.recorded_objects_array.size == 0:
            return np.array([], dtype=np.float32)
        
        return self.recorded_objects_array[:, :, :2]
    
    def get_metadata_array(self) -> np.ndarray:
        """
        Get just the metadata (confidence, class_id, box coordinates).
        
        Returns:
            Array of shape (num_frames, num_markers, 4)
        """
        if not hasattr(self, 'recorded_objects_array'):
            self.process_tracked_objects()
        
        if self.recorded_objects_array.size == 0:
            return np.array([], dtype=np.float32)
        
        return self.recorded_objects_array[:, :, 2:]
    
    def get_marker_names(self) -> List[str]:
        """
        Get the list of marker names in the order they appear in the arrays.
        
        Returns:
            List of marker names
        """
        return self.marker_names
    
    def get_detection_status(self) -> np.ndarray:
        """
        Get detection status for each marker in each frame.
        
        Returns:
            Boolean array of shape (num_frames, num_markers)
        """
        num_frames = len(self.recorded_objects)
        num_markers = len(self.marker_names) if self.marker_names else 0
        
        if num_frames == 0 or num_markers == 0:
            return np.array([], dtype=bool)
        
        detection_status = np.zeros((num_frames, num_markers), dtype=bool)
        
        for frame_idx, frame_data in enumerate(self.recorded_objects):
            for marker_idx, marker_name in enumerate(self.marker_names):
                if marker_name in frame_data:
                    detection_status[frame_idx, marker_idx] = frame_data[marker_name]["detected"]
        
        return detection_status
    
    def get_detection_rate(self) -> Dict[str, float]:
        """
        Calculate detection rate for each marker.
        
        Returns:
            Dictionary mapping marker names to detection rate (0-1)
        """
        detection_status = self.get_detection_status()
        
        if detection_status.size == 0:
            return {}
        
        detection_rates = {}
        for marker_idx, marker_name in enumerate(self.marker_names):
            if detection_status.shape[1] > marker_idx:
                total_frames = detection_status.shape[0]
                detected_frames = np.sum(detection_status[:, marker_idx])
                detection_rates[marker_name] = detected_frames / total_frames if total_frames > 0 else 0
        
        return detection_rates
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of detected classes across all frames.
        
        Returns:
            Dictionary mapping class names to count of detections
        """
        class_counts = {}
        
        for frame_data in self.recorded_objects:
            for marker_name, point_data in frame_data.items():
                if point_data["detected"]:
                    class_name = point_data["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts
