from typing import List, Optional
from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams


yolo_object_model_dictionary = {
    "nano": "yolo11n.pt",
    "small": "yolo11s.pt",
    "medium": "yolo11m.pt",
    "large": "yolo11l.pt",
    "extra_large": "yolo11x.pt",
}


class YOLOObjectTrackingParams(BaseTrackingParams):
    model_path: Optional[str] = None
    model_size: str = "medium"
    person_only: bool = False
    confidence_threshold: float = 0.5
    classes_to_track: Optional[List[int]] = None
    max_detections: int = 10


from skellytracker.trackers.yolo_object_tracker.yolo_object_tracker import YOLOObjectTracker

class YOLOObjectModelInfo:
    """Model info for YOLO object tracker."""
    
    # Class attributes
    name = "yolo_object_tracker"
    tracker_name = "YOLOObjectTracker"
    tracker = YOLOObjectTracker
    
    # These will be set dynamically based on model
    model_name = ""
    landmark_names = []
    num_tracked_points = 0
    tracked_object_names = []
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "medium",
        person_only: bool = False,
        confidence_threshold: float = 0.5,
        classes_to_track: Optional[List[int]] = None,
        max_detections: int = 10,
    ):
        """
        Initialize YOLO object model info.
        
        Args:
            model_path: Path to custom trained model
            model_size: Size of pre-trained model
            person_only: Whether to track only person class
            confidence_threshold: Minimum confidence for detections
            classes_to_track: List of class IDs to track
            max_detections: Maximum number of detections per frame
        """
        # Create tracker instance
        self.tracker_instance = YOLOObjectTracker(
            model_path=model_path,
            model_size=model_size,
            person_only=person_only,
            confidence_threshold=confidence_threshold,
            classes_to_track=classes_to_track,
            max_detections=max_detections,
        )
        
        # Extract info from tracker
        self.model_name = model_path or f"yolo11{model_size[0]}"  # e.g., "best.pt" or "yolo11m"
        self.tracked_object_names = list(self.tracker_instance.tracked_objects.keys())
        self.landmark_names = self.tracked_object_names
        self.num_tracked_points = len(self.tracked_object_names)
        self.class_names = self.tracker_instance.class_names
        
        # Store configuration
        self.model_path = model_path
        self.model_size = model_size
        self.person_only = person_only
        self.confidence_threshold = confidence_threshold
        self.classes_to_track = classes_to_track
        self.max_detections = max_detections
    
    def get_tracker(self) -> YOLOObjectTracker:
        """Get the tracker instance."""
        return self.tracker_instance
    
    def get_config(self) -> dict:
        """Get the configuration as a dictionary."""
        return {
            'model_path': self.model_path,
            'model_size': self.model_size,
            'person_only': self.person_only,
            'confidence_threshold': self.confidence_threshold,
            'classes_to_track': self.classes_to_track,
            'max_detections': self.max_detections,
            'tracked_object_names': self.tracked_object_names,
            'class_names': self.class_names,
            'num_tracked_points': self.num_tracked_points,
        }
    
    def update_config(
        self,
        model_path: Optional[str] = None,
        model_size: Optional[str] = None,
        person_only: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
        classes_to_track: Optional[List[int]] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        """Update the configuration."""
        # Update parameters
        if model_path is not None:
            self.model_path = model_path
        if model_size is not None:
            self.model_size = model_size
        if person_only is not None:
            self.person_only = person_only
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if classes_to_track is not None:
            self.classes_to_track = classes_to_track
        if max_detections is not None:
            self.max_detections = max_detections
        
        # Recreate tracker instance with updated configuration
        self.tracker_instance = YOLOObjectTracker(
            model_path=self.model_path,
            model_size=self.model_size,
            person_only=self.person_only,
            confidence_threshold=self.confidence_threshold,
            classes_to_track=self.classes_to_track,
            max_detections=self.max_detections,
        )
        
        # Update extracted info
        self.model_name = self.model_path or f"yolo11{self.model_size[0]}"
        self.tracked_object_names = list(self.tracker_instance.tracked_objects.keys())
        self.landmark_names = self.tracked_object_names
        self.num_tracked_points = len(self.tracked_object_names)
        self.class_names = self.tracker_instance.class_names
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get class ID for a given class name."""
        for class_id, name in self.class_names.items():
            if name == class_name:
                return class_id
        return None
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID."""
        return self.class_names.get(class_id, f"class_{class_id}")
    
    def get_all_class_names(self) -> List[str]:
        """Get all class names from the model."""
        return list(self.class_names.values())
    
    def get_all_class_ids(self) -> List[int]:
        """Get all class IDs from the model."""
        return list(self.class_names.keys())
    
    def __str__(self) -> str:
        """String representation of the model info."""
        config = self.get_config()
        return (
            f"YOLOObjectModelInfo:\n"
            f"  Model: {config['model_path'] or config['model_size']}\n"
            f"  Tracked objects: {config['tracked_object_names']}\n"
            f"  Number of classes: {len(config['class_names'])}\n"
            f"  Class names: {list(config['class_names'].values())}\n"
            f"  Confidence threshold: {config['confidence_threshold']}\n"
            f"  Max detections: {config['max_detections']}\n"
            f"  Person only: {config['person_only']}\n"
            f"  Classes to track: {config['classes_to_track']}"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the model info."""
        return f"YOLOObjectModelInfo(model_path={self.model_path}, " \
               f"tracked_objects={self.tracked_object_names}, " \
               f"num_classes={len(self.class_names)})"


# Example usage
if __name__ == "__main__":
    # Example 1: Using a custom trained model
    model_info = YOLOObjectModelInfo(
        model_path="path/to/your/best.pt",
        person_only=False,
        confidence_threshold=0.3,
        max_detections=3,
    )
    
    print(model_info)
    print(f"\nAll class names: {model_info.get_all_class_names()}")
    
    # Get tracker instance
    tracker = model_info.get_tracker()
    
    # Example 2: Using pre-trained model
    model_info2 = YOLOObjectModelInfo(
        model_size="medium",
        person_only=True,  # Track only people
        confidence_threshold=0.5,
        max_detections=5,
    )
    
    print("\n" + str(model_info2))
