import numpy as np
from typing import Dict, List, Optional
from ultralytics import YOLO
import logging

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.yolo_object_tracker.yolo_object_model_info import (
    yolo_object_model_dictionary,
)
from skellytracker.trackers.yolo_object_tracker.yolo_object_recorder import (
    YOLOObjectRecorder,
)

logger = logging.getLogger(__name__)


class YOLOObjectTracker(BaseTracker):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = "nano",
        person_only: bool = False,
        confidence_threshold: float = 0.5,
        classes_to_track: Optional[List[int]] = None,
        max_detections: int = 10,
        tracked_object_names: Optional[List[str]] = None,
    ):
        """
        Initialize a YOLO object tracker.
        
        Args:
            model_path: Path to custom trained model (e.g., "best.pt")
            model_size: Size of pre-trained model if no custom model provided
            person_only: Whether to track only person class (class 0)
            confidence_threshold: Minimum confidence for detections
            classes_to_track: List of class IDs to track (overrides person_only)
            max_detections: Maximum number of detections per frame
            tracked_object_names: Names for tracked objects (will use model class names if None)
        """
        # Load model
        if model_path:
            self.model = YOLO(model_path)
            self.custom_model = True
        else:
            pytorch_model = yolo_object_model_dictionary[model_size]
            self.model = YOLO(pytorch_model)
            self.custom_model = False
        
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        
        # Get class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        logger.info(f"Model class names: {self.class_names}")
        
        # Determine which classes to track
        if classes_to_track is not None:
            self.classes = classes_to_track
        elif person_only and not self.custom_model:
            self.classes = [0]  # 0 is the YOLO class for person detection
        else:
            self.classes = None  # None includes all classes
        
        # Determine tracked object names
        if tracked_object_names is None:
            if self.classes is not None:
                # Use model class names for the specified classes
                tracked_object_names = []
                for class_id in self.classes:
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    tracked_object_names.append(class_name)
            else:
                # If tracking all classes, use all model class names
                tracked_object_names = [self.class_names.get(i, f"class_{i}") 
                                      for i in range(len(self.class_names))]
        
        # Initialize with tracked objects
        super().__init__(
            tracked_object_names=tracked_object_names,
            recorder=YOLOObjectRecorder(),
        )
        
        # Store additional parameters
        self.model_path = model_path
        self.model_size = model_size
        self.person_only = person_only
        self.classes_to_track = classes_to_track
        
        logger.info(f"Initialized YOLOObjectTracker with {len(tracked_object_names)} tracked objects")
        logger.info(f"Tracked objects: {tracked_object_names}")
        if self.custom_model:
            logger.info(f"Using custom model: {model_path}")
        else:
            logger.info(f"Using pre-trained model: {model_size}")
        logger.info(f"Classes to track: {self.classes}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def process_image(self, image, **kwargs) -> Dict[str, TrackedObject]:
        results = self.model(
            image,
            classes=self.classes,
            max_det=self.max_detections,
            verbose=False,
            conf=self.confidence_threshold,
        )

        # Reset all tracked objects
        for obj_name in self.tracked_objects:
            self.tracked_objects[obj_name].pixel_x = None
            self.tracked_objects[obj_name].pixel_y = None
            self.tracked_objects[obj_name].extra["boxes_xyxy"] = None
            self.tracked_objects[obj_name].extra["confidence"] = 0
            self.tracked_objects[obj_name].extra["class_id"] = -1
            self.tracked_objects[obj_name].extra["class_name"] = ""
            self.tracked_objects[obj_name].extra["detected"] = False
        
        # Process detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            box_xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            # Group detections by class name
            detections_by_class = {}
            for i, (box, conf, class_id) in enumerate(zip(box_xyxy, confidences, class_ids)):
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = []
                
                detections_by_class[class_name].append({
                    'box': box,
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                    'index': i
                })
            
            # For each tracked object (class name), use the highest confidence detection
            for obj_name in self.tracked_objects:
                if obj_name in detections_by_class:
                    # Get the highest confidence detection for this class
                    best_detection = max(detections_by_class[obj_name], key=lambda x: x['confidence'])
                    
                    box = best_detection['box']
                    self.tracked_objects[obj_name].pixel_x = (box[0] + box[2]) / 2
                    self.tracked_objects[obj_name].pixel_y = (box[1] + box[3]) / 2
                    self.tracked_objects[obj_name].extra["boxes_xyxy"] = box
                    self.tracked_objects[obj_name].extra["confidence"] = best_detection['confidence']
                    self.tracked_objects[obj_name].extra["class_id"] = best_detection['class_id']
                    self.tracked_objects[obj_name].extra["class_name"] = best_detection['class_name']
                    self.tracked_objects[obj_name].extra["detected"] = True
        
        self.annotated_image = self.annotate_image(image, results=results, **kwargs)

        return self.tracked_objects

    def annotate_image(self, image: np.ndarray, results, **kwargs) -> np.ndarray:
        return results[0].plot()

    def update_config(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        classes_to_track: Optional[List[int]] = None,
        max_detections: Optional[int] = None,
    ) -> None:
        """
        Update tracker configuration.
        """
        if model_path is not None:
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.custom_model = True
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            # Update tracked objects based on new model
            if classes_to_track is None:
                classes_to_track = self.classes
            
            if classes_to_track is not None:
                new_tracked_names = []
                for class_id in classes_to_track:
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    new_tracked_names.append(class_name)
                
                # Update tracked objects
                self.tracked_objects = {}
                for name in new_tracked_names:
                    self.tracked_objects[name] = TrackedObject(name=name)
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        if classes_to_track is not None:
            self.classes = classes_to_track
        
        if max_detections is not None:
            self.max_detections = max_detections
        
        logger.info(f"Updated YOLOObjectTracker configuration")
        logger.info(f"Tracked objects: {list(self.tracked_objects.keys())}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Classes to track: {self.classes}")
        logger.info(f"Max detections: {self.max_detections}")

    def get_config(self) -> Dict:
        """
        Get current tracker configuration.
        """
        return {
            'model_path': self.model_path,
            'model_size': self.model_size,
            'confidence_threshold': self.confidence_threshold,
            'classes_to_track': self.classes,
            'max_detections': self.max_detections,
            'person_only': self.person_only,
            'custom_model': self.custom_model,
            'class_names': self.class_names,
            'tracked_object_names': list(self.tracked_objects.keys()),
        }


if __name__ == "__main__":
    # Example 1: Track person with default model
    tracker = YOLOObjectTracker(person_only=True)
    tracker.demo()
