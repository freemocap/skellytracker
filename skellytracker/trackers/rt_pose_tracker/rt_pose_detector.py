import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import torch

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector, BaseDetectorConfig, TrackerType
from skellytracker.trackers.rt_pose_tracker.names_and_connections import RT_POSE_DEFINITION
from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation
from skellytracker.trackers.rt_pose_tracker.rt_pose_preprocessing import (
    post_process_pose_estimation,
    preprocess,
)

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {list(_DTYPE_MAP)}")
    return dtype


def _load_cached(cls: type, checkpoint: str, **kwargs) -> object:
    """Load a HuggingFace model or processor from local cache if available.

    Tries local_files_only first to skip the Hub freshness-check network
    request. Falls back to a normal download on cache miss.
    """
    try:
        return cls.from_pretrained(checkpoint, local_files_only=True, **kwargs)
    except OSError:
        logger.info(f"Cache miss for '{checkpoint}', downloading...")
        return cls.from_pretrained(checkpoint, **kwargs)


class RtPoseDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.RT_POSE] = TrackerType.RT_POSE
    yolo_model: str = "yolov8n.pt"
    pose_estimation_checkpoint: str = "usyd-community/vitpose-plus-small"
    device: str | None = None
    dtype: str = "float32"
    compile_models: bool = False
    detection_threshold: float = 0.3
    yolo_imgsz: int = 640
    yolo_half: bool = False
    max_people: int = 1
    upsample_heatmap: bool = True


@dataclass
class RtPoseDetector(BaseDetector):
    config: RtPoseDetectorConfig
    _detector: object = field(repr=False, default=None)
    _pose_estimator: object = field(repr=False, default=None)
    _pose_estimator_processor: object = field(repr=False, default=None)
    _device: str = field(repr=False, default="cpu")
    _dtype: torch.dtype = field(repr=False, default=torch.float32)

    @classmethod
    def create(cls, config: RtPoseDetectorConfig | None = None) -> "RtPoseDetector":
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        from ultralytics import YOLO

        config = config or RtPoseDetectorConfig()
        device = config.device or _resolve_device()
        dtype = _resolve_dtype(config.dtype)

        logger.info(f"Loading YOLO detector '{config.yolo_model}'...")
        detector = YOLO(config.yolo_model)

        logger.info(f"Loading VitPose estimator from '{config.pose_estimation_checkpoint}'...")
        pose_estimator = _load_cached(
            VitPoseForPoseEstimation, config.pose_estimation_checkpoint, torch_dtype=dtype
        ).to(device)
        pose_estimator_processor = _load_cached(AutoProcessor, config.pose_estimation_checkpoint)

        if config.compile_models:
            logger.info("Applying torch.compile to VitPose...")
            pose_estimator = torch.compile(pose_estimator, mode="reduce-overhead", dynamic=True)

        return cls(
            config=config,
            tracked_object=RT_POSE_DEFINITION,
            _detector=detector,
            _pose_estimator=pose_estimator,
            _pose_estimator_processor=pose_estimator_processor,
            _device=device,
            _dtype=dtype,
        )

    def detect(self, frame_number: int, image: np.ndarray) -> RtPoseObservation:
        person_boxes = self._run_detection(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints_xy, scores = self._run_pose_estimation(rgb, person_boxes)
        return RtPoseObservation.from_detection_results(
            frame_number=frame_number,
            keypoints_xy=keypoints_xy,
            scores=scores,
            image_size=(image.shape[1], image.shape[0]),
        )

    def _run_detection(self, image: np.ndarray) -> torch.Tensor:
        results = self._detector.predict(
            image,
            classes=[0],  # person only
            conf=self.config.detection_threshold,
            device=self._device,
            imgsz=self.config.yolo_imgsz,
            half=self.config.yolo_half,
            max_det=self.config.max_people,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return torch.zeros((0, 4), device=self._device)
        return boxes.xyxy.to(self._device)

    def _run_pose_estimation(
        self, image: np.ndarray, person_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if person_boxes.shape[0] == 0:
            empty = torch.zeros((0, 17, 2), device=self._device)
            empty_scores = torch.zeros((0, 17), device=self._device)
            return empty, empty_scores

        proc = self._pose_estimator_processor
        crop_height = proc.size["height"]
        crop_width = proc.size["width"]

        inputs, preprocessed_boxes = preprocess(
            image=image,
            boxes_xyxy=person_boxes,
            mean=proc.image_mean,
            std=proc.image_std,
            crop_height=crop_height,
            crop_width=crop_width,
            dtype=self._dtype,
            device=self._device,
        )

        if self._pose_estimator.config.backbone_config.num_experts > 1:
            batch_size = person_boxes.shape[0]
            inputs["dataset_index"] = torch.full(
                (batch_size,), 0, dtype=torch.int64, device=self._device
            )

        with torch.no_grad():
            outputs = self._pose_estimator(**inputs)

        keypoints_xy, scores = post_process_pose_estimation(
            outputs.heatmaps,
            crop_height=crop_height,
            crop_width=crop_width,
            boxes_xyxy=preprocessed_boxes,
            upsample_heatmap=self.config.upsample_heatmap,
        )
        return keypoints_xy, scores
