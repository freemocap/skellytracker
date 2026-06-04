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


def _detection_device_for(device: str) -> str:
    # RT-DETR uses float64 internally for position embeddings, which MPS
    # doesn't support. Run detection on CPU; pose estimation stays on MPS.
    return "cpu" if device == "mps" else device


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {list(_DTYPE_MAP)}")
    return dtype


class RtPoseDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.RT_POSE] = TrackerType.RT_POSE
    object_detection_checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365"
    pose_estimation_checkpoint: str = "usyd-community/vitpose-plus-small"
    device: str | None = None
    dtype: str = "float32"
    compile_models: bool = True
    detection_threshold: float = 0.3


@dataclass
class RtPoseDetector(BaseDetector):
    config: RtPoseDetectorConfig
    _detector: object = field(repr=False, default=None)
    _detector_processor: object = field(repr=False, default=None)
    _pose_estimator: object = field(repr=False, default=None)
    _pose_estimator_processor: object = field(repr=False, default=None)
    _device: str = field(repr=False, default="cpu")           # pose estimation device
    _detection_device: str = field(repr=False, default="cpu") # RT-DETR device (CPU when _device=mps)
    _dtype: torch.dtype = field(repr=False, default=torch.float32)

    @classmethod
    def create(cls, config: RtPoseDetectorConfig | None = None) -> "RtPoseDetector":
        from transformers import AutoModelForObjectDetection, AutoProcessor, VitPoseForPoseEstimation

        config = config or RtPoseDetectorConfig()
        device = config.device or _resolve_device()
        detection_device = _detection_device_for(device)
        dtype = _resolve_dtype(config.dtype)

        if detection_device != device:
            logger.info(
                f"Mixed-device mode: RT-DETR on '{detection_device}', "
                f"VitPose on '{device}'"
            )

        # RT-DETR always loads on detection_device with float32 (its position
        # embedding uses float64 internally, which only CPU supports).
        logger.info(f"Loading RT-DETR detector from '{config.object_detection_checkpoint}'...")
        detector = AutoModelForObjectDetection.from_pretrained(
            config.object_detection_checkpoint, torch_dtype=torch.float32
        ).to(detection_device)
        detector_processor = AutoProcessor.from_pretrained(
            config.object_detection_checkpoint, use_fast=True
        )

        logger.info(f"Loading VitPose estimator from '{config.pose_estimation_checkpoint}'...")
        pose_estimator = VitPoseForPoseEstimation.from_pretrained(
            config.pose_estimation_checkpoint, torch_dtype=dtype
        ).to(device)
        pose_estimator_processor = AutoProcessor.from_pretrained(config.pose_estimation_checkpoint)

        if config.compile_models:
            logger.info("Applying torch.compile to models...")
            detector = torch.compile(detector, mode="reduce-overhead")
            pose_estimator = torch.compile(pose_estimator, mode="reduce-overhead", dynamic=True)

        instance = cls(
            config=config,
            tracked_object=RT_POSE_DEFINITION,
            _detector=detector,
            _detector_processor=detector_processor,
            _pose_estimator=pose_estimator,
            _pose_estimator_processor=pose_estimator_processor,
            _device=device,
            _detection_device=detection_device,
            _dtype=dtype,
        )
        return instance

    def detect(self, frame_number: int, image: np.ndarray) -> RtPoseObservation:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_f32 = rgb.astype(np.float32)

        detection_tensor = torch.from_numpy(rgb_f32).to(self._detection_device)
        person_boxes = self._run_detection(detection_tensor)

        # Move boxes to the pose device; rebuild image tensor there if needed.
        pose_boxes = person_boxes.to(self._device)
        if self._detection_device == self._device:
            pose_tensor = detection_tensor
        else:
            pose_tensor = torch.from_numpy(rgb_f32).to(self._device)

        keypoints_xy, scores = self._run_pose_estimation(pose_tensor, pose_boxes)

        return RtPoseObservation.from_detection_results(
            frame_number=frame_number,
            keypoints_xy=keypoints_xy,
            scores=scores,
            image_size=(image.shape[1], image.shape[0]),
        )

    def _run_detection(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self._detector_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._detection_device).to(torch.float32) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._detector(**inputs)

        height, width = image.shape[:2]
        results = self._detector_processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=self.config.detection_threshold
        )
        detections = results[0]
        person_boxes = detections["boxes"][detections["labels"] == 0]
        return person_boxes

    def _run_pose_estimation(
        self, image: torch.Tensor, person_boxes: torch.Tensor
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
        )
        return keypoints_xy, scores
