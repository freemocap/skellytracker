from pathlib import Path
from typing import Literal

import numpy as np
from easy_ViTPose import VitInference
from huggingface_hub import hf_hub_download

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector, BaseDetectorConfig, TrackerType
from skellytracker.trackers.vitpose_tracker.vitpose_observation import VITPoseObservation

HF_VIT_REPO = "JunkyByte/easy_ViTPose"
HF_YOLO_REPO = "ultralytics/YOLOv8"

VITPOSE_WHOLEBODY = {
    "s": "torch/wholebody/vitpose-s-wholebody.pth",
    "b": "torch/wholebody/vitpose-b-wholebody.pth",
    "l": "torch/wholebody/vitpose-l-wholebody.pth",
    "h": "torch/wholebody/vitpose-h-wholebody.pth",
}

YOLO = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "extralarge": "yolov8x.pt"
}

def resolve_vitpose_wholebody(name:str) -> Path:
    model_name = VITPOSE_WHOLEBODY.get(name)
    if model_name is None:
        raise ValueError(f"Unknown VITPose model name: {name}")
    return Path(hf_hub_download(repo_id = HF_VIT_REPO, filename = model_name))

def resolve_yolo_model(name:str) -> Path:
    model_name = YOLO.get(name)
    if model_name is None:
        raise ValueError(f"Unknown YOLO model name: {name}")
    return Path(hf_hub_download(repo_id = HF_YOLO_REPO, filename = model_name))

class VITPoseDetectorConfig(BaseDetectorConfig):
    tracker_type: Literal[TrackerType.VITPOSE] = TrackerType.VITPOSE
    confidence_threshold: float = 0.5
    vit_model: str = "huge" #options are 'small', 'base', 'large', and 'huge'
    yolo_model: str = "medium" #options are 'nano', 'small', "medium", "large", "extralarge"
    yolo_size: int = 640 #Size of the input image for YOLO model
    yolo_step: int = 1 #how often YOLO is applied (1 is every frame), when >1 the tracker will try to predict bboxs to increase performance speed
    device: str|None = None #options are 'cuda', 'mps', or 'cpu', but with None the inferencer will auto check in cuda -> mps -> cpu order

class VITPoseDetector(BaseDetector):
    config: VITPoseDetectorConfig
    detector: VitInference

    @classmethod
    def create(cls, config: VITPoseDetectorConfig | None = None):
        
        config = config or VITPoseDetectorConfig()

        vit_model_name = cls._resolve_vit_model_type(config.vit_model)

        vit_model_path = resolve_vitpose_wholebody(vit_model_name)
        yolo_model_path = resolve_yolo_model(config.yolo_model)

        detector = VitInference(
            model_name=str(vit_model_name),
            model = str(vit_model_path),
            yolo = str(yolo_model_path),
            yolo_size = config.yolo_size,
            is_video = True,
            device = config.device,
            single_pose = True, #keep this as true until we're ready to deal with multi-person tracking
            yolo_step = config.yolo_step
        )

        return cls(config=config, detector=detector)

    def detect(self, frame_number:int, image:np.ndarray):
        results = self.detector.inference(image)
        return VITPoseObservation.from_detection_results(
            frame_number = frame_number,
            results = results,
            image_size = (int(image.shape[1]), int(image.shape[0]))
        )

    @staticmethod
    def _resolve_vit_model_type(model):
        mappings = {"small": "s",
                    "base": "b",
                    "large": "l",
                    "huge": "h"}

        if model not in mappings:
            raise ValueError(
                f"Invalid VITPose model '{model}'. "
                f"Must be one of {list(mappings.keys())}"
            )
        
        return mappings[model]
