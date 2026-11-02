"""Independent inference reconstruction of the explicitly supported DLC family.

No DLC modules are imported. Architecture/decoder provenance:
[@dlc302ResnetBackbone; @dlc302SimpleHead; @dlc302SinglePredictor].
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from torch import nn


@dataclass(frozen=True)
class ModelContract:
    keypoint_names: tuple[str, ...]
    backbone_stride: int
    head_stride: int
    location_scale: float
    apply_sigmoid: bool
    clip_scores: bool

    @property
    def output_stride(self) -> float:
        return self.backbone_stride / self.head_stride


def read_contract(config: dict) -> ModelContract:
    """Reject unsupported inference behavior rather than guessing defaults."""
    if config.get("detector") is not None:
        raise ValueError("A separate detector pipeline is unsupported")
    inference = config.get("inference") or {}
    known_inference_settings = {
        "multithreading",
        "compile",
        "autocast",
        "conditions",
        "snapshot",
        "eval",
        "output_dir",
    }
    if set(inference) - known_inference_settings:
        raise ValueError("Unsupported inference pipeline settings")
    if inference.get("conditions") is not None:
        raise ValueError("Conditional inference is unsupported")
    if any(
        (inference.get(name) or {}).get("enabled", False)
        for name in ("compile", "autocast")
    ):
        raise ValueError(
            "Compiled or autocast inference must be disabled for FP32 conversion"
        )
    model = config["model"]
    if set(model) != {
        "backbone",
        "heads",
        "backbone_output_channels",
        "neck",
        "pose_model",
    }:
        raise ValueError("Unsupported model configuration fields")
    backbone = model["backbone"]
    if backbone != {
        "type": "ResNet",
        "model_name": "resnet50_gn",
        "output_stride": 16,
        "freeze_bn_stats": False,
        "freeze_bn_weights": False,
    }:
        raise ValueError(
            "Only the verified ResNet50 GroupNorm stride16 backbone is supported"
        )
    if model.get("neck") is not None or model.get("pose_model") is not None:
        raise ValueError("Neck and alternate pose models are unsupported")
    if model.get("backbone_output_channels") != 2048 or set(model["heads"]) != {
        "bodypart"
    }:
        raise ValueError("Expected one bodypart head with 2048 backbone channels")
    names = tuple(config["metadata"]["bodyparts"])
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Keypoint names must be nonempty and unique")
    if config["method"] != "bu" or config["metadata"].get("unique_bodyparts"):
        raise ValueError("Only the single-subject bottom-up model is supported")
    if len(config["metadata"]["individuals"]) != 1:
        raise ValueError("Only a single subject is supported")
    if config["metadata"].get("with_identity") not in (False, None):
        raise ValueError("Identity heads are unsupported")
    head = model["heads"]["bodypart"]
    inference_head_keys = {"type", "predictor", "heatmap_config", "locref_config"}
    training_head_keys = {"weight_init", "target_generator", "criterion", "aggregator"}
    if set(head) - inference_head_keys - training_head_keys:
        raise ValueError("Unsupported head configuration fields")
    if head["type"] != "HeatmapHead":
        raise ValueError("Expected a heatmap head")
    for field, channels in (
        ("heatmap_config", len(names)),
        ("locref_config", 2 * len(names)),
    ):
        if head[field] != {
            "channels": [2048, channels],
            "kernel_size": [3],
            "strides": [2],
        }:
            raise ValueError(
                f"Unsupported {field}; expected one stride2/kernel3 transpose convolution"
            )
    predictor = head["predictor"]
    if set(predictor) != {
        "type",
        "apply_sigmoid",
        "clip_scores",
        "location_refinement",
        "locref_std",
    }:
        raise ValueError("Unsupported predictor settings")
    if (
        predictor["type"] != "HeatmapPredictor"
        or predictor["location_refinement"] is not True
    ):
        raise ValueError("Expected heatmap prediction with location refinement")
    if (
        not isinstance(predictor["apply_sigmoid"], bool)
        or predictor["clip_scores"] is not True
    ):
        raise ValueError("Expected explicit sigmoid flag and clipped confidence")
    location_scale = float(predictor["locref_std"])
    if not np.isfinite(location_scale) or location_scale <= 0:
        raise ValueError("Location refinement scale must be finite and positive")
    data = config["data"]
    if (
        data["colormode"] != "RGB"
        or data["inference"].get("normalize_images") is not True
    ):
        raise ValueError("Only ImageNet-normalized RGB input is supported")
    for name, setting in data["inference"].items():
        if name != "normalize_images" and setting is not None and setting is not False:
            raise ValueError(f"Unsupported inference transform: {name}")
    return ModelContract(names, 16, 2, location_scale, predictor["apply_sigmoid"], True)


def decode_heatmaps(
    heatmaps: torch.Tensor, location_refinements: torch.Tensor, contract: ModelContract
) -> torch.Tensor:
    """Vectorized row-major maximum and interleaved x/y refinement selection."""
    batch_count, keypoint_count, height, width = heatmaps.shape
    scores = torch.sigmoid(heatmaps) if contract.apply_sigmoid else heatmaps
    flat_scores = scores.flatten(start_dim=2)
    indices = torch.argmax(flat_scores, dim=-1, keepdim=True)
    confidence = torch.gather(flat_scores, dim=-1, index=indices).squeeze(-1)
    offsets = location_refinements.reshape(
        batch_count, keypoint_count, 2, height * width
    )
    selected_offsets = torch.gather(
        offsets, -1, indices.unsqueeze(2).expand(-1, -1, 2, -1)
    ).squeeze(-1)
    peak_index = indices.squeeze(-1)
    pixel_x = (peak_index % width).to(
        heatmaps.dtype
    ) * contract.output_stride + 0.5 * contract.output_stride
    pixel_y = (peak_index // width).to(
        heatmaps.dtype
    ) * contract.output_stride + 0.5 * contract.output_stride
    coordinates = (
        torch.stack((pixel_x, pixel_y), dim=-1)
        + selected_offsets * contract.location_scale
    )
    if contract.clip_scores:
        confidence = confidence.clamp(0, 1)
    return torch.cat((coordinates, confidence.unsqueeze(-1)), dim=-1)


class ReconstructedPoseModel(nn.Module):
    """Timm backbone plus independently assembled inference-only heads."""

    def __init__(self, contract: ModelContract):
        super().__init__()
        self.contract = contract
        self.backbone = timm.create_model(
            "resnet50_gn", output_stride=contract.backbone_stride, pretrained=False
        )
        self.backbone.fc = nn.Identity()
        self.heatmap_head = nn.ConvTranspose2d(
            2048, len(contract.keypoint_names), kernel_size=3, stride=2
        )
        self.location_head = nn.ConvTranspose2d(
            2048, 2 * len(contract.keypoint_names), kernel_size=3, stride=2
        )

    def raw_outputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(inputs)
        return self.heatmap_head(features), self.location_head(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        heatmaps, offsets = self.raw_outputs(inputs)
        return decode_heatmaps(heatmaps, offsets, self.contract)


def load_model(config_path: Path, checkpoint_path: Path) -> ReconstructedPoseModel:
    contract = read_contract(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    prefixes = {
        "backbone.model.": "backbone.",
        "heads.bodypart.heatmap_head.deconv_layers.0.": "heatmap_head.",
        "heads.bodypart.locref_head.deconv_layers.0.": "location_head.",
    }
    mapped_state = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"Expected FP32 inference tensor: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
        for source, destination in prefixes.items():
            if name.startswith(source):
                mapped_state[destination + name[len(source) :]] = tensor
                break
        else:
            raise ValueError(f"Unsupported checkpoint tensor: {name}")
    model = ReconstructedPoseModel(contract)
    model.load_state_dict(mapped_state, strict=True)
    return model.cpu().eval()
