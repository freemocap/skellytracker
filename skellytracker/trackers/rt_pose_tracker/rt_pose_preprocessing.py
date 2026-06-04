"""
Fast preprocessing and postprocessing for VitPose models.

Adapted from https://github.com/qubvel/rt-pose/blob/main/rt_pose/processing.py
(Apache-2.0 license).
"""

from typing import Dict, Tuple

import torch
import torchvision.ops


def preprocess_boxes(
    boxes_xyxy: torch.Tensor,
    crop_height: int = 256,
    crop_width: int = 192,
    padding_factor: float = 1.25,
) -> torch.Tensor:
    """
    Align box aspect ratio to the crop dimensions, then expand by padding_factor.

    Args:
        boxes_xyxy: Bounding boxes in (x_min, y_min, x_max, y_max) format.
        crop_height: Height of the target crop.
        crop_width: Width of the target crop.
        padding_factor: Factor to expand the box by.

    Returns:
        Processed boxes in xyxy format.
    """
    aspect_ratio = crop_width / crop_height
    x_min, y_min, x_max, y_max = boxes_xyxy.T

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    height = torch.where(width > height * aspect_ratio, width * 1 / aspect_ratio, height)
    width = torch.where(width < height * aspect_ratio, height * aspect_ratio, width)

    height = height * padding_factor
    width = width * padding_factor

    x_min = x_center - width / 2
    y_min = y_center - height / 2

    return torch.stack([x_min, y_min, x_min + width, y_min + height], dim=1)


def preprocess(
    image: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    crop_height: int = 256,
    crop_width: int = 192,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    scale: float = 1 / 255.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Crop and normalize image regions for VitPose inference.

    Args:
        image: RGB image tensor (H, W, C).
        boxes_xyxy: Bounding boxes in xyxy format.
        crop_height: Target crop height.
        crop_width: Target crop width.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        scale: Pixel value scale factor (1/255 to convert uint8 range).
        dtype: Output tensor dtype.

    Returns:
        Tuple of (model_inputs dict with "pixel_values", preprocessed boxes).
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("Image must be a torch.Tensor")
    if image.ndim != 3:
        raise ValueError("Image must be a 3D tensor with shape (H, W, C)")

    boxes_xyxy = preprocess_boxes(boxes_xyxy, crop_height, crop_width)
    boxes_xyxy = boxes_xyxy.round().int()

    image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW

    image = image.to(torch.float32)
    boxes_xyxy = boxes_xyxy.to(torch.float32)

    crops = torchvision.ops.roi_align(image, [boxes_xyxy], (crop_height, crop_width), 1)
    crops = crops.to(dtype)

    mean_tensor = torch.tensor(mean, dtype=crops.dtype, device=crops.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=crops.dtype, device=crops.device).view(1, 3, 1, 1)
    crops = (crops * scale - mean_tensor) / std_tensor

    return {"pixel_values": crops}, boxes_xyxy


def post_process_pose_estimation(
    heatmaps: torch.Tensor,
    crop_height: int,
    crop_width: int,
    boxes_xyxy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert VitPose heatmaps to keypoint coordinates and confidence scores.

    Args:
        heatmaps: Model heatmaps (batch_size, num_keypoints, H, W).
        crop_height: Height of the crop passed to the pose model.
        crop_width: Width of the crop passed to the pose model.
        boxes_xyxy: Crop bounding boxes in xyxy format.

    Returns:
        Tuple of (keypoints_xy, scores) where keypoints_xy is (N, K, 2) and
        scores is (N, K).
    """
    batch_size, num_keypoints, _, _ = heatmaps.shape

    heatmaps = torch.nn.functional.interpolate(
        heatmaps, size=(crop_height, crop_width), mode="bilinear", align_corners=True
    )

    flattened = heatmaps.reshape(batch_size, num_keypoints, -1)
    scores, indices = torch.max(flattened, dim=-1)

    keypoints_x = indices % crop_width
    keypoints_y = indices // crop_width

    box_x1, box_y1, box_x2, box_y2 = boxes_xyxy.split(1, dim=-1)
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    keypoints_x = keypoints_x.float() * box_width / crop_width + box_x1
    keypoints_y = keypoints_y.float() * box_height / crop_height + box_y1

    keypoints_xy = torch.stack([keypoints_x, keypoints_y], dim=-1)

    return keypoints_xy, scores
