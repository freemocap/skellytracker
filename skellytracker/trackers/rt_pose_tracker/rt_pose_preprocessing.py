"""
Fast preprocessing and postprocessing for VitPose models.

Adapted from https://github.com/qubvel/rt-pose/blob/main/rt_pose/processing.py
(Apache-2.0 license).
"""

from typing import Dict, Tuple

import cv2
import numpy as np
import torch


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
    image: np.ndarray,
    boxes_xyxy: torch.Tensor,
    crop_height: int = 256,
    crop_width: int = 192,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    scale: float = 1 / 255.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Crop and normalize image regions for VitPose inference.

    Crops are extracted on CPU with cv2.resize and only the small crop batch
    (N × 3 × crop_height × crop_width) is transferred to device — avoiding a
    full-resolution GPU image transfer.

    Args:
        image: RGB image array (H, W, C) uint8.
        boxes_xyxy: Bounding boxes in xyxy format (any device).
        crop_height: Target crop height.
        crop_width: Target crop width.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        scale: Pixel value scale factor (1/255 to convert uint8 range).
        dtype: Output tensor dtype.
        device: Target device for the output crops tensor.

    Returns:
        Tuple of (model_inputs dict with "pixel_values", preprocessed boxes on device).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("image must be a (H, W, C) uint8 numpy array")

    img_h, img_w = image.shape[:2]

    # Aspect-ratio-adjusted boxes (may extend outside image bounds)
    boxes_float = preprocess_boxes(boxes_xyxy.cpu().float(), crop_height, crop_width)
    boxes_int = boxes_float.round().int()

    crops: list[np.ndarray] = []
    for box in boxes_int:
        x1, y1, x2, y2 = box.tolist()
        # Pad when box extends outside image (matches roi_align zero-pad behavior)
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - img_w)
        pad_b = max(0, y2 - img_h)
        patch = image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        if pad_l or pad_t or pad_r or pad_b:
            patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
        if patch.size == 0:
            patch = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
        else:
            patch = cv2.resize(patch, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
        crops.append(patch)

    # (N, H, W, C) -> (N, C, H, W) float32; astype() produces a contiguous copy
    crops_np = np.stack(crops).transpose(0, 3, 1, 2).astype(np.float32)
    crops_tensor = torch.from_numpy(crops_np).to(device).to(dtype)

    mean_t = torch.tensor(mean, dtype=dtype, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=dtype, device=device).view(1, 3, 1, 1)
    crops_tensor = (crops_tensor * scale - mean_t) / std_t

    return {"pixel_values": crops_tensor}, boxes_float.to(device)


def post_process_pose_estimation(
    heatmaps: torch.Tensor,
    crop_height: int,
    crop_width: int,
    boxes_xyxy: torch.Tensor,
    upsample_heatmap: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert VitPose heatmaps to keypoint coordinates and confidence scores.

    Args:
        heatmaps: Model heatmaps (batch_size, num_keypoints, H, W).
        crop_height: Height of the crop passed to the pose model.
        crop_width: Width of the crop passed to the pose model.
        boxes_xyxy: Crop bounding boxes in xyxy format.
        upsample_heatmap: If True (default), bilinear-upsample heatmaps to
            crop resolution before argmax for sub-pixel accuracy. If False,
            take argmax on the raw heatmap and scale coordinates — faster but
            quantised to the heatmap grid (~4px steps for a 256×192 crop).

    Returns:
        Tuple of (keypoints_xy, scores) where keypoints_xy is (N, K, 2) and
        scores is (N, K).
    """
    batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape

    if upsample_heatmap:
        heatmaps = torch.nn.functional.interpolate(
            heatmaps, size=(crop_height, crop_width), mode="bilinear", align_corners=True
        )
        grid_h, grid_w = crop_height, crop_width
    else:
        grid_h, grid_w = heatmap_h, heatmap_w

    flattened = heatmaps.reshape(batch_size, num_keypoints, -1)
    scores, indices = torch.max(flattened, dim=-1)

    keypoints_x = indices % grid_w
    keypoints_y = indices // grid_w

    box_x1, box_y1, box_x2, box_y2 = boxes_xyxy.split(1, dim=-1)
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    keypoints_x = keypoints_x.float() * box_width / grid_w + box_x1
    keypoints_y = keypoints_y.float() * box_height / grid_h + box_y1

    keypoints_xy = torch.stack([keypoints_x, keypoints_y], dim=-1)

    return keypoints_xy, scores
