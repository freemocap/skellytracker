from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from skellytracker.core.data_primitives.bounding_box import BoundingBox
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.temporal_processing.multi_person_config import (
    MultiPersonTrackingConfig,
)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two boxes; 0.0 if they don't overlap."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_cost_matrix(
    track_bboxes: list[BoundingBox | None],
    det_bboxes: list[BoundingBox],
) -> NDArray[np.float64]:
    """(n_tracks, n_detections) cost matrix: 1 - IoU. inf where a track has no bbox."""
    n_tracks = len(track_bboxes)
    n_dets = len(det_bboxes)
    cost = np.full((n_tracks, n_dets), np.inf, dtype=np.float64)
    for i, tb in enumerate(track_bboxes):
        if tb is None:
            continue
        for j, db in enumerate(det_bboxes):
            cost[i, j] = 1.0 - iou(tb, db)
    return cost


def keypoint_distance_cost_matrix(
    track_keypoints: list[Keypoints | None],
    det_keypoints: list[Keypoints],
    *,
    confidence_threshold: float = 0.3,
    normalization_px: float = 200.0,
) -> NDArray[np.float64]:
    """(n_tracks, n_detections) cost matrix from mean keypoint displacement.

    For each pair, compares points with matching names visible (confidence
    above threshold) on both sides, averages the pixel distance, and clips
    the result to [0, 1] after dividing by normalization_px. Pairs sharing
    no visible points (or where a track has no prior keypoints) get inf.
    """
    n_tracks = len(track_keypoints)
    n_dets = len(det_keypoints)
    cost = np.full((n_tracks, n_dets), np.inf, dtype=np.float64)
    for i, tk in enumerate(track_keypoints):
        if tk is None:
            continue
        t_mask = tk.visibility >= confidence_threshold
        if not t_mask.any():
            continue
        t_names = {name for name, valid in zip(tk.names, t_mask, strict=False) if valid}
        for j, dk in enumerate(det_keypoints):
            d_mask = dk.visibility >= confidence_threshold
            shared = tuple(
                name for name, valid in zip(dk.names, d_mask, strict=False)
                if valid and name in t_names
            )
            if not shared:
                continue
            t_xy = tk.slice_by_names(shared).xy
            d_xy = dk.slice_by_names(shared).xy
            dist = float(np.linalg.norm(t_xy - d_xy, axis=1).mean())
            cost[i, j] = min(dist / normalization_px, 1.0)
    return cost


def combined_cost_matrix(
    track_bboxes: list[BoundingBox | None],
    track_keypoints: list[Keypoints | None],
    det_bboxes: list[BoundingBox],
    det_keypoints: list[Keypoints],
    config: MultiPersonTrackingConfig,
) -> NDArray[np.float64]:
    """Weighted blend of IoU cost and keypoint-distance cost.

    A pair is only usable if at least one of the two signals is finite; if
    exactly one side is inf, the finite side alone is used (renormalized by
    its own weight) rather than propagating inf, so a new track with no
    keypoint history yet can still match on IoU alone.
    """
    iou_cost = iou_cost_matrix(track_bboxes, det_bboxes)
    kp_cost = keypoint_distance_cost_matrix(track_keypoints, det_keypoints)

    iou_finite = np.isfinite(iou_cost)
    kp_finite = np.isfinite(kp_cost)
    both = iou_finite & kp_finite
    only_iou = iou_finite & ~kp_finite
    only_kp = ~iou_finite & kp_finite

    total_weight = config.iou_weight + config.keypoint_weight
    combined = np.full_like(iou_cost, np.inf)
    combined[both] = (
        config.iou_weight * iou_cost[both] + config.keypoint_weight * kp_cost[both]
    ) / total_weight
    combined[only_iou] = iou_cost[only_iou]
    combined[only_kp] = kp_cost[only_kp]
    return combined


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def associate(
    track_bboxes: list[BoundingBox | None],
    track_keypoints: list[Keypoints | None],
    det_bboxes: list[BoundingBox],
    det_keypoints: list[Keypoints],
    config: MultiPersonTrackingConfig,
) -> AssociationResult:
    """Match existing tracks to this frame's detections via the Hungarian algorithm.

    Pairs costing more than config.max_match_cost are rejected even if the
    solver would otherwise assign them, and fall through to unmatched.
    """
    n_tracks = len(track_bboxes)
    n_dets = len(det_bboxes)

    if n_tracks == 0 or n_dets == 0:
        return AssociationResult(
            matches=[],
            unmatched_tracks=list(range(n_tracks)),
            unmatched_detections=list(range(n_dets)),
        )

    cost = combined_cost_matrix(track_bboxes, track_keypoints, det_bboxes, det_keypoints, config)

    # linear_sum_assignment can't handle inf; replace with a large finite
    # sentinel so unmatchable pairs are simply never chosen when a cheaper
    # option exists, then gate them out below regardless.
    finite_cost = np.where(np.isfinite(cost), cost, 1e6)
    track_idx, det_idx = linear_sum_assignment(finite_cost)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for t, d in zip(track_idx, det_idx, strict=False):
        if cost[t, d] <= config.max_match_cost:
            matches.append((int(t), int(d)))
            matched_tracks.add(int(t))
            matched_dets.add(int(d))

    unmatched_tracks = [t for t in range(n_tracks) if t not in matched_tracks]
    unmatched_detections = [d for d in range(n_dets) if d not in matched_dets]
    return AssociationResult(
        matches=matches,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
    )
