from __future__ import annotations

from pydantic import BaseModel


class MultiPersonTrackingConfig(BaseModel):
    """Controls cross-frame identity assignment for multi-person tracking.

    Each frame's person detections are matched against existing tracks via a
    cost matrix blending bounding-box IoU and keypoint displacement, solved
    with the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    iou_weight, keypoint_weight:
        Blend weights for the association cost. Costs are each in [0, 1]
        (iou cost = 1 - IoU; keypoint cost = normalized mean displacement
        clipped to [0, 1]) before blending, so weights are directly comparable.
    max_match_cost:
        Gate: a track/detection pair costing more than this is never matched,
        regardless of what the Hungarian solver would otherwise assign.
    max_age:
        Number of consecutive frames a track may go unmatched before it is
        dropped.
    min_hits:
        Number of matched frames required before a track is "confirmed" and
        exposed to downstream consumers (suppresses one-frame false positives
        from the object detector).
    """

    iou_weight: float = 0.5
    keypoint_weight: float = 0.5
    max_match_cost: float = 0.8
    max_age: int = 10
    min_hits: int = 3
