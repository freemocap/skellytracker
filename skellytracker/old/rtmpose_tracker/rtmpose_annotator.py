import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from skellytracker.trackers.rtmpose_tracker._skeleton_viz import draw_skeleton

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseImageAnnotatorConfig, BaseImageAnnotator
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation

logger = logging.getLogger(__name__)

# Bbox debug colors (BGR for OpenCV).
YOLOX_BBOX_COLOR = (0, 255, 0)       # green = YOLOX detection
TRACKING_BBOX_COLOR = (0, 165, 255)  # orange = tracking-predicted


@dataclass
class RTMPoseImageAnnotatorConfig(BaseImageAnnotatorConfig):
    # Minimum SIMCC softmax peak to draw a keypoint / skeleton connection.
    # MUST match the tracking system's visibility threshold
    # (``rtmpose_tracking_state._DEFAULT_KPT_VISIBILITY_THRESHOLD``) so the
    # drawn skeleton reflects exactly what the YOLOX-skip logic considers "visible".
    # 0.004 = top ~50% of keypoints on a typical frame.
    confidence_threshold: float = 0.004
    # When True, draws the person bounding box on annotated images.
    # Green = from YOLOX detector, orange = from tracking prediction.
    draw_debug_bbox: bool = False


@dataclass
class RTMPoseImageAnnotator(BaseImageAnnotator):
    config: RTMPoseImageAnnotatorConfig
    observations: list[RTMPoseObservation]

    @classmethod
    def create(cls, config: RTMPoseImageAnnotatorConfig | None = None) -> "RTMPoseImageAnnotator":
        if config is None:
            config = RTMPoseImageAnnotatorConfig()
        return cls(config=config, observations=[])

    def annotate_image(
        self,
        image: NDArray[np.uint8],
        observation: RTMPoseObservation | None = None,
    ) -> NDArray[np.uint8]:
        if observation is None or observation.keypoints.shape[0] == 0:
            return image
        annotated_image = draw_skeleton(
            img=image,
            keypoints=observation.keypoints,
            # RTMPose confidence scores are in arbitrary units based on heatmap
            # peak height. A threshold of 2.0 filters out weak detections better
            # than the default 0.5.
            kpt_thr=self.config.confidence_threshold,
            scores=observation.scores,
        )
        return annotated_image

    def annotate_image_from_keypoints_and_scores(
        self,
        image: NDArray[np.uint8],
        keypoints: NDArray[np.float32],
        scores: NDArray[np.float32],
    ) -> NDArray[np.uint8]:
        annotated_image = draw_skeleton(img=image, keypoints=keypoints, scores=scores)
        return annotated_image

    # ------------------------------------------------------------------
    # Debug bbox drawing
    # ------------------------------------------------------------------

    @staticmethod
    def draw_bbox_on_image(
        image: NDArray[np.uint8],
        bbox: NDArray,
        *,
        from_detector: bool = True,
        label: str | None = None,
        confidence: float | None = None,
        consecutive_skips: int | None = None,
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw a person bounding box rectangle on *image* (mutates in-place).

        Args:
            image: BGR uint8 image (mutated in-place).
            bbox: ``(4,)`` xyxy array in image pixel coords.
            from_detector: True → green (YOLOX), False → orange (tracking).
            label: Optional text label drawn above the bbox.
            confidence: Mean keypoint confidence [0, 1] — shown in stats line.
            consecutive_skips: Consecutive frames since last YOLOX — shown in stats line.
            thickness: Line thickness in pixels.
        """
        if bbox is None or len(bbox) == 0:
            return image
        # Handle (N,4) — take first row.
        if bbox.ndim == 2:
            bbox = bbox[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = YOLOX_BBOX_COLOR if from_detector else TRACKING_BBOX_COLOR

        # ---- Stats label ----
        if confidence is not None or consecutive_skips is not None:
            # Build compact stats line.
            parts: list[str] = []
            if from_detector:
                parts.append("YOLOX")
            else:
                parts.append("track")
            if confidence is not None:
                parts.append(f"conf:{confidence:.2f}")
            if consecutive_skips is not None:
                parts.append(f"skips:{consecutive_skips}")
            stats_text = "  ".join(parts)
            draw_text_with_background(
                image, stats_text,
                anchor_xy=(x1, y1 - 2),
                anchor_edge="bottom",
                text_color=color,
            )
        elif label is not None:
            draw_text_with_background(
                image, label,
                anchor_xy=(x1, y1 - 2),
                anchor_edge="bottom",
                text_color=color,
            )

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image


# ---------------------------------------------------------------------------
# Shared text-drawing helper — used by both the in-memory annotator and the
# freemocap debug-frame writer.
#
# Uses the "doubled text" technique: a thick dark outline drawn first,
# then the colored fill text on top.  This is readable on any background
# without the clutter of a filled rectangle.
# ---------------------------------------------------------------------------

_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.4
_LABEL_FILL_THICKNESS = 1
_LABEL_OUTLINE_THICKNESS = 3


def draw_text_with_background(
    image: NDArray[np.uint8],
    text: str,
    *,
    anchor_xy: tuple[int, int],
    anchor_edge: str,  # "top" | "bottom"
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw *text* with a dark outline (doubled-text technique), mutating *image* in-place.

    Args:
        image: BGR uint8 image.
        text: The text string to draw.
        anchor_xy: (x, y) pixel position of the anchor point.
        anchor_edge: ``"top"`` → text sits BELOW anchor (inside-top of bbox).
                     ``"bottom"`` → text sits ABOVE anchor (above-top of bbox).
        text_color: BGR fill color for the text.
        outline_color: BGR color for the outline stroke (default black).
    """
    h_img, w_img = image.shape[:2]
    ax, ay = anchor_xy

    (tw, th), baseline = cv2.getTextSize(
        text, _LABEL_FONT, _LABEL_FONT_SCALE, _LABEL_OUTLINE_THICKNESS,
    )

    if anchor_edge == "bottom":
        text_org = (ax, ay - baseline - 2)
    else:
        text_org = (ax, ay + th + 2)

    # Clamp to image bounds.
    tx = max(text_org[0], 0)
    ty = max(text_org[1], 0)
    ty = min(ty, h_img - baseline)

    # Outline pass — thick, dark.
    cv2.putText(
        image, text, (tx, ty),
        _LABEL_FONT, _LABEL_FONT_SCALE, outline_color,
        _LABEL_OUTLINE_THICKNESS, cv2.LINE_AA,
    )
    # Fill pass — thin, colored.
    cv2.putText(
        image, text, (tx, ty),
        _LABEL_FONT, _LABEL_FONT_SCALE, text_color,
        _LABEL_FILL_THICKNESS, cv2.LINE_AA,
    )
