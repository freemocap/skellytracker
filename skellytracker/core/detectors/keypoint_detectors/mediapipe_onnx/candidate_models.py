"""Validated sidecar descriptions for the pinned two-network hand candidate.

Use the existing MediaPipe hand schema rather than authoring a second list.
The descriptions refer to prepared graphs; URLs/hashes identify raw artifacts.
Model preparation is part of this detector's contract [@opencvZooHand2023;
@opencvZooPalm2023], not a generic session transformation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from skellytracker import __version__
from skellytracker.core.detectors.keypoint_detectors._schema_loader import (
    load_point_names,
    load_connections,
)
from skellytracker.core.sidecar.model import SidecarModel
from .hand_geometry import PALM_KEYPOINT_NAMES
from .model_preparation import CANDIDATE_HASHES

HAND_SCHEMA = Path(__file__).parents[1] / "mediapipe/hands/mediapipe_hand.yaml"
SOURCE_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
HAND_FIELDS = (
    "x",
    "y",
    "z",
    "hand_presence",
    "handedness",
    "world_x",
    "world_y",
    "world_z",
)
PALM_FIELDS = (
    ("center_x_offset", "center_y_offset", "width", "height")
    + tuple(
        f"{name}_{axis}_offset" for name in PALM_KEYPOINT_NAMES for axis in ("x", "y")
    )
    + ("score_logit",)
)


@lru_cache(maxsize=2)
def candidate_sidecar(kind: Literal["hand", "palm"]) -> SidecarModel:
    hand = kind == "hand"
    model_directory = (
        "handpose_estimation_mediapipe" if hand else "palm_detection_mediapipe"
    )
    filename = model_directory + "_2023feb.onnx"
    spatial_size = 224 if hand else 192
    names = load_point_names(HAND_SCHEMA)
    fields = HAND_FIELDS if hand else PALM_FIELDS
    values = dict(
        schema_version=__version__,
        model_id=f"experimental_mediapipe_{kind}",
        display_name=f"Experimental OpenCV Zoo MediaPipe {kind}",
        role=["pose_estimator"] if hand else ["object_detector"],
        input=dict(
            name="input_1",
            dtype={"fp32": "float32"},
            layout="NCHW",
            normalization=dict(mode="custom", color_format="rgb", scale=1 / 255),
            resize=(
                dict(
                    method="stretch",
                    target_size=(spatial_size, spatial_size),
                    interpolation="area",
                )
                if hand
                else dict(
                    method="letterbox",
                    target_size=(spatial_size, spatial_size),
                    pad_position="center",
                    pad_value=0,
                )
            ),
        ),
        outputs=[
            dict(
                name="hand_predictions" if hand else "palm_predictions",
                dtype={"fp32": "float32"},
                shape=[-1, len(names) if hand else 2016, len(fields)],
                fields=list(fields),
                semantic="keypoints" if hand else "detections",
                **(dict(keypoint_axis=1, keypoint_count=len(names)) if hand else {}),
            )
        ],
        sizes={
            "default": dict(
                input=dict(shape=[-1, 3, spatial_size, spatial_size]),
                onnx=dict(
                    batch_artifacts={
                        "dynamic": dict(
                            precision_artifacts={
                                "fp32": dict(
                                    filename=filename,
                                    url=f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{SOURCE_REVISION}/models/{model_directory}/{filename}",
                                    url_sha256=CANDIDATE_HASHES[filename],
                                )
                            }
                        )
                    }
                ),
            )
        },
    )
    if hand:
        values["pose"] = dict(
            estimator_type="top_down_single_person",
            requires_object_detector=True,
            tracked_points=list(names),
            connections=[
                dict(name="hand_connections", edges=load_connections(HAND_SCHEMA))
            ],
            decode=dict(
                method="coordinate",
                coordinate_scale="pixel",
                is_3d=True,
                depth_unit="pixel",
            ),
        )
    else:
        # Offsets require the model-owned SSD anchors before they are image boxes.
        values["decode"] = dict(
            score_field="score_logit",
            max_detections=1,
            requires_nms=False,
            may_include_non_person_classes=True,
            confidence_threshold_default=0.5,
        )
    return SidecarModel.model_validate(values)
