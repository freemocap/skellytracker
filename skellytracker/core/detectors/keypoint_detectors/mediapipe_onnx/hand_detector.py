"""Experimental palm -> aligned hand inference; never selected by the CPU preset."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import time

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KeypointDetector,
    KEYPOINT_DETECTOR_REGISTRY,
)
from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.detectors.processing.image_preprocessing import (
    build_normalization_fn,
    preprocess_image,
)
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig
from skellytracker.core.sessions.onnx_session import OnnxSession
from skellytracker.core.sessions.session import Session
from skellytracker.core.sidecar.runtime import sidecar_model_spec
from .candidate_models import candidate_sidecar, HAND_FIELDS
from .hand_geometry import HandAlignment, decode_single_palm, prepare_aligned_hand
from .model_preparation import prepare_detector_model


class OnnxHandDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["experimental_mediapipe_onnx_hand"] = (
        "experimental_mediapipe_onnx_hand"
    )
    session_backend: Literal["onnx"] = "onnx"
    assumed_handedness: Literal["left", "right"]
    min_hand_detection_confidence: float = Field(default=0.5, ge=0, le=1)
    min_hand_presence_confidence: float = Field(default=0.5, ge=0, le=1)
    model_directory: Path | None = None


@dataclass
class OnnxHandKeypointDetector(KeypointDetector):
    @staticmethod
    def validate_session_config(config: OnnxSessionConfig) -> None:
        if config.execution_provider not in ("cpu", "cuda"):
            raise ValueError("Experimental ONNX hands require an explicit CPU or CUDA provider; Auto and TensorRT are not supported")

    config: OnnxHandDetectorConfig
    session: OnnxSession
    _point_names: tuple[str, ...] = field(init=False)
    _schema: Keypoints = field(init=False, repr=False)
    _hand_input: Any = field(init=False, repr=False)
    _palm_input: Any = field(init=False, repr=False)
    _hand_normalize: Any = field(init=False, repr=False)
    _palm_normalize: Any = field(init=False, repr=False)
    _hand_size: tuple[int, int] = field(init=False)
    _palm_size: tuple[int, int] = field(init=False)
    _hand_name: str = field(init=False)
    _palm_name: str = field(init=False)
    _palm_output_shape: tuple[int, ...] = field(init=False)
    _presence_index: int = field(init=False)

    def __post_init__(self):
        hand, palm = candidate_sidecar("hand"), candidate_sidecar("palm")
        self._point_names = tuple(
            f"{self.config.assumed_handedness}_hand_{name}"
            for name in hand.pose.tracked_points
        )
        self._schema = Keypoints.empty(self._point_names)
        self._hand_input, self._palm_input = hand.input, palm.input
        self._hand_size = hand.input.resize.target_size
        self._palm_size = palm.input.resize.target_size
        self._hand_normalize = build_normalization_fn(hand.input, "fp32")
        self._palm_normalize = build_normalization_fn(palm.input, "fp32")
        self._hand_name, self._palm_name = hand.model_id, palm.model_id
        self._palm_output_shape = (1, *palm.outputs[0].shape[1:])
        self._presence_index = HAND_FIELDS.index("hand_presence")
        for sidecar in (hand, palm):
            native = self.session.get_session(sidecar.model_id)
            inputs, outputs = native.get_inputs(), native.get_outputs()
            expected = sidecar.outputs[0]
            if (
                len(inputs) != 1
                or inputs[0].name != sidecar.input.name
                or inputs[0].type != "tensor(float)"
                or inputs[0].shape[1:]
                != sidecar.resolved_size("default").input.shape[1:]
                or len(outputs) != 1
                or outputs[0].name != expected.name
                or outputs[0].type != "tensor(float)"
                or outputs[0].shape[1:] != expected.shape[1:]
            ):
                raise ValueError(
                    f"Prepared graph does not match sidecar: {sidecar.model_id}"
                )

    @classmethod
    def model_specs(cls, config: OnnxHandDetectorConfig) -> tuple[OnnxModelSpec, ...]:
        result = []
        for kind in ("hand", "palm"):
            sidecar = candidate_sidecar(kind)
            descriptor = sidecar_model_spec(
                sidecar,
                size="default",
                batch_key="dynamic",
                precision="fp32",
                name=sidecar.model_id,
                sidecar_dir=Path(__file__).parent,
                prepare=prepare_detector_model,
            )
            if config.model_directory is not None:
                descriptor = replace(
                    descriptor,
                    source=ModelSource(
                        local_path=str(
                            config.model_directory / descriptor.expected_filename
                        )
                    ),
                )
            result.append(descriptor)
        return tuple(result)

    @classmethod
    def create(
        cls, config: OnnxHandDetectorConfig, session: Session
    ) -> OnnxHandKeypointDetector:
        if not isinstance(session, OnnxSession):
            raise TypeError(
                "Experimental ONNX hands require a factory-owned OnnxSession"
            )
        return cls(config=config, session=session)

    def preprocess(self, image: NDArray[np.uint8]):
        return preprocess_image(
            image,
            self._palm_size,
            self._palm_input,
            normalization_fn=self._palm_normalize,
        )

    def postprocess(self, raw: Any, metadata: HandAlignment) -> Keypoints:
        if len(raw) != 1 or raw[0].shape != (1, len(self._point_names), len(HAND_FIELDS)):
            raise ValueError("Hand output does not match prepared sidecar columns")
        values = raw[0][0]
        presence = values[0, self._presence_index]
        if (
            not np.isfinite(presence)
            or presence < self.config.min_hand_presence_confidence
        ):
            return self._empty_result()
        xyz = metadata.restore_keypoints(values[:, :3])
        valid = np.isfinite(xyz).all(axis=-1)
        xyz[~valid] = np.nan
        # Match CPU task visibility: a detected point has visibility 1 when the
        # task supplies no per-keypoint score; hand presence is only a hand gate.
        return self._schema.with_values(xyz=xyz, visibility=valid.astype(np.float64))

    def _empty_result(self) -> Keypoints:
        return self._schema.with_values(
            xyz=np.full_like(self._schema.xyz, np.nan),
            visibility=np.zeros_like(self._schema.visibility),
        )

    def detect(
        self, image: NDArray[np.uint8], context: DetectionContext | None = None
    ) -> Keypoints:
        return self.detect_batch({"image": image}, context)["image"]

    def detect_batch(
        self,
        images: dict[str, NDArray[np.uint8]],
        context: DetectionContext | None = None,
        *,
        executor: Executor | None = None,
    ) -> dict[str, Keypoints]:
        # Camera dictionaries carry identity only; all numerical operations use
        # dense arrays. No camera state or private inference sessions are retained.
        valid_images = {camera: image for camera, image in images.items() if image.size}
        results = {
            camera: self._empty_result()
            for camera in images
            if camera not in valid_images
        }
        if not valid_images:
            return results

        def prepare_images(function, keys):
            if executor is None:
                return {camera: function(camera) for camera in keys}
            futures = {camera: executor.submit(function, camera) for camera in keys}
            return {camera: future.result() for camera, future in futures.items()}

        started = time.perf_counter()
        palm_inputs = prepare_images(
            lambda camera: self.preprocess(images[camera]), valid_images
        )
        palm_raw = self.session.run_batched(
            self._palm_name, {camera: pair[0] for camera, pair in palm_inputs.items()}
        )
        if context is not None and context.timings is not None:
            context.timings.stop(
                f"{self.config.assumed_handedness}_hand.palm_prepare_infer", started
            )

        def align_hand(camera):
            raw = palm_raw[camera]
            if len(raw) != 1 or raw[0].shape != self._palm_output_shape:
                raise ValueError("Palm output does not match prepared sidecar columns")
            transform = palm_inputs[camera][1]
            if not isinstance(transform, ImageTransform):
                raise TypeError("Palm preprocessing must preserve its image transform")
            palm = decode_single_palm(
                raw[0][0, :, :-1],
                raw[0][0, :, -1:],
                self._palm_size,
                transform,
                self.config.min_hand_detection_confidence,
            )
            return (
                None
                if palm is None
                else prepare_aligned_hand(
                    images[camera],
                    palm,
                    self._hand_input,
                    self._hand_size,
                    normalization_fn=self._hand_normalize,
                )
            )

        started = time.perf_counter()
        hand_inputs = prepare_images(align_hand, valid_images)
        hand_tensors = {
            camera: pair[0] for camera, pair in hand_inputs.items() if pair is not None
        }
        hand_raw = (
            self.session.run_batched(self._hand_name, hand_tensors)
            if hand_tensors
            else {}
        )
        if context is not None and context.timings is not None:
            context.timings.stop(
                f"{self.config.assumed_handedness}_hand.align_prepare_infer", started
            )
        for camera, pair in hand_inputs.items():
            results[camera] = (
                self._empty_result()
                if pair is None
                else self.postprocess(hand_raw[camera], pair[1])
            )
        return results


KEYPOINT_DETECTOR_REGISTRY["experimental_mediapipe_onnx_hand"] = (
    OnnxHandKeypointDetector
)
