"""Sidecar-driven DLC inference; no training framework or DLC runtime imports."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from skellytracker.core.config.detector_configs import KeypointDetectorConfig
from skellytracker.core.data_primitives import Keypoints
from skellytracker.core.detectors.detector_base_classes import (
    KEYPOINT_DETECTOR_REGISTRY,
    KeypointDetector,
)
from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.detectors.processing.coordinate_decode import decode_coordinates
from skellytracker.core.detectors.processing.image_preprocessing import (
    build_normalization_fn,
    preprocess_image,
)
from skellytracker.core.sessions.onnx_model_spec import OnnxModelSpec, OnnxSessionConfig
from skellytracker.core.sessions.session import Session
from skellytracker.core.sidecar.loader import load_sidecar
from skellytracker.core.sidecar.runtime import sidecar_model_spec


class DlcDetectorConfig(KeypointDetectorConfig):
    detector_type: Literal["dlc"] = "dlc"
    session_backend: Literal["onnx"] = "onnx"
    # The sidecar path is the model identity, allowing independent models in one session.
    model_name: str
    confidence_threshold: float = Field(default=0.4, ge=0, le=1)

    @field_validator("model_name")
    @classmethod
    def absolute_sidecar_path(cls, value: str) -> str:
        return str(Path(value).expanduser().resolve())


def _model_contract(model_name: str):
    path = Path(model_name)
    sidecar = load_sidecar(path, cache_dir=path.parent)
    resolved = sidecar.resolved_size("native")
    pose = sidecar.pose
    if (
        pose is None
        or pose.decode.method != "coordinate"
        or pose.decode.coordinate_scale != "pixel"
        or pose.decode.is_3d
    ):
        raise ValueError("DLC requires a pixel-coordinate pose sidecar")
    count = len(pose.tracked_points)
    if len(sidecar.outputs) != 1:
        raise ValueError("DLC requires one x/y/confidence output")
    output = sidecar.outputs[0]
    if (
        output.shape != [-1, count, 3]
        or output.fields != ["x", "y", "confidence"]
        or output.keypoint_axis != 1
        or output.keypoint_count != count
        or output.dtype.get("fp32") != "float32"
        or output.semantic != "keypoints"
    ):
        raise ValueError(
            "DLC output must declare float32 [batch, keypoints, x/y/confidence]"
        )
    image = resolved.input
    if (
        image.dtype.get("fp32") != "float32"
        or image.layout != "NCHW"
        or image.shape[:2] != [-1, 3]
        or image.normalization != "imagenet_rgb"
        or image.normalization_by_precision
        or image.resize is None
    ):
        raise ValueError(
            "DLC requires declared NCHW float32 ImageNet RGB preprocessing"
        )
    resize = image.resize
    if resize.supports_dynamic_size or resize.method not in (
        "none",
        "fit",
        "letterbox",
    ):
        raise ValueError("DLC supports native, fit or explicit letterbox resizing only")
    if resize.method != "none":
        if resize.target_size != tuple(image.shape[2:]) or resize.pad_value is None:
            raise ValueError(
                "DLC resize must declare the model target_size and pad_value"
            )
        if resize.method == "letterbox" and (
            resize.pad_position is None
            or resize.rounding is None
            or resize.preserve_aspect_ratio is False
        ):
            raise ValueError(
                "DLC letterbox requires explicit pad_position and rounding with preserved aspect ratio"
            )
    descriptor = sidecar_model_spec(
        sidecar,
        size="native",
        batch_key="dynamic",
        precision="fp32",
        name=model_name,
        sidecar_dir=path.parent,
    )
    if descriptor.cuda_use_tf32 is not False or not descriptor.expected_sha256:
        raise ValueError(
            "DLC artifact must declare its SHA-256 and cuda_use_tf32: false"
        )
    return sidecar, resolved, descriptor


@dataclass
class DlcDetector(KeypointDetector):
    config: DlcDetectorConfig
    session: Session
    _point_names: tuple[str, ...] = field(init=False)
    _template: Keypoints = field(init=False, repr=False)
    _normalize: Any = field(init=False, repr=False)
    _input_name: str = field(init=False)
    _input_size: tuple[int, int] = field(init=False)
    _connections: tuple = field(init=False)
    _input_spec: Any = field(init=False, repr=False)

    def __post_init__(self):
        sidecar, resolved, descriptor = _model_contract(self.config.model_name)
        self._point_names = tuple(sidecar.pose.tracked_points)
        self._template = Keypoints.empty(self._point_names)
        self._normalize = build_normalization_fn(resolved.input, "fp32")
        self._input_name = resolved.input.name
        self._input_size = descriptor.input_size
        self._input_spec = resolved.input
        self._connections = tuple(
            edge for skeleton in sidecar.pose.connections for edge in skeleton.edges
        )

    @classmethod
    def model_spec(cls, model_name: str) -> OnnxModelSpec:
        return _model_contract(model_name)[2]

    @classmethod
    def validate_session_config(cls, config: OnnxSessionConfig) -> None:
        if config.execution_provider not in ("cpu", "cuda") or config.fp16:
            raise ValueError(
                "DLC supports explicit CPU or CUDA FP32 only; set onnx_overrides "
                "to execution_provider='cpu' or 'cuda' and fp16=False"
            )

    @classmethod
    def create(cls, config: KeypointDetectorConfig, session):
        from skellytracker.core.sessions.onnx_session import OnnxSession

        if not isinstance(config, DlcDetectorConfig) or not isinstance(
            session, OnnxSession
        ):
            raise TypeError("DLC requires DlcDetectorConfig and OnnxSession")
        provenance = session.creation_config
        if provenance is None:
            raise ValueError(
                "DLC requires a session created through OnnxSession.create with recorded model provenance"
            )
        cls.validate_session_config(provenance)
        descriptor = cls.model_spec(config.model_name)
        if (
            descriptor not in provenance.models
            or session.execution_provider != provenance.execution_provider
        ):
            raise ValueError(
                "DLC session model descriptor or selected provider does not match its contract"
            )
        runtime = session.get_session(config.model_name)
        providers = runtime.get_providers()
        expected = (
            "CUDAExecutionProvider"
            if session.execution_provider == "cuda"
            else "CPUExecutionProvider"
        )
        if (
            not providers
            or providers[0] != expected
            or any(p not in (expected, "CPUExecutionProvider") for p in providers)
        ):
            raise ValueError(f"DLC requires {expected}, received {providers}")
        if (
            expected == "CUDAExecutionProvider"
            and runtime.get_provider_options().get(expected, {}).get("use_tf32") != "0"
        ):
            raise ValueError(
                "DLC CUDA session requires use_tf32=0; TF32 accuracy is not certified"
            )
        sidecar, resolved, _ = _model_contract(config.model_name)
        inputs, outputs = runtime.get_inputs(), runtime.get_outputs()
        count = len(sidecar.pose.tracked_points)
        if (
            len(inputs) != 1
            or inputs[0].name != resolved.input.name
            or inputs[0].type != "tensor(float)"
            or inputs[0].shape[1:] != [3, *descriptor.input_size]
            or len(outputs) != 1
            or outputs[0].name != sidecar.outputs[0].name
            or outputs[0].type != "tensor(float)"
            or outputs[0].shape[1:] != [count, 3]
        ):
            raise ValueError(
                "DLC runtime input/output metadata does not match the sidecar"
            )
        runtime.disable_fallback()
        return cls(config=config, session=session)

    def preprocess(self, image):
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) < 1
        ):
            raise ValueError("DLC expects a nonempty uint8 BGR image")
        if self._input_spec.resize.method != "none":
            return preprocess_image(
                image,
                self._input_size,
                self._input_spec,
                precision="fp32",
                normalization_fn=self._normalize,
            )
        if image.shape != (*self._input_size, 3):
            raise ValueError(
                f"DLC native preprocessing expects image {(*self._input_size, 3)}; export a matching model or explicitly declare fit resizing"
            )
        return np.ascontiguousarray(
            self._normalize(image).transpose(2, 0, 1)
        ), ImageTransform(1.0, 1.0)

    def postprocess(self, raw, metadata):
        if len(raw) != 1 or raw[0].shape[0] != 1:
            raise ValueError("DLC postprocess requires one image's coordinate output")
        xyz, scores = decode_coordinates(
            raw[0],
            metadata,
            keypoint_count=len(self._point_names),
            confidence_threshold=self.config.confidence_threshold,
        )
        return self._template.with_values(xyz=xyz[0], visibility=scores[0])

    def detect(self, image, context=None):
        tensor, metadata = self.preprocess(image)
        raw = self.session.run(self.config.model_name, {self._input_name: tensor[None]})
        return self.postprocess(raw, metadata)

    def connections(self):
        return self._connections


KEYPOINT_DETECTOR_REGISTRY["dlc"] = DlcDetector
