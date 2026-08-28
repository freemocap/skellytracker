"""Pydantic `SidecarModel` hierarchy mirroring specs/sidecar-spec.md.

This is the single source of runtime truth for what a sidecar YAML file must
contain; the spec document is the canonical human-readable description and
must stay aligned with this module (see spec, "Single source of truth").

Composition (`$ref`/`base`) is resolved before any of these models see the
data — see `resolution.py`/`loader.py`. `sizes` is intentionally kept as raw
dicts on `SidecarModel` and validated lazily per-size via `resolved_size()`,
since size-specific fields (`input.shape`, `resize.target_size`) are legally
absent at the top level.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from skellytracker.core.io.canonical_mapping_expansion import expand_prefixed_mapping
from skellytracker.core.sidecar.resolution import _deep_merge

Precision = Literal["fp32", "fp16", "int8"]
OnnxDtype = Literal["float32", "float16", "uint8", "int8"]
Role = Literal["object_detector", "pose_estimator"]
NamedNormalizationMode = Literal["none", "unit_float", "imagenet_bgr", "imagenet_rgb"]
OutputSemantic = Literal["detections", "simcc_x", "simcc_y", "simcc_z", "heatmap", "keypoints", "poses"]

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class CustomNormalization(_StrictModel):
    mode: Literal["custom"]
    color_format: Literal["rgb", "bgr"] = "rgb"
    scale: float = 1.0
    mean: list[float] | None = None
    std: list[float] | None = None

    @model_validator(mode="after")
    def _check_lengths(self) -> "CustomNormalization":
        if self.mean is not None and len(self.mean) != 3:
            raise ValueError(f"normalization.mean must have length 3, got {len(self.mean)}")
        if self.std is not None and len(self.std) != 3:
            raise ValueError(f"normalization.std must have length 3, got {len(self.std)}")
        if self.scale <= 0:
            raise ValueError(f"normalization.scale must be positive, got {self.scale}")
        return self


NormalizationMode = Annotated[Union[NamedNormalizationMode, CustomNormalization], Field(union_mode="left_to_right")]


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class CropPolicySpec(_StrictModel):
    expand_ratio: float = 1.25
    maintain_aspect_ratio: bool = True

    @model_validator(mode="after")
    def _check_expand_ratio(self) -> "CropPolicySpec":
        if self.expand_ratio <= 0:
            raise ValueError(f"crop_policy.expand_ratio must be positive, got {self.expand_ratio}")
        return self


class ResizeSpec(_StrictModel):
    method: Literal["letterbox", "affine_person_crop", "none"]
    target_size: tuple[int, int] | None = None
    supports_dynamic_size: bool = False
    preserve_aspect_ratio: bool | None = None
    pad_value: int | None = None
    interpolation: Literal["linear", "area", "cubic", "nearest"] = "linear"
    crop_policy: CropPolicySpec | None = None

    @model_validator(mode="after")
    def _check_method_specific_fields(self) -> "ResizeSpec":
        if self.crop_policy is not None and self.method != "affine_person_crop":
            raise ValueError("resize.crop_policy is only valid when method: affine_person_crop")
        if (self.pad_value is not None or self.preserve_aspect_ratio is not None) and self.method != "letterbox":
            raise ValueError("resize.pad_value/preserve_aspect_ratio are only valid when method: letterbox")
        if self.method == "none" and self.target_size is not None:
            raise ValueError("resize.target_size must be omitted when method: none")
        return self


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class InputSpec(_StrictModel):
    name: str
    dtype: dict[Precision, OnnxDtype]
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    normalization: NormalizationMode = "imagenet_bgr"
    normalization_by_precision: dict[Precision, NamedNormalizationMode] | None = None
    resize: ResizeSpec | None = None
    shape: list[int] | None = None  # only required on the per-size resolved model

    @model_validator(mode="after")
    def _check_dtype_nonempty(self) -> "InputSpec":
        if not self.dtype:
            raise ValueError("input.dtype must declare at least one precision")
        return self


class SizedInputSpec(InputSpec):
    """`input` after a size has been merged in — `shape` is required."""

    shape: list[int]

    @model_validator(mode="after")
    def _check_shape(self) -> "SizedInputSpec":
        if self.shape[0] != -1:
            raise ValueError(f"input.shape batch axis (index 0) must be -1, got {self.shape[0]}")
        if self.resize is not None and self.resize.target_size is not None:
            target_size = self.resize.target_size
            spatial = self.shape[2:4] if self.layout == "NCHW" else self.shape[1:3]
            if list(spatial) != list(target_size):
                raise ValueError(
                    f"input.shape spatial dims {spatial} must equal resize.target_size {target_size} "
                    f"for layout {self.layout}"
                )
        if self.resize is not None:
            needs_target = self.resize.method in ("letterbox", "affine_person_crop") and not self.resize.supports_dynamic_size
            if needs_target and self.resize.target_size is None:
                raise ValueError(
                    f"input.resize.target_size is required when method={self.resize.method!r} "
                    f"and supports_dynamic_size is not set"
                )
            if self.resize.supports_dynamic_size and self.resize.target_size is not None:
                raise ValueError("input.resize.target_size must be omitted when supports_dynamic_size: true")
        return self


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class BatchingSpec(_StrictModel):
    batch_axis: int = 0

    @model_validator(mode="after")
    def _check_batch_axis(self) -> "BatchingSpec":
        if self.batch_axis != 0:
            raise ValueError(f"batching.batch_axis must be 0 (non-zero batch axes are out of scope), got {self.batch_axis}")
        return self


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class OutputSpec(_StrictModel):
    name: str
    dtype: dict[Precision, OnnxDtype]
    shape: list[int] | None = None
    rank: int | None = None
    semantic: OutputSemantic
    fields: list[str] | None = None
    keypoint_axis: int | None = None
    keypoint_count: int | None = None

    @model_validator(mode="after")
    def _check_output(self) -> "OutputSpec":
        if not self.dtype:
            raise ValueError(f"output {self.name!r}: dtype must declare at least one precision")
        if self.rank is not None and self.shape is None:
            raise ValueError(f"output {self.name!r}: rank is present but shape is omitted")
        if self.shape is not None and self.rank is not None and self.rank != len(self.shape):
            raise ValueError(f"output {self.name!r}: rank {self.rank} != len(shape) {len(self.shape)}")
        if self.semantic == "detections" and not self.fields:
            raise ValueError(f"output {self.name!r}: semantic 'detections' requires non-empty `fields`")
        if self.semantic == "poses" and self.keypoint_axis != 2:
            raise ValueError(f"output {self.name!r}: semantic 'poses' requires keypoint_axis: 2")
        return self


# ---------------------------------------------------------------------------
# Detection decode
# ---------------------------------------------------------------------------


class DetectionDecodeSpec(_StrictModel):
    box_format: Literal["xyxy", "xywh", "cxcywh"] | None = None
    score_field: str | None = None
    class_field: str | None = None
    class_id_base: int = 0
    person_class_id: int = 0
    max_detections: int = 300
    may_include_non_person_classes: bool = True
    requires_nms: bool = False
    confidence_threshold_default: float = 0.7

    @model_validator(mode="after")
    def _check_ranges(self) -> "DetectionDecodeSpec":
        if not (0.0 <= self.confidence_threshold_default <= 1.0):
            raise ValueError(
                f"decode.confidence_threshold_default must be in [0, 1], got {self.confidence_threshold_default}"
            )
        if self.max_detections <= 0:
            raise ValueError(f"decode.max_detections must be positive, got {self.max_detections}")
        if self.class_id_base < 0:
            raise ValueError(f"decode.class_id_base must be non-negative, got {self.class_id_base}")
        if self.person_class_id < 0:
            raise ValueError(f"decode.person_class_id must be non-negative, got {self.person_class_id}")
        if self.person_class_id < self.class_id_base:
            raise ValueError(
                f"decode.person_class_id ({self.person_class_id}) must be >= decode.class_id_base ({self.class_id_base})"
            )
        return self


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

MappingEntry = Union[str, list[str], dict[str, float]]


class SkeletonSpec(_StrictModel):
    name: str
    edges: list[tuple[str, str]]


class PoseDecodeSpec(_StrictModel):
    method: Literal["simcc", "heatmap", "coordinate"]
    coordinate_scale: Literal["pixel", "normalized_01", "normalized_11"] | None = None
    is_3d: bool = False
    depth_unit: Literal["m", "mm", "pixel"] | None = None
    depth_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_decode(self) -> "PoseDecodeSpec":
        if self.coordinate_scale is not None and self.method != "coordinate":
            raise ValueError("pose.decode.coordinate_scale is only valid when method: coordinate")
        if self.method == "heatmap" and self.is_3d:
            raise ValueError("pose.decode.method: heatmap with is_3d: true is not supported")
        if self.is_3d and self.depth_unit is None:
            raise ValueError("pose.decode.depth_unit is required when is_3d: true")
        if not self.is_3d and (self.depth_unit is not None or self.depth_range is not None):
            raise ValueError("pose.decode.depth_unit/depth_range are only valid when is_3d: true")
        if self.is_3d and self.method == "simcc" and self.depth_range is None:
            raise ValueError("pose.decode.depth_range is required when is_3d: true and method: simcc")
        return self


class PoseSpec(_StrictModel):
    estimator_type: Literal["top_down_single_person", "top_down_multi_person", "bottom_up"]
    requires_object_detector: bool
    landmark_schema: str | None = None
    tracked_points: list[str]
    connections: list[SkeletonSpec]
    derived_points: dict[str, MappingEntry] | None = None
    canonical_mapping: dict[str, MappingEntry] | None = None
    decode: PoseDecodeSpec

    @model_validator(mode="before")
    @classmethod
    def _expand_prefixed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for field_name in ("derived_points", "canonical_mapping"):
                raw = data.get(field_name)
                if isinstance(raw, dict):
                    data[field_name] = expand_prefixed_mapping(raw)
        return data

    @model_validator(mode="after")
    def _check_pose(self) -> "PoseSpec":
        expects_detector = self.estimator_type != "bottom_up"
        if self.requires_object_detector != expects_detector:
            raise ValueError(
                f"pose.requires_object_detector ({self.requires_object_detector}) is inconsistent with "
                f"estimator_type {self.estimator_type!r} (expected {expects_detector})"
            )

        if len(set(self.tracked_points)) != len(self.tracked_points):
            raise ValueError("pose.tracked_points names must be unique")

        valid_names = set(self.tracked_points) | set((self.derived_points or {}).keys())
        skeleton_names = [s.name for s in self.connections]
        if len(set(skeleton_names)) != len(skeleton_names):
            raise ValueError("pose.connections[].name values must be unique")
        for skeleton in self.connections:
            for a, b in skeleton.edges:
                if a not in valid_names or b not in valid_names:
                    raise ValueError(
                        f"pose.connections[{skeleton.name!r}] edge ({a!r}, {b!r}) references an "
                        f"undeclared point (not in tracked_points or derived_points)"
                    )

        for source_dict, label in (
            (self.derived_points, "derived_points"),
            (self.canonical_mapping, "canonical_mapping"),
        ):
            if not source_dict:
                continue
            for name, entry in source_dict.items():
                for source in _entry_sources(entry):
                    if label == "derived_points" and source not in self.tracked_points:
                        raise ValueError(f"pose.derived_points[{name!r}] references unknown source {source!r}")
                    if label == "canonical_mapping" and source not in self.tracked_points and source not in (self.derived_points or {}):
                        raise ValueError(f"pose.canonical_mapping[{name!r}] references unknown source {source!r}")
        return self


def _entry_sources(entry: MappingEntry) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, list):
        return list(entry)
    return list(entry.keys())


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


class OverlayGroupSpec(_StrictModel):
    connections: list[tuple[str, str]] | None = None
    prefix: str | None = None
    connection_color: tuple[int, int, int]
    connection_thickness: int = 2
    keypoint_color: tuple[int, int, int] | None = None

    @model_validator(mode="after")
    def _check_group(self) -> "OverlayGroupSpec":
        if self.connections is not None and self.prefix is not None:
            raise ValueError("overlay group must use `connections` or `prefix`, not both")
        if self.connection_thickness <= 0:
            raise ValueError(f"overlay group connection_thickness must be positive, got {self.connection_thickness}")
        return self


class OverlaySpec(_StrictModel):
    skeleton: str | list[tuple[str, str]] | None = None
    groups: dict[str, OverlayGroupSpec] | None = None
    keypoint_color: tuple[int, int, int] | None = None

    @model_validator(mode="after")
    def _check_default_group_count(self) -> "OverlaySpec":
        if self.groups:
            defaults = [name for name, g in self.groups.items() if g.connections is None and g.prefix is None]
            if len(defaults) > 1:
                raise ValueError(f"overlay.groups has more than one default group (neither connections nor prefix): {defaults}")
        return self


# ---------------------------------------------------------------------------
# ONNX artifacts / sizes
# ---------------------------------------------------------------------------


class PrecisionArtifact(_StrictModel):
    filename: str
    url: str | None = None
    url_sha256: str | None = None
    input_dtype: OnnxDtype | None = None

    @model_validator(mode="after")
    def _check_url_sha(self) -> "PrecisionArtifact":
        if self.url is not None and self.url_sha256 is None:
            raise ValueError(f"artifact {self.filename!r}: url_sha256 is required when url is present")
        return self


class BatchArtifactGroup(_StrictModel):
    precision_artifacts: dict[Precision, PrecisionArtifact]
    output_shapes: list[list[int]] | None = None

    @model_validator(mode="after")
    def _check_nonempty(self) -> "BatchArtifactGroup":
        if not self.precision_artifacts:
            raise ValueError("onnx batch_artifacts group must declare at least one precision")
        return self


class OnnxSpec(_StrictModel):
    batch_artifacts: dict[str, BatchArtifactGroup]

    @model_validator(mode="after")
    def _check_batch_keys(self) -> "OnnxSpec":
        if not self.batch_artifacts:
            raise ValueError("onnx.batch_artifacts must declare at least one batch key")
        keys = list(self.batch_artifacts.keys())
        if keys == ["dynamic"]:
            return self
        if "dynamic" in keys:
            raise ValueError("onnx.batch_artifacts cannot mix `dynamic` with integer batch keys")
        for key in keys:
            if not key.isdigit() or int(key) <= 0:
                raise ValueError(f"onnx.batch_artifacts key must be a positive integer or 'dynamic', got {key!r}")
        return self

    @property
    def supports_dynamic_batch(self) -> bool:
        return "dynamic" in self.batch_artifacts

    @property
    def native_batch_sizes(self) -> list[int]:
        return sorted(int(k) for k in self.batch_artifacts if k != "dynamic")


class SizedSidecarInputWrapper(_StrictModel):
    """Placeholder retained for symmetry; SizedInputSpec is used directly."""


class SizeSpec(_StrictModel):
    input: SizedInputSpec
    onnx: OnnxSpec

    @model_validator(mode="after")
    def _check_output_shapes(self) -> "SizeSpec":
        for batch_key, group in self.onnx.batch_artifacts.items():
            if len(self.onnx.batch_artifacts) > 1 and group.output_shapes is None:
                raise ValueError(
                    f"onnx.batch_artifacts[{batch_key!r}] must declare output_shapes when more than one "
                    f"batch key is present"
                )
        return self


# ---------------------------------------------------------------------------
# Top-level SidecarModel
# ---------------------------------------------------------------------------


class SidecarModel(_StrictModel):
    schema_version: str
    model_id: str
    display_name: str
    role: list[Role]
    input: InputSpec
    batching: BatchingSpec = Field(default_factory=BatchingSpec)
    outputs: list[OutputSpec]
    decode: DetectionDecodeSpec | None = None
    pose: PoseSpec | None = None
    overlay: OverlaySpec | None = None
    sizes: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def _check_schema_version_pattern(self) -> "SidecarModel":
        from skellytracker.core.sidecar.errors import SidecarError
        from skellytracker.core.sidecar.versioning import parse_skellytracker_version, require_stable_version

        try:
            parse_skellytracker_version(self.schema_version)
            require_stable_version(self.schema_version)
        except SidecarError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _check_role(self) -> "SidecarModel":
        if not self.role:
            raise ValueError("role must be a non-empty subset of object_detector/pose_estimator")
        if len(set(self.role)) != len(self.role):
            raise ValueError(f"role must not repeat values, got {self.role}")
        return self

    @model_validator(mode="after")
    def _check_sizes(self) -> "SidecarModel":
        if not self.sizes:
            raise ValueError("sizes must declare at least one size")
        return self

    @model_validator(mode="after")
    def _check_role_requirements(self) -> "SidecarModel":
        if "object_detector" in self.role and self.decode is None:
            raise ValueError("role includes object_detector but `decode` is missing")
        if "object_detector" not in self.role and self.decode is not None:
            raise ValueError("`decode` is present but role does not include object_detector")
        if "pose_estimator" in self.role and self.pose is None:
            raise ValueError("role includes pose_estimator but `pose` is missing")
        if "pose_estimator" not in self.role and self.pose is not None:
            raise ValueError("`pose` is present but role does not include pose_estimator")
        return self

    @model_validator(mode="after")
    def _check_pose_requires_affine_crop(self) -> "SidecarModel":
        if self.pose is not None and self.pose.requires_object_detector:
            resize = self.input.resize
            if resize is None or resize.method != "affine_person_crop":
                raise ValueError(
                    "pose.requires_object_detector: true requires input.resize.method: affine_person_crop"
                )
        return self

    @model_validator(mode="after")
    def _check_outputs_semantics(self) -> "SidecarModel":
        semantics = [o.semantic for o in self.outputs]
        detection_semantics = {"detections"}
        pose_semantics = {"simcc_x", "simcc_y", "simcc_z", "heatmap", "keypoints", "poses"}

        for output in self.outputs:
            if output.semantic in detection_semantics and "object_detector" not in self.role:
                raise ValueError(f"output {output.name!r} has semantic 'detections' but role lacks object_detector")
            if output.semantic in pose_semantics and "pose_estimator" not in self.role:
                raise ValueError(f"output {output.name!r} has semantic {output.semantic!r} but role lacks pose_estimator")

        if "object_detector" in self.role:
            det_outputs = [o for o in self.outputs if o.semantic == "detections"]
            if len(det_outputs) != 1:
                raise ValueError(f"role includes object_detector: exactly one `detections` output is required, found {len(det_outputs)}")

        if self.decode is not None:
            det_output = next(o for o in self.outputs if o.semantic == "detections")
            fields = det_output.fields or []
            if self.decode.score_field is not None and self.decode.score_field not in fields:
                raise ValueError(f"decode.score_field {self.decode.score_field!r} not present in detections `fields` {fields}")
            if self.decode.class_field is not None and self.decode.class_field not in fields:
                raise ValueError(f"decode.class_field {self.decode.class_field!r} not present in detections `fields` {fields}")

        if self.pose is not None:
            self._check_pose_output_shapes(semantics)
        return self

    def _check_pose_output_shapes(self, semantics: list[str]) -> None:
        method = self.pose.decode.method
        is_3d = self.pose.decode.is_3d
        count = lambda sem: semantics.count(sem)  # noqa: E731

        if method == "simcc":
            expected = {"simcc_x": 1, "simcc_y": 1, "simcc_z": 1 if is_3d else 0}
            for sem, exp in expected.items():
                if count(sem) != exp:
                    raise ValueError(f"pose.decode.method=simcc requires exactly {exp} {sem!r} output(s), found {count(sem)}")
        elif method == "heatmap":
            if count("heatmap") != 1:
                raise ValueError(f"pose.decode.method=heatmap requires exactly one 'heatmap' output, found {count('heatmap')}")
        elif method == "coordinate":
            has_keypoints = count("keypoints") == 1
            has_bottom_up = count("detections") == 1 and count("poses") == 1
            if not (has_keypoints or has_bottom_up):
                raise ValueError(
                    "pose.decode.method=coordinate requires exactly one 'keypoints' output (single-person) "
                    "or one 'detections' + one 'poses' output (bottom_up)"
                )

        for output in self.outputs:
            if output.semantic in ("simcc_x", "simcc_y", "simcc_z", "heatmap", "keypoints"):
                kc = output.keypoint_count
                if kc is not None and kc != len(self.pose.tracked_points):
                    raise ValueError(
                        f"output {output.name!r} keypoint_count {kc} != len(pose.tracked_points) {len(self.pose.tracked_points)}"
                    )

    @model_validator(mode="after")
    def _check_overlay(self) -> "SidecarModel":
        if self.overlay is None or self.pose is None:
            return self
        skeleton_edge_sets: dict[str, set[tuple[str, str]]] = {s.name: set(s.edges) for s in self.pose.connections}
        valid_names = set(self.pose.tracked_points) | set((self.pose.derived_points or {}).keys())

        if isinstance(self.overlay.skeleton, str):
            if self.overlay.skeleton not in skeleton_edge_sets:
                raise ValueError(f"overlay.skeleton {self.overlay.skeleton!r} does not name a pose.connections entry")
            selected_edges = skeleton_edge_sets[self.overlay.skeleton]
        elif isinstance(self.overlay.skeleton, list):
            for a, b in self.overlay.skeleton:
                if a not in valid_names or b not in valid_names:
                    raise ValueError(f"overlay.skeleton inline edge ({a!r}, {b!r}) references an undeclared point")
            selected_edges = set(self.overlay.skeleton)
        else:
            selected_edges = set(self.pose.connections[0].edges) if self.pose.connections else set()

        if self.overlay.groups:
            for group_name, group in self.overlay.groups.items():
                if group.connections:
                    for edge in group.connections:
                        if edge not in selected_edges and tuple(reversed(edge)) not in selected_edges:
                            raise ValueError(
                                f"overlay.groups[{group_name!r}] connection {edge} is not part of the selected skeleton's edges"
                            )
        return self

    # ------------------------------------------------------------------
    # Per-size resolution
    # ------------------------------------------------------------------

    def resolved_size(self, size_name: str) -> SizeSpec:
        """Deep-merge `sizes[size_name]` over the shared top level and validate.

        Uses the same current-wins deep-merge semantics as `base` inheritance
        (see specs/sidecar-spec.md, "Sizes").
        """
        if size_name not in self.sizes:
            raise KeyError(f"No size {size_name!r} in sidecar {self.model_id!r}. Available: {list(self.sizes)}")
        top_level = self.model_dump(exclude={"sizes"}, mode="json")
        merged = _deep_merge(top_level, self.sizes[size_name])
        return SizeSpec.model_validate({"input": merged["input"], "onnx": merged["onnx"]})

    @property
    def default_size(self) -> str:
        return next(iter(self.sizes))
