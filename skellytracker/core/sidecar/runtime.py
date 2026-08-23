"""Glue between a validated `SidecarModel` and the existing session machinery
(`OnnxSession`).

Free functions only — sidecar-driven detectors (e.g. `YoloxPersonDetector`,
the RTMW wholebody detector) call `sidecar_model_spec` from their
`model_spec` classmethod rather than sidecars getting their own detector base
classes. This keeps the existing `ObjectDetector`/`KeypointDetector` ABCs and
`OBJECT_DETECTOR_REGISTRY`/`KEYPOINT_DETECTOR_REGISTRY` as the only plug-in
point.

Image preprocessing and detection decode also read sidecar-described specs
(`InputSpec`, `DetectionDecodeSpec`) but aren't sidecar *parsing* concerns —
see `skellytracker/core/detectors/image_preprocessing.py` and
`skellytracker/core/detectors/object_detection_decode.py`.
"""

from __future__ import annotations

from pathlib import Path

from beartype.typing import Callable

from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_session import OnnxModelSpec
from skellytracker.core.sidecar.model import Precision, SidecarModel


def sidecar_model_spec(
    sidecar: SidecarModel,
    *,
    size: str,
    batch_key: str,
    precision: Precision,
    name: str,
    sidecar_dir: Path,
    prepare: Callable[[Path], Path] | None = None,
    coreml_options: dict | None = None,
) -> OnnxModelSpec:
    """Build a lazy `OnnxModelSpec` for one (size, batch, precision) artifact.

    Pure — no I/O. Resolution (download, archive extraction, SHA-256
    verification against `artifact.url_sha256`) happens later, at the single
    choke point `OnnxSession.create()` already resolves every model through —
    this just carries `expected_filename`/`expected_sha256` for it to use.

    `name` is the caller's choice, not derived from `sidecar.model_id`/`size`
    — it's the key detectors later look up via `session.get_session(name)`,
    and a fixed naming convention here could collide across sidecars sharing
    one session. For a `local_path` artifact, `sidecar_dir` is used to build
    the absolute path (per specs/sidecar-spec.md "Storage layout" — the
    sidecar and its ONNX files live together in one leaf directory); its
    existence is not checked here, only when actually resolved.
    """
    resolved_size = sidecar.resolved_size(size)
    group = resolved_size.onnx.batch_artifacts[batch_key]
    artifact = group.precision_artifacts[precision]

    if artifact.url is not None:
        source = ModelSource(url=artifact.url)
    else:
        source = ModelSource(local_path=str(sidecar_dir / artifact.filename))

    target_size = (
        resolved_size.input.resize.target_size if resolved_size.input.resize else None
    )
    input_size = (
        tuple(target_size)
        if target_size is not None
        else tuple(resolved_size.input.shape[2:4])
    )

    return OnnxModelSpec(
        name=name,
        source=source,
        input_size=input_size,  # type: ignore[arg-type]
        prepare=prepare,
        coreml_options=coreml_options,
        expected_filename=artifact.filename,
        expected_sha256=artifact.url_sha256,
    )
