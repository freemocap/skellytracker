"""Detached ONNX preparation; native work is confined to the job's worker."""

import hashlib
import json
from pathlib import Path

from filelock import FileLock

from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationJob,
    OnnxPreparationTicket,
)
from skellytracker.core.sessions.onnx_session import OnnxSession, _warmup


def preparation_job(session: OnnxSession, ticket: OnnxPreparationTicket) -> OnnxPreparationJob:
    config = session.creation_config
    if config is None or not session._model_hashes or session.execution_provider != "cuda":
        raise ValueError("Preparation requires a loaded CUDA session with model identity")
    config.execution_provider = "trt"
    config.device_id = session.device_id
    # Freeze metadata/contract without retaining native CUDA handles in a worker.
    hashes = dict(session._model_hashes)
    paths = dict(session._model_paths)
    contracts = {
        name: tuple(tuple((item.name, item.type, tuple(item.shape)) for item in getattr(native, accessor)())
                    for accessor in ("get_inputs", "get_outputs"))
        for name, native in session._sessions.items()
    }

    def build() -> OnnxSession:
        import onnxruntime
        import tensorrt

        resolved = config.model_copy(deep=True)
        for model in resolved.models:
            model.source = ModelSource(local_path=str(paths[model.name]))
            model.prepare = None
            model.expected_filename = None
            model.expected_sha256 = hashes[model.name]
        identity = json.dumps({
            "models": hashes, "config": config.model_dump(exclude={"models"}),
            "ort": onnxruntime.__version__, "trt": tensorrt.__version__,
            "contracts": contracts,
        }, sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        cache = Path.home() / ".cache" / "skellytracker" / "trt_prepared" / digest
        cache.mkdir(parents=True, exist_ok=True)
        # Different model/runtime/config identities never invalidate live caches;
        # equivalent builders serialize cache writes across application processes.
        with FileLock(str(cache / "build.lock")):
            candidate = OnnxSession.create(resolved, engine_cache_dir=cache)
            candidate._creation_config = config.model_copy(deep=True)
            try:
                # Validate all admitted crop counts before publishing ready.
                for batch in range(1, config.batch_size):
                    _warmup(candidate, resolved.models, "trt", batch_size=batch)
                candidate._prepared_warmed = True
            except BaseException:
                candidate.close()
                raise
        return candidate

    def validate(candidate: OnnxSession) -> None:
        if candidate._model_hashes != hashes or candidate.creation_config != config:
            raise ValueError("Prepared model identity changed")
        if candidate.device_id != config.device_id or candidate.execution_provider != "trt":
            raise ValueError("Prepared device/provider changed")
        for name, expected in contracts.items():
            native = candidate.get_session(name)
            if "TensorrtExecutionProvider" not in native.get_providers():
                raise ValueError("TensorRT provider did not load")
            actual = tuple(tuple((item.name, item.type, tuple(item.shape)) for item in getattr(native, accessor)())
                           for accessor in ("get_inputs", "get_outputs"))
            if actual != expected:
                raise ValueError("Prepared tensor contract changed")

    return OnnxPreparationJob(ticket=ticket, build=build, validate=validate)
