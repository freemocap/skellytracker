"""Adapt an isolated native runtime to the existing preparation/adoption contract."""

from pathlib import Path

from skellytracker.core.sessions.isolated_onnx_runtime import (
    IsolatedOnnxRuntime,
    IsolatedPreparationCancelled,
)
from skellytracker.core.sessions.model_registry import ModelSource
from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationJob,
    OnnxPreparationTicket,
    PreparationPhase,
)
from skellytracker.core.sessions.onnx_session import OnnxSession


class IsolatedOnnxPreparationJob(OnnxPreparationJob[OnnxSession]):
    """Cancellation terminates only the isolated child still owned by this job."""

    def __init__(self, *, runtime: IsolatedOnnxRuntime, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime

    def request_stop(self) -> None:
        super().request_stop()
        if self.snapshot().phase == PreparationPhase.STOPPING:
            self.runtime.request_stop()

    def run(self) -> None:
        try:
            super().run()
        except IsolatedPreparationCancelled:
            # Runtime raises this only after close has confirmed process exit.
            with self._lock:
                self._phase = PreparationPhase.CANCELLED
        except Exception:
            if self.runtime.status().alive:
                with self._lock:
                    self._phase = PreparationPhase.STOPPING
            raise

    def dispose(self) -> None:
        if self.snapshot().phase == PreparationPhase.STOPPING and self._backend is None:
            self.runtime.close()
            with self._lock:
                self._phase = PreparationPhase.CANCELLED
            return
        try:
            super().dispose()
        except Exception:
            if self.runtime.status().alive:
                with self._lock:
                    self._phase = PreparationPhase.STOPPING
            raise


def isolated_preparation_job(session: OnnxSession, ticket: OnnxPreparationTicket,
                             *, cache_root: Path | None = None) -> IsolatedOnnxPreparationJob:
    original = session.creation_config
    if original is None or not session._model_hashes or session.execution_provider != "cuda":
        raise ValueError("Isolated preparation requires a loaded CUDA session with model identity")
    original.execution_provider = "trt"
    original.device_id = session.device_id
    hashes, paths = dict(session._model_hashes), dict(session._model_paths)
    contracts = {
        name: tuple(tuple((item.name, item.type, tuple(item.shape)) for item in getattr(native, accessor)())
                    for accessor in ("get_inputs", "get_outputs"))
        for name, native in session._sessions.items()
    }
    resolved = original.model_copy(deep=True)
    for model in resolved.models:
        model.source = ModelSource(local_path=str(paths[model.name]))
        model.expected_filename = None
        model.expected_sha256 = hashes[model.name]
        model.prepare = None
    runtime = IsolatedOnnxRuntime(config=resolved, cache_root=cache_root)

    def build() -> OnnxSession:
        metadata = runtime.prepare()
        try:
            if metadata["hashes"] != hashes or metadata["provider"] != "trt" or metadata["device_id"] != original.device_id:
                raise ValueError("Isolated native identity does not match the CUDA session")
            candidate = OnnxSession(
                _sessions={name: runtime.proxy(name) for name in contracts}, execution_provider="trt",
                device_id=original.device_id, batch_size=original.batch_size,
                _creation_config=original.model_copy(deep=True), _model_paths=paths,
                _model_hashes=hashes, _prepared_warmed=True, _resource_owner=runtime,
            )
        except BaseException:
            runtime.close()
            raise
        return candidate

    def validate(candidate: OnnxSession) -> None:
        for name, expected in contracts.items():
            proxy = candidate.get_session(name)
            actual = tuple(tuple((item.name, item.type, tuple(item.shape)) for item in getattr(proxy, accessor)())
                           for accessor in ("get_inputs", "get_outputs"))
            if actual != expected or "TensorrtExecutionProvider" not in proxy.get_providers():
                raise ValueError("Isolated native tensor/provider contract changed")

    return IsolatedOnnxPreparationJob(runtime=runtime, ticket=ticket, build=build, validate=validate)
