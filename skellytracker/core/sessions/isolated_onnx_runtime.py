"""Own native model construction and inference in a dedicated spawn process.

Only descriptors, metadata and numerical arrays cross the pipe. A ready worker
is retained for inference; cached files never stand in for a live native session.
Applications must explicitly opt into this resource and schedule preparation
away from their event loop. Termination affects only this object's owned child.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from types import SimpleNamespace

from skellytracker.core.sessions.onnx_model_spec import OnnxSessionConfig


class IsolatedPreparationCancelled(RuntimeError):
    """Preparation stopped and the isolated worker has exited."""


@dataclass(frozen=True, slots=True)
class IsolatedRuntimeStatus:
    phase: str
    elapsed_seconds: float
    process_id: int | None
    alive: bool
    reason: str | None
    model_name: str | None


def _describe_native(native) -> dict:
    return {
        "inputs": [(item.name, item.type, tuple(item.shape)) for item in native.get_inputs()],
        "outputs": [(item.name, item.type, tuple(item.shape)) for item in native.get_outputs()],
        "providers": native.get_providers(),
    }


def _reject_external_model_data(config: OnnxSessionConfig) -> None:
    """A digest of the ONNX file alone cannot identify external tensor bytes."""
    import onnx

    def has_external_tensor(message) -> bool:
        if isinstance(message, onnx.TensorProto):
            return message.data_location == onnx.TensorProto.EXTERNAL
        for descriptor, value in message.ListFields():
            if descriptor.message_type is not None:
                children = value if descriptor.is_repeated else (value,)
                if any(has_external_tensor(child) for child in children):
                    return True
        return False

    for model in config.models:
        descriptor = onnx.load_model(model.source.local_path, load_external_data=False)
        if has_external_tensor(descriptor):
            raise ValueError("Isolated TensorRT preparation requires self-contained ONNX model bytes; "
                             f"model {model.name!r} uses external tensor data")


def _build_native_session(config: OnnxSessionConfig, job_directory: Path, progress):
    """Seed a private build cache; publish a reusable copy only after warmup."""
    from skellytracker.core.sessions.onnx_session import OnnxSession, _warmup

    cache = job_directory / "engines"
    cache.mkdir()
    seed = None
    lock = nullcontext()
    if config.execution_provider == "trt":
        _reject_external_model_data(config)
        import onnxruntime
        import tensorrt
        from filelock import FileLock

        identity = json.dumps({"config": config.model_dump(mode="json"),
                               "ort": onnxruntime.__version__, "trt": tensorrt.__version__}, sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        seed_root = job_directory.parent / "completed"
        seed_root.mkdir(exist_ok=True)
        seed = seed_root / digest
        lock = FileLock(str(seed_root / f"{digest}.lock"))
    with lock:
        if seed is not None and (seed / "preparation_complete.json").is_file():
            shutil.copytree(seed, cache, dirs_exist_ok=True)
        session = OnnxSession.create(config, engine_cache_dir=cache, progress=progress)
        try:
            for batch in range(1, config.batch_size):
                _warmup(session, config.models, session.execution_provider, batch_size=batch, progress=progress)
            if config.execution_provider == "trt" and any(
                "TensorrtExecutionProvider" not in native.get_providers() for native in session._sessions.values()
            ):
                raise ValueError("TensorRT did not load for every requested model")
            if seed is not None and not seed.exists():
                publication = job_directory / "publication"
                shutil.copytree(cache, publication)
                (publication / "preparation_complete.json").write_text(identity, encoding="utf-8")
                publication.rename(seed)
            return session
        except BaseException:
            session.close()
            raise


def _native_process(connection, config: OnnxSessionConfig, cache_directory: str) -> None:
    """Import and construct the native runtime only inside the child."""
    session = None
    try:
        connection.send(("phase", "loading_models"))
        def progress(phase: str, model: str) -> None:
            connection.send(("phase", {"phase": phase, "model_name": model}))

        session = _build_native_session(config, Path(cache_directory), progress)
        models = {name: _describe_native(native) for name, native in session._sessions.items()}
        if config.execution_provider == "trt" and any(
            "TensorrtExecutionProvider" not in model["providers"] for model in models.values()
        ):
            raise ValueError("TensorRT did not load for every requested model")
        connection.send(("ready", {
            "models": models, "hashes": session._model_hashes,
            "device_id": session.device_id, "provider": session.execution_provider,
        }))
        while True:
            command = connection.recv()
            if command[0] == "close":
                return
            if command[0] != "run":
                raise ValueError("Unknown isolated runtime command")
            _, model_name, output_names, inputs = command
            try:
                result = session.run(model_name, inputs, output_names)
                connection.send(("result", result))
            except Exception as error:  # noqa: BLE001 - marshal inference failures across the process boundary
                connection.send(("inference_error", f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:  # noqa: BLE001 - report child failure before releasing native resources
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if session is not None:
            session.close()
        connection.close()


class IsolatedOnnxRuntime:
    """One explicit preparation and its persistent inference process.

    ``prepare`` and ``run`` are blocking worker operations with finite deadlines.
    ``status`` never waits on native work. ``request_stop`` signals only the
    owned process and does not claim completion until ``close`` observes exit.
    The application retains this owner until close succeeds.
    """

    def __init__(self, *, config: OnnxSessionConfig, cache_root: Path | None = None,
                 preparation_timeout: float = 1200.0, inference_timeout: float = 30.0,
                 process_target=None) -> None:
        if any(model.prepare is not None or model.source.local_path is None or model.expected_sha256 is None
               for model in config.models):
            raise ValueError("Isolated preparation requires resolved local model bytes and hashes without callbacks")
        if preparation_timeout <= 0 or inference_timeout <= 0:
            raise ValueError("Isolated runtime deadlines must be positive")
        self.config = config.model_copy(deep=True)
        self._cache_root = cache_root or (Path.home() / ".cache" / "skellytracker" / "isolated_onnx")
        self._preparation_timeout = preparation_timeout
        self._inference_timeout = inference_timeout
        self._target = process_target or _native_process
        self._context = multiprocessing.get_context("spawn")
        self._connection = None
        self._process = None
        self._directory: Path | None = None
        self._stop = Event()
        self._lifecycle = Lock()
        self._rpc = Lock()
        self._started: float | None = None
        self._ended: float | None = None
        self._phase = "queued"
        self._reason: str | None = None
        self._metadata: dict | None = None
        self._model_name: str | None = None
        self._owned_root: Path | None = None

    def status(self) -> IsolatedRuntimeStatus:
        process = self._process
        alive = process.is_alive() if process is not None else False
        if self._phase == "ready" and not alive:
            self._phase = "failed"
            self._reason = "Isolated native worker exited"
        return IsolatedRuntimeStatus(
            phase=self._phase,
            elapsed_seconds=0.0 if self._started is None else (
                (self._ended if self._ended is not None else monotonic()) - self._started
            ),
            process_id=process.pid if process is not None else None,
            alive=alive,
            reason=self._reason,
            model_name=self._model_name,
        )

    def prepare(self) -> dict:
        with self._lifecycle:
            if self._process is not None:
                raise RuntimeError("Isolated runtime has already started")
            if self._stop.is_set():
                self._phase = "cancelled"
                raise IsolatedPreparationCancelled("Preparation stopped before native work started")
            self._started = monotonic()
            self._phase = "starting_worker"
            if self._cache_root is not None:
                self._cache_root.mkdir(parents=True, exist_ok=True)
            self._directory = Path(tempfile.mkdtemp(prefix="trt-job-", dir=self._cache_root))
            self._owned_root = self._directory.parent.resolve()
            parent, child = self._context.Pipe()
            self._connection = parent
            self._process = self._context.Process(
                target=self._target, args=(child, self.config, str(self._directory)),
                name="OnnxPreparation", daemon=True,
            )
            try:
                self._process.start()
            except BaseException:
                self._process = None
                parent.close()
                shutil.rmtree(self._directory)
                self._directory = None
                self._phase = "failed"
                raise
            finally:
                child.close()
        deadline = monotonic() + self._preparation_timeout
        try:
            while True:
                if self._stop.is_set():
                    raise IsolatedPreparationCancelled("Preparation was stopped")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Native model preparation exceeded its deadline")
                if self._connection.poll(min(0.05, remaining)):
                    kind, payload = self._connection.recv()
                    if kind == "phase":
                        if isinstance(payload, dict):
                            self._phase = payload["phase"]
                            self._model_name = payload["model_name"]
                        else:
                            self._phase = payload
                    elif kind == "ready":
                        if self._stop.is_set():
                            raise IsolatedPreparationCancelled("Preparation was stopped")
                        self._metadata = payload
                        self._phase = "ready"
                        self._ended = monotonic()
                        return payload
                    else:
                        raise RuntimeError(f"Native preparation failed: {payload}")
                if not self._process.is_alive():
                    raise RuntimeError(f"Native preparation worker exited with code {self._process.exitcode}")
        except BaseException as error:
            cancelled = self._stop.is_set()
            self._reason = str(error)
            self.close()
            self._phase = "cancelled" if cancelled else "failed"
            if cancelled:
                raise IsolatedPreparationCancelled("Preparation stopped; native worker exited") from error
            raise

    def proxy(self, model_name: str) -> IsolatedModelProxy:
        self.ensure_ready()
        return IsolatedModelProxy(self, model_name, self._metadata["models"][model_name])

    def ensure_ready(self) -> None:
        if self._metadata is None or self.status().phase != "ready" or self._stop.is_set():
            raise RuntimeError("Isolated runtime is not ready")

    def run(self, model_name: str, output_names: list[str] | None, inputs: dict) -> list:
        with self._rpc:
            self.ensure_ready()
            try:
                self._connection.send(("run", model_name, output_names, inputs))
                if not self._connection.poll(self._inference_timeout):
                    self.request_stop()
                    raise TimeoutError("Isolated inference exceeded its deadline")
                kind, payload = self._connection.recv()
                if kind != "result":
                    raise RuntimeError(f"Isolated inference failed: {payload}")
                return payload
            except (EOFError, BrokenPipeError, OSError) as error:
                self._phase = "failed"
                self._reason = "Isolated inference worker exited"
                raise RuntimeError(self._reason) from error

    def request_stop(self) -> None:
        self._stop.set()
        self._phase = "stopping"
        with self._lifecycle:
            if self._process is not None and self._process.is_alive():
                self._process.terminate()

    def close(self) -> None:
        """Bounded teardown; retain handles and cache if the child has not exited."""
        self.request_stop()
        process = self._process
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            if process.is_alive():
                self._reason = "Native worker is still stopping"
                raise RuntimeError(self._reason)
        if self._connection is not None:
            self._connection.close()
        if self._directory is not None:
            resolved = self._directory.resolve()
            if resolved.parent != self._owned_root or not resolved.name.startswith("trt-job-"):
                raise RuntimeError("Refusing cleanup outside the owned preparation directory")
            shutil.rmtree(resolved)
            self._directory = None
        self._phase = "closed"
        if self._ended is None:
            self._ended = monotonic()


class IsolatedModelProxy:
    """Immutable local metadata plus RPC, never a native session handle."""

    def __init__(self, owner: IsolatedOnnxRuntime, model_name: str, metadata: dict) -> None:
        self._owner = owner
        self._model_name = model_name
        self._inputs = tuple(tuple(item) for item in metadata["inputs"])
        self._outputs = tuple(tuple(item) for item in metadata["outputs"])
        self._providers = tuple(metadata["providers"])

    def get_inputs(self) -> list:
        return [SimpleNamespace(name=name, type=kind, shape=list(shape)) for name, kind, shape in self._inputs]

    def get_outputs(self) -> list:
        return [SimpleNamespace(name=name, type=kind, shape=list(shape)) for name, kind, shape in self._outputs]

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(self, output_names: list[str] | None, inputs: dict) -> list:
        return self._owner.run(self._model_name, output_names, inputs)
