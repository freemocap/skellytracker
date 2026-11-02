"""CPU lifecycle tests; no native construction or GPU ownership claims."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, get_ident

import pytest

from skellytracker.core.sessions.onnx_preparation import (
    OnnxPreparationJob,
    OnnxPreparationTicket,
    PreparationPhase,
)


class Backend:
    def __init__(self, *, close_error=False):
        self.closes = 0
        self.close_thread = None
        self.close_error = close_error

    def close(self):
        self.closes += 1
        self.close_thread = get_ident()
        if self.close_error:
            raise RuntimeError("close failed")


@pytest.fixture
def ticket():
    return OnnxPreparationTicket("lease-1", 1, "resolved-configuration-identity")


def test_success_transfers_once_without_closing(ticket):
    backend = Backend()
    validated = []
    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=validated.append)
    assert job.snapshot().phase == PreparationPhase.QUEUED
    assert job.take_ready(expected_ticket=ticket) is None
    job.run()
    assert validated == [backend]
    assert job.snapshot().phase == PreparationPhase.READY
    assert job.take_ready(expected_ticket=ticket) is backend
    assert job.take_ready(expected_ticket=ticket) is None
    job.request_stop()
    job.dispose()
    assert backend.closes == 0
    assert job.snapshot().phase == PreparationPhase.TRANSFERRED
    with pytest.raises(RuntimeError, match="already started"):
        job.run()


def test_queued_stop_never_constructs(ticket):
    def forbidden():
        pytest.fail("cancelled job constructed a backend")

    job = OnnxPreparationJob(ticket=ticket, build=forbidden, validate=lambda _: forbidden())
    job.request_stop()
    job.run()
    job.dispose()
    assert job.snapshot().phase == PreparationPhase.CANCELLED


@pytest.mark.parametrize("stage", ["build", "validate"])
def test_blocked_native_step_does_not_block_polling_and_stop_is_honest(ticket, stage):
    entered, release = Event(), Event()
    backend = Backend()
    worker_thread = []
    validated = []

    def gate():
        worker_thread.append(get_ident())
        entered.set()
        assert release.wait(5), "test failed to release worker"

    def build():
        if stage == "build":
            gate()
        return backend

    def validate(value):
        validated.append(value)
        if stage == "validate":
            gate()

    job = OnnxPreparationJob(ticket=ticket, build=build, validate=validate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(job.run)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="already started"):
                job.run()
            assert job.take_ready(expected_ticket=ticket) is None
            job.request_stop()
            assert job.snapshot().phase == PreparationPhase.STOPPING
            assert not future.done()
            assert backend.closes == 0
            with pytest.raises(RuntimeError, match="Request stop"):
                job.dispose()
        finally:
            release.set()
        future.result(timeout=5)
    assert job.snapshot().phase == PreparationPhase.CANCELLED
    assert backend.closes == 1
    assert backend.close_thread == worker_thread[0] != get_ident()
    assert validated == ([] if stage == "build" else [backend])


def test_ready_stop_requires_scheduled_disposal(ticket):
    backend = Backend()
    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=lambda _: None)
    job.run()
    job.request_stop()
    assert job.take_ready(expected_ticket=ticket) is None
    assert backend.closes == 0
    assert job.snapshot().phase == PreparationPhase.STOPPING
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(job.dispose).result(timeout=5)
    job.dispose()
    assert backend.closes == 1
    assert backend.close_thread != get_ident()


@pytest.mark.parametrize("field,value", [("request_id", "another"), ("generation", 2), ("identity", "different")])
def test_stale_identity_cannot_take_resource(ticket, field, value):
    backend = Backend()
    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=lambda _: None)
    job.run()
    with pytest.raises(ValueError, match="does not match"):
        job.take_ready(expected_ticket=replace(ticket, **{field: value}))
    assert job.snapshot().phase == PreparationPhase.READY
    job.dispose()
    assert backend.closes == 1


@pytest.mark.parametrize("close_error", [False, True])
def test_validation_failure_closes_once_and_preserves_errors(ticket, close_error):
    backend = Backend(close_error=close_error)

    def invalid(_):
        raise ValueError("actual provider mismatch")

    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=invalid)
    with pytest.raises(BaseExceptionGroup if close_error else ValueError) as caught:
        job.run()
    if close_error:
        assert [str(error) for error in caught.value.exceptions] == ["actual provider mismatch", "close failed"]
    assert job.snapshot().phase == PreparationPhase.FAILED
    assert job.take_ready(expected_ticket=ticket) is None
    job.dispose()
    assert backend.closes == 1


def test_build_failure_never_becomes_ready(ticket):
    def failed():
        raise RuntimeError("construction failed")

    job = OnnxPreparationJob(ticket=ticket, build=failed, validate=lambda _: None)
    with pytest.raises(RuntimeError, match="construction failed"):
        job.run()
    assert job.snapshot().reason == "construction failed"
    assert job.snapshot().phase == PreparationPhase.FAILED
    assert job.take_ready(expected_ticket=ticket) is None


def test_cleanup_failure_does_not_retry_native_close(ticket):
    backend = Backend(close_error=True)
    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=lambda _: None)
    job.run()
    with pytest.raises(RuntimeError, match="close failed"):
        job.dispose()
    job.dispose()
    job.request_stop()
    assert backend.closes == 1
    assert job.snapshot().phase == PreparationPhase.FAILED


def test_disposal_callback_does_not_hold_status_lock(ticket):
    entered, release = Event(), Event()

    class BlockingClose(Backend):
        def close(self):
            super().close()
            entered.set()
            assert release.wait(5)

    backend = BlockingClose()
    job = OnnxPreparationJob(ticket=ticket, build=lambda: backend, validate=lambda _: None)
    job.run()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(job.dispose)
        try:
            assert entered.wait(5)
            assert job.snapshot().phase == PreparationPhase.DISPOSING
            assert job.take_ready(expected_ticket=ticket) is None
            job.dispose()
            job.request_stop()
        finally:
            release.set()
        future.result(timeout=5)
    assert backend.closes == 1


@pytest.mark.parametrize("args", [("", 0, "identity"), ("id", -1, "identity"), ("id", 0, "")])
def test_ticket_requires_identity(args):
    with pytest.raises(ValueError):
        OnnxPreparationTicket(*args)


def test_import_does_not_load_native_runtime():
    script = '''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'tensorrt', 'torch'}:
            raise AssertionError('native import: ' + fullname)
sys.meta_path.insert(0, Guard())
from skellytracker.core.sessions.onnx_preparation import OnnxPreparationJob
'''
    subprocess.run(  # noqa: S603 - fixed local interpreter and literal import-guard script
        [sys.executable, "-B", "-c", script], check=True, capture_output=True, text=True,
    )
