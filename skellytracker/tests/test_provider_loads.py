"""Auto provider selection must use providers that load, not ones merely installed."""
import pytest

from skellytracker.core.sessions import ort_session_utils as utils

LISTED = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def probes(monkeypatch):
    """Simulate session probes; record which execution providers were probed."""
    calls = []
    results = {}
    monkeypatch.setattr(utils, "_session_probe_results", {})
    monkeypatch.setattr(utils.ort, "get_available_providers", lambda: LISTED)

    def probe(ep, *, cuda):
        calls.append(ep)
        return results[ep]

    monkeypatch.setattr(utils, "_probe_session_provider", probe)
    return calls, results


def test_installed_cuda_without_a_device_selects_cpu(probes):
    calls, results = probes
    results["CUDAExecutionProvider"] = False
    assert not utils.provider_loads("cuda")
    assert not utils.provider_loads("trt")
    assert utils.auto_detect_provider() == "cpu"
    # TensorRT is never probed when CUDA cannot load, and probes are cached.
    assert calls == ["CUDAExecutionProvider"]


def test_loadable_tensorrt_keeps_its_auto_preference(probes):
    calls, results = probes
    results.update({"CUDAExecutionProvider": True, "TensorrtExecutionProvider": True})
    assert utils.auto_detect_provider() == "trt"
    assert calls == ["CUDAExecutionProvider", "TensorrtExecutionProvider"]


def test_uninstalled_provider_is_not_probed(probes):
    calls, _ = probes
    assert not utils.provider_loads("directml")
    assert utils.provider_loads("cpu")
    assert calls == []


def test_real_cpu_probe_builds_a_session():
    assert utils._probe_session_provider("CPUExecutionProvider", cuda=False)
