"""Check provider selection and benchmark reporting without inference packages."""
from types import SimpleNamespace

import numpy as np
import pytest

from skellytracker.tests.benchmark_providers import latency_summary, output_agreement
from skellytracker.tests.onnx_provider_checks import requested_providers, verify_session_provider


def test_explicit_cpu_cuda_matrix(monkeypatch):
    monkeypatch.setenv("SKELLYTRACKER_TEST_PROVIDERS", "cpu,cuda")
    assert requested_providers() == ["cpu", "cuda"]
    monkeypatch.delenv("SKELLYTRACKER_TEST_PROVIDERS")
    assert requested_providers() == [None]


@pytest.mark.parametrize("value", ["", "cuda,cuda", "auto", "cpu,gpu"])
def test_invalid_provider_matrix_fails(monkeypatch, value):
    monkeypatch.setenv("SKELLYTRACKER_TEST_PROVIDERS", value)
    with pytest.raises(ValueError):
        requested_providers()


def test_cpu_fallback_cannot_be_reported_as_cuda():
    session = SimpleNamespace(get_session=lambda name: SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
    with pytest.raises(AssertionError, match="requested cuda"):
        verify_session_provider(session, ["test-model"], "cuda")
    assert verify_session_provider(session, ["test-model"], "cpu") == {"test-model": ["CPUExecutionProvider"]}


def test_cuda_session_allows_cpu_for_unsupported_operations():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = SimpleNamespace(get_session=lambda name: SimpleNamespace(get_providers=lambda: providers))
    assert verify_session_provider(session, ["test-model"], "cuda") == {"test-model": providers}


def test_throughput_uses_total_time_instead_of_averaging_fps():
    summary = latency_summary([0.01, 0.03])
    assert summary["frames_per_second"] == pytest.approx(50)
    assert summary["median_ms"] == pytest.approx(20)
    with pytest.raises(ValueError):
        latency_summary([])


def test_output_agreement_distinguishes_missing_detections_from_coordinate_error():
    cpu = np.array([[[0., 0., 0.], [np.nan, np.nan, np.nan]]])
    cuda = np.array([[[3., 4., 0.], [1., 1., 0.]]])
    report = output_agreement(cpu, cuda)
    assert report["comparable_keypoints"] == 1
    assert report["finite_mask_disagreements"] == 1
    assert report["median_xy_difference_pixels"] == 5
