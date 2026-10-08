"""Explicit provider selection shared by inference tests and benchmarks."""
import os


def requested_providers():
    """Preserve autodetection unless a CPU/CUDA test matrix is requested."""
    value = os.environ.get("SKELLYTRACKER_TEST_PROVIDERS")
    if value is None:
        return [None]
    providers = [part.strip() for part in value.split(",")]
    if not providers or any(p not in ("cpu", "cuda") for p in providers):
        raise ValueError("SKELLYTRACKER_TEST_PROVIDERS must be cpu, cuda, or cpu,cuda")
    if len(set(providers)) != len(providers):
        raise ValueError("Duplicate test providers")
    return providers


def verify_session_provider(session, model_names, requested):
    """Reject a session that silently lost the explicitly requested provider.

    CUDA may still assign unsupported or shape operations to CPU. This checks
    session activation, not a claim that every graph node executes on the GPU.
    """
    actual = {name: session.get_session(name).get_providers() for name in model_names}
    if requested is not None:
        expected = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[requested]
        for name, providers in actual.items():
            if not providers or providers[0] != expected:
                raise AssertionError(f"{name}: requested {requested}, active providers are {providers}")
    return actual
