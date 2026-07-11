"""Structured error types for skellytracker sessions.

FMC (freemocap) catches these at the UI layer to show appropriate error messages
rather than raw Python exceptions.
"""


class SkellytrackerSessionError(Exception):
    """Base class for all skellytracker session errors."""


class SessionCreationError(SkellytrackerSessionError):
    """Raised when a session cannot be created.

    Causes: the requested execution provider is not installed, or the ONNX
    Runtime failed to load a model with that provider. There is no fallback —
    FMC should surface this to the user so they can pick a different EP.
    """


class InferenceError(SkellytrackerSessionError):
    """Base class for errors that occur during inference (after session creation)."""


class VRAMExhaustionError(InferenceError):
    """Raised when inference fails due to GPU out-of-memory.

    Typical causes: batch size too large for available VRAM, or model footprint
    grew beyond the arena ceiling. FMC can suggest reducing batch size or
    switching to a lighter model preset.
    """


class InferencePipelineError(InferenceError):
    """Raised when the ONNX Runtime pipeline fails during session.run().

    Typical causes: corrupted model weights, unsupported operator, shape
    mismatch between input tensor and model expectations.
    """
