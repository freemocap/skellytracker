"""Backend-free home for the ExecutionProviderName type alias.

Kept in its own module (no onnxruntime/onnx import) so that config classes and
downstream consumers can reference the type without dragging the ONNX Runtime
native library into the import graph. `ort_session_utils` and the rtmpose
config/detector all import the alias from here.
"""
from typing import Literal

ExecutionProviderName = Literal["trt", "cuda", "coreml", "cpu"]