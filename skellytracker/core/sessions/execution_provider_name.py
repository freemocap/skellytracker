"""Backend-free home for the ExecutionProviderName type alias.

Kept in its own module (no onnxruntime/onnx import) so that config classes and
downstream consumers can reference the type without dragging the ONNX Runtime
native library into the import graph.
"""
from typing import Literal

ExecutionProviderName = Literal["trt", "cuda", "coreml", "directml", "cpu"]
