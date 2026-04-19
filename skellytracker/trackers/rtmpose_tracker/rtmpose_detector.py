import ctypes
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
from numpy.typing import NDArray
from rtmlib import Wholebody

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetector, BaseDetectorConfig, TrackerType
from skellytracker.trackers.rtmpose_tracker.rtmpose_observation import RTMPoseObservation


def _make_nvidia_pip_dlls_discoverable_on_windows() -> None:
    """Ensure cuDNN can find its own sub-DLLs when it lazy-loads them at runtime.

    cuDNN 9's main `cudnn64_9.dll` loads sibling sub-DLLs (`cudnn_graph64_9.dll`,
    `cudnn_engines_tensor_ir64_9.dll`, ...) the first time certain ops run. Those
    lazy loads come from inside a C library via `LoadLibrary`, which uses the
    process `PATH` — NOT the directories registered by `os.add_dll_directory`
    (those only apply to `LoadLibraryEx` calls with `LOAD_LIBRARY_SEARCH_USER_DIRS`,
    which is what Python uses for its own imports but not what cuDNN uses
    internally). `onnxruntime.preload_dlls()` also only covers a fixed hardcoded
    subset of cuDNN sub-DLLs and misses newer ones like
    `cudnn_engines_tensor_ir64_9.dll`, so by itself it leaves a landmine that
    detonates with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` mid-inference.

    Three layered guarantees, all of which must succeed:
      1. Prepend every `nvidia/*/bin` dir to `PATH` so OS-level `LoadLibrary`
         calls find the DLLs.
      2. Register those dirs with `os.add_dll_directory` so Python-side loads
         find them too.
      3. Proactively load every DLL in `nvidia/cudnn/bin` with `ctypes.WinDLL`
         by absolute path. Once a DLL is in the process's loaded-module table,
         any subsequent `LoadLibrary("<basename>")` resolves from that table
         and never touches the filesystem search path — making this the
         bulletproof layer.
    """
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Could not find `nvidia` pip namespace package. "
            "Install skellytracker with the `rtmpose-gpu` extra: "
            "`uv sync --extra rtmpose-gpu`."
        )
    nvidia_root = Path(spec.submodule_search_locations[0])
    bin_dirs = sorted(nvidia_root.glob("*/bin"))
    if not bin_dirs:
        raise RuntimeError(
            f"Found `nvidia` package at {nvidia_root} but no `nvidia/*/bin` "
            f"subdirectories inside it — the CUDA/cuDNN runtime pip packages "
            f"are not installed."
        )

    # Layer 1: prepend to PATH (cuDNN's internal LoadLibrary calls read this).
    bin_dir_strs = [str(d) for d in bin_dirs]
    os.environ["PATH"] = os.pathsep.join([*bin_dir_strs, os.environ.get("PATH", "")])

    # Layer 2: register with Python's DLL loader for Python-side loads.
    for bin_dir in bin_dirs:
        os.add_dll_directory(str(bin_dir))

    # Layer 3: proactively load every cuDNN DLL by full path so cuDNN never
    # needs to search for them.
    cudnn_bin = nvidia_root / "cudnn" / "bin"
    if not cudnn_bin.is_dir():
        raise RuntimeError(
            f"Expected cuDNN bin directory at {cudnn_bin}, but it does not exist. "
            f"nvidia-cudnn-cu12 is not installed or is corrupted."
        )
    cudnn_dlls = sorted(cudnn_bin.glob("*.dll"))
    if not cudnn_dlls:
        raise RuntimeError(f"No cuDNN DLLs found in {cudnn_bin}.")
    for dll_path in cudnn_dlls:
        ctypes.WinDLL(str(dll_path))


class RTMPoseDetectorConfig(BaseDetectorConfig):
    tracker_type:TrackerType = TrackerType.RTMPOSE
    confidence_threshold: float = 0.5
    mode: str = "performance"
    backend: str = "onnxruntime"
    device: str = "cuda"


@dataclass
class RTMPoseDetector(BaseDetector):
    config: RTMPoseDetectorConfig
    detector: Wholebody

    @classmethod
    def create(cls, config: RTMPoseDetectorConfig | None = None) -> "RTMPoseDetector":
        config = config or RTMPoseDetectorConfig()
        if config.device == "cuda":
            if sys.platform == "win32":
                _make_nvidia_pip_dlls_discoverable_on_windows()
            onnxruntime.preload_dlls()
        detector = Wholebody(
            to_openpose=False,
            mode=config.mode,
            backend=config.backend,
            device=config.device,
        )
        return cls(config=config, detector=detector)

    def detect(self, frame_number: int, image: NDArray[np.uint8]) -> RTMPoseObservation:
        # rtmlib's type stubs are incorrect — keypoints is float64 at runtime, scores is float32.
        keypoints: NDArray[np.float64]
        scores: NDArray[np.float32]
        keypoints, scores = self.detector(image)
        return RTMPoseObservation.from_detection_results(
            frame_number=frame_number,
            keypoints=keypoints,
            scores=scores,
            image_size=(int(image.shape[0]), int(image.shape[1])),
        )
