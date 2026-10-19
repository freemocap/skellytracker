"""Live RTMPose wholebody demo (133 keypoints).

Usage::

    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.run_demo
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.run_demo --model rtmw-x-l_384x288
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.run_demo --provider cpu
    uv run python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.wholebody.run_demo --no-detector
"""
from __future__ import annotations

# Both entry points use the same builder and command-line options.
from skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo import main


if __name__ == "__main__":
    main()
