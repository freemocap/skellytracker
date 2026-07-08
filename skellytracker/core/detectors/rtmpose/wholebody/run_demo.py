"""Live RTMPose wholebody demo (133 keypoints).

Usage::

    uv run python -m skellytracker.core.detectors.rtmpose.wholebody.run_demo
    uv run python -m skellytracker.core.detectors.rtmpose.wholebody.run_demo --camera 1
    uv run python -m skellytracker.core.detectors.rtmpose.wholebody.run_demo --model rtmw-x-l_384x288
    uv run python -m skellytracker.core.detectors.rtmpose.wholebody.run_demo --provider cpu
    uv run python -m skellytracker.core.detectors.rtmpose.wholebody.run_demo --no-detector
"""
from __future__ import annotations

import argparse

from skellytracker.core.detectors.rtmpose.run_demo import build_rtmpose_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMPose wholebody live demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["rtmw-x-l_256x192", "rtmw-x-l_384x288", "rtmw-l-m_256x192"],
        default="rtmw-x-l_256x192",
    )
    parser.add_argument("--provider", choices=["cuda", "trt", "coreml", "directml", "cpu"], default=None)
    parser.add_argument("--no-detector", action="store_true")
    args = parser.parse_args()

    demo = build_rtmpose_demo(
        model_name=args.model,
        provider=args.provider,
        use_person_detector=not args.no_detector,
    )
    demo.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
