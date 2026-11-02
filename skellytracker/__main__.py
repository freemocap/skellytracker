from __future__ import annotations

import argparse

from skellytracker.core.detectors.keypoint_detectors.aruco.run_demo import build_aruco_demo
from skellytracker.core.detectors.keypoint_detectors.charuco.run_demo import build_charuco_demo
from skellytracker.core.detectors.keypoint_detectors.dlc.run_demo import build_dlc_demo
from skellytracker.core.detectors.keypoint_detectors.mediapipe.run_demo import build_mediapipe_demo
from skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo import build_rtmpose_demo

_BUILDERS = {
    "mediapipe": build_mediapipe_demo,
    "rtmpose": build_rtmpose_demo,
    "aruco": build_aruco_demo,
    "charuco": build_charuco_demo,
    "dlc": build_dlc_demo,
}


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="skellytracker live demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  skellytracker\n"
            "  skellytracker --tracker rtmpose\n"
            "  skellytracker --tracker aruco --camera 1\n"
            "\n"
            "for tracker-specific options (model, provider, etc.) use the tracker's module directly:\n"
            "  python -m skellytracker.core.detectors.keypoint_detectors.rtmpose.run_demo --help"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available trackers and exit",
    )
    parser.add_argument(
        "--tracker",
        choices=list(_BUILDERS),
        default="mediapipe",
        help="tracker to run (default: mediapipe)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera index (default: 0)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="rotate incoming frames clockwise by this many degrees before tracking (default: 0)",
    )
    parser.add_argument("--sidecar", help="Converted DLC model YAML (required for --tracker dlc)")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default=None, help="DLC FP32 provider (default: cpu); use other trackers' modules for their provider options")
    args = parser.parse_args()

    if args.list:
        for name in _BUILDERS:
            print(name)
        return

    if args.tracker == "dlc":
        if not args.sidecar:
            parser.error("--tracker dlc requires --sidecar; use the dlc.run_demo module for video and confidence options")
        demo = build_dlc_demo(args.sidecar, provider=args.provider or "cpu")
    else:
        if args.sidecar or args.provider:
            parser.error("--sidecar and --provider apply to --tracker dlc; use other trackers' modules for model options")
        demo = _BUILDERS[args.tracker]()
    demo.run_webcam(camera_index=args.camera, rotation=args.rotate)


if __name__ == "__main__":
    cli_main()
