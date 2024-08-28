"""Top-level package for skellytracker"""

__package_name__ = "skellytracker"
__version__ = "v2024.08.1017"

__author__ = """Skelly FreeMoCap"""
__email__ = "info@freemocap.org"
__repo_owner_github_user_name__ = "freemocap"
__repo_url__ = (
    f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}"
)
__repo_issues_url__ = f"{__repo_url__}/issues"

# ruff: noqa: F401, E402

import sys
from pathlib import Path

print(f"Thank you for using {__package_name__}!")
print(f"This is printing from: {__file__}")
print(f"Source code for this package is available at: {__repo_url__}")

base_package_path = Path(__file__).parent
print(f"adding base_package_path: {base_package_path} : to sys.path")
sys.path.insert(0, str(base_package_path))  # add parent directory to sys.path

from skellytracker.system.default_paths import get_log_file_path
from skellytracker.system.logging_configuration import configure_logging

try:
    from skellytracker.trackers.mediapipe_tracker.mediapipe_holistic_tracker import (
        MediapipeHolisticTracker,
    )
    from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
        MediapipeModelInfo,
    )
except ModuleNotFoundError:
    print("To use mediapipe_holistic_tracker, install skellytracker[mediapipe]")
try:
    from skellytracker.trackers.yolo_tracker.yolo_tracker import YOLOPoseTracker
    from skellytracker.trackers.yolo_tracker.yolo_model_info import YOLOModelInfo
except ModuleNotFoundError:
    print("To use yolo_tracker, install skellytracker[yolo]")
try:
    from skellytracker.trackers.yolo_mediapipe_combo_tracker.yolo_mediapipe_combo_tracker import (
        YOLOMediapipeComboTracker,
    )
except ModuleNotFoundError:
    print(
        "To use yolo_mediapipe_combo_tracker, install skellytracker[mediapipe, yolo] or skellytracker[all]"
    )


configure_logging(log_file_path=str(get_log_file_path()))
