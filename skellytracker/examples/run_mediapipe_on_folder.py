from pathlib import Path
import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401 — registers detectors
from skellytracker.core import DetectionStageConfig, Tracker, TrackerConfig, process_folder
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediapipePoseDetectorConfig, MediaPipeSession, MediaPipeSessionConfig
)


def run_mediapipe_on_folder(synchronized_videos_folder: Path, output_folder: Path) -> None:
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    tracker = Tracker.create(
        TrackerConfig(stages=[DetectionStageConfig(name="body",
    keypoint_detectors=[MediapipePoseDetectorConfig()])]),
        {"mediapipe": session}
    )
    process_folder(tracker, None, synchronized_videos_folder, output_folder, profile=True)

if __name__ == "__main__":
    import sys
    synchronized_videos_folder = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/synchronized_videos/")
    output_folder = Path("/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/skellytracker_test/")

    if len(sys.argv) > 1:
        synchronized_videos_folder = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_folder = Path(sys.argv[2])

    run_mediapipe_on_folder(synchronized_videos_folder, output_folder)
