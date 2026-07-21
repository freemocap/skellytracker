"""Tests for process_video and process_folder batch processing utilities.

Validates that process_video writes a correctly shaped .npy array and that
process_folder handles a directory of videos. Uses the mediapipe body detector
since it requires no GPU and has a predictable keypoint count (33).

Skips automatically when the test recording is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

import skellytracker.core.detectors.keypoint_detectors.mediapipe  # noqa: F401
from skellytracker.core import (
    DetectionStageConfig,
    Tracker,
    TrackerConfig,
    process_folder,
    process_video,
)
from skellytracker.core.detectors.keypoint_detectors.mediapipe import (
    MediapipePoseDetectorConfig,
    MediapipePoseModelComplexity,
    MediaPipeSession,
    MediaPipeSessionConfig,
)

pytestmark = pytest.mark.video

_N_KEYPOINTS = 33


@pytest.fixture(scope="module")
def mediapipe_session() -> MediaPipeSession:
    session = MediaPipeSession.create(MediaPipeSessionConfig())
    yield session
    session.close()


@pytest.fixture(scope="module")
def mediapipe_tracker(mediapipe_session: MediaPipeSession) -> Tracker:
    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="body",
                keypoint_detectors=[
                    MediapipePoseDetectorConfig(
                        model_complexity=MediapipePoseModelComplexity.LITE
                    )
                ],
            )
        ]
    )
    return Tracker.create(config, {"mediapipe": mediapipe_session})


class TestProcessVideo:
    def test_saves_npy_file(self, tmp_path, test_video_path, mediapipe_tracker):
        process_video(
            tracker=mediapipe_tracker,
            annotator=None,
            input_path=test_video_path,
            output_dir=tmp_path,
            show_progress=False,
        )
        output_file = tmp_path / f"{test_video_path.stem}.npy"
        assert output_file.exists(), f"Expected .npy at {output_file}"
        arr = np.load(output_file)
        assert arr.ndim == 3
        assert arr.shape[1] == _N_KEYPOINTS
        assert arr.shape[2] == 3

    def test_array_shape_matches_datastore(self, tmp_path, test_video_path, mediapipe_tracker):
        store = process_video(
            tracker=mediapipe_tracker,
            annotator=None,
            input_path=test_video_path,
            output_dir=tmp_path,
            show_progress=False,
        )
        arr = store.to_array()
        assert arr.ndim == 3
        assert arr.shape[1] == _N_KEYPOINTS

    def test_frame_count_matches_observations(self, tmp_path, test_video_path, mediapipe_tracker):
        store = process_video(
            tracker=mediapipe_tracker,
            annotator=None,
            input_path=test_video_path,
            output_dir=tmp_path,
            show_progress=False,
        )
        assert len(store.observations) > 0
        arr = store.to_array()
        assert arr.shape[0] == len(store.observations)

    def test_saves_json_file(self, tmp_path, test_video_path, mediapipe_tracker):
        process_video(
            tracker=mediapipe_tracker,
            annotator=None,
            input_path=test_video_path,
            output_dir=tmp_path,
            fmt="json",
            show_progress=False,
        )
        output_file = tmp_path / f"{test_video_path.stem}.json"
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestProcessFolder:
    def test_processes_all_videos(self, tmp_path, sync_videos_dir, mediapipe_tracker):
        video_paths = sorted(sync_videos_dir.glob("*.mp4"))
        if not video_paths:
            pytest.skip(f"No .mp4 files in {sync_videos_dir}")

        results = process_folder(
            tracker=mediapipe_tracker,
            annotator=None,
            video_dir=sync_videos_dir,
            output_dir=tmp_path / "output",
            show_progress=False,
        )
        assert len(results) == len(video_paths)

    def test_all_output_files_exist(self, tmp_path, sync_videos_dir, mediapipe_tracker):
        video_paths = sorted(sync_videos_dir.glob("*.mp4"))
        if not video_paths:
            pytest.skip(f"No .mp4 files in {sync_videos_dir}")

        output_dir = tmp_path / "output"
        process_folder(
            tracker=mediapipe_tracker,
            annotator=None,
            video_dir=sync_videos_dir,
            output_dir=output_dir,
            show_progress=False,
        )
        for video_path in video_paths:
            assert (output_dir / f"{video_path.stem}.npy").exists()

    def test_consistent_keypoint_count_across_cameras(self, tmp_path, sync_videos_dir, mediapipe_tracker):
        video_paths = sorted(sync_videos_dir.glob("*.mp4"))
        if len(video_paths) < 2:
            pytest.skip("Need at least 2 camera videos to check consistency")

        output_dir = tmp_path / "output"
        results = process_folder(
            tracker=mediapipe_tracker,
            annotator=None,
            video_dir=sync_videos_dir,
            output_dir=output_dir,
            show_progress=False,
        )
        point_counts = {stem: store.to_array().shape[1] for stem, store in results.items()}
        unique_counts = set(point_counts.values())
        assert len(unique_counts) == 1, f"Inconsistent keypoint counts across cameras: {point_counts}"

    def test_shapes_match_single_camera(self, tmp_path, sync_videos_dir, mediapipe_tracker):
        """process_folder produces the same array shape as the single-camera process_video path."""
        video_paths = sorted(sync_videos_dir.glob("*.mp4"))
        if not video_paths:
            pytest.skip(f"No .mp4 files in {sync_videos_dir}")
        video_path = video_paths[0]

        single_store = process_video(
            tracker=mediapipe_tracker,
            annotator=None,
            input_path=video_path,
            output_dir=tmp_path / "single",
            show_progress=False,
        )

        import shutil
        single_dir = tmp_path / "single_cam_dir"
        single_dir.mkdir()
        shutil.copy(video_path, single_dir / video_path.name)

        folder_results = process_folder(
            tracker=mediapipe_tracker,
            annotator=None,
            video_dir=single_dir,
            output_dir=tmp_path / "folder",
            show_progress=False,
        )
        folder_store = folder_results[video_path.stem]

        single_arr = single_store.to_array()
        folder_arr = folder_store.to_array()
        assert single_arr.shape == folder_arr.shape, (
            f"Shape mismatch: single={single_arr.shape}, folder={folder_arr.shape}"
        )
