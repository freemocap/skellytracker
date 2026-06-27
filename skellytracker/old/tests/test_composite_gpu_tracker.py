"""Test the CompositeGPU tracker with the freemocap test data set.

Test data location: ~/freemocap_data/recordings/freemocap_test_data/synchronized_videos
Expected: 3 mp4 files, 222 frames each.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

logger = logging.getLogger(__name__)

_TEST_DATA_DIR = Path.home() / "freemocap_data" / "recordings" / "freemocap_test_data" / "synchronized_videos"
_EXPECTED_FRAME_COUNT = 222


def _validate_test_data() -> list[Path]:
    """Validate the test data directory exists and contains expected videos.

    Returns list of video paths sorted alphabetically.
    """
    if not _TEST_DATA_DIR.exists():
        pytest.fail(
            f"Test data directory not found: {_TEST_DATA_DIR}\n"
            f"Expected: ~/freemocap_data/recordings/freemocap_test_data/synchronized_videos\n"
            f"Download the freemocap test data set before running this test."
        )

    mp4_files = sorted(_TEST_DATA_DIR.glob("*.mp4"))
    if len(mp4_files) == 0:
        pytest.fail(f"No .mp4 files found in {_TEST_DATA_DIR}")

    if len(mp4_files) != 3:
        pytest.fail(
            f"Expected 3 .mp4 files in {_TEST_DATA_DIR}, found {len(mp4_files)}: "
            f"{[f.name for f in mp4_files]}"
        )

    # Validate frame counts
    for mp4_path in mp4_files:
        cap = cv2.VideoCapture(str(mp4_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count != _EXPECTED_FRAME_COUNT:
            pytest.fail(
                f"Expected {_EXPECTED_FRAME_COUNT} frames in {mp4_path.name}, "
                f"found {frame_count}"
            )

    return mp4_files


def _read_frames(video_paths: list[Path], max_frames: int = 5) -> list[np.ndarray]:
    """Read the first `max_frames` from each video."""
    frames = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return frames


# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TEST_DATA_DIR.exists(),
    reason=f"Test data directory not found: {_TEST_DATA_DIR}",
)
class TestCompositeGPUTrackerWithVideos:
    """Integration tests using real freemocap test videos."""

    def test_validate_test_data(self):
        """Verify test data directory structure and video properties."""
        mp4_files = _validate_test_data()
        assert len(mp4_files) == 3

        for mp4_path in mp4_files:
            assert mp4_path.exists()
            cap = cv2.VideoCapture(str(mp4_path))
            assert cap.isOpened(), f"Could not open {mp4_path.name}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert w > 0 and h > 0, f"Invalid dimensions for {mp4_path.name}: {w}x{h}"
            cap.release()

    def test_read_frames(self):
        """Verify we can read frames from all test videos."""
        mp4_files = _validate_test_data()
        frames = _read_frames(mp4_files, max_frames=5)
        assert len(frames) == 15  # 3 videos × 5 frames

        for i, frame in enumerate(frames):
            assert frame is not None, f"Frame {i} is None"
            assert frame.ndim == 3, f"Frame {i} has ndim={frame.ndim}"
            assert frame.shape[-1] == 3, f"Frame {i} is not 3-channel"

    def test_config_and_observation_roundtrip(self):
        """Verify config construction and observation creation without GPU."""
        from skellytracker.old.composite_gpu_tracker.composite_gpu_config import (
            CompositeGPUDetectorConfig,
            CompositeGPUTrackerConfig,
        )
        from skellytracker.old.composite_gpu_tracker.composite_gpu_observation import (
            CompositeGPUObservation,
        )
        from skellytracker.old.composite_gpu_tracker.names_and_connections import (
            RTMO_HYBRID_DEFINITION,
        )

        # Config construction
        config = CompositeGPUTrackerConfig()
        assert config.detector_config is not None
        assert config.annotator_config is not None

        detector_config = CompositeGPUDetectorConfig()
        assert detector_config.tracker_type == "rtmo_hybrid"
        assert detector_config.session_config.execution_provider == "cuda"

        # Observation construction with synthetic data
        body_kpts = np.random.randn(1, 17, 2).astype(np.float64)
        body_scores = np.random.rand(1, 17).astype(np.float32) * 0.5 + 0.5
        hands_kpts = np.random.randn(1, 42, 2).astype(np.float64)
        hands_scores = np.random.rand(1, 42).astype(np.float32) * 0.5 + 0.5
        face_kpts = np.random.randn(1, 68, 2).astype(np.float64)
        face_scores = np.random.rand(1, 68).astype(np.float32) * 0.5 + 0.5

        obs = CompositeGPUObservation.from_detection_results(
            frame_number=0,
            image_size=(720, 1280),
            body_keypoints=body_kpts,
            body_scores=body_scores,
            hands_keypoints=hands_kpts,
            hands_scores=hands_scores,
            face_keypoints=face_kpts,
            face_scores=face_scores,
        )

        assert obs.points.n_points == 127
        assert len(obs.points.names) == 127
        assert obs.points.xyz.shape == (127, 3)
        assert obs.points.visibility.shape == (127,)

        # Verify point names match the composition definition
        assert obs.points.names == RTMO_HYBRID_DEFINITION.tracked_points

        # Check component slice boundaries
        names = obs.points.names
        assert names[0] == "nose"  # body first point
        assert names[16] == "right_ankle"  # body last point
        assert names[17].startswith("right_hand_")  # right hand starts
        assert names[38].startswith("left_hand_")  # left hand starts
        assert names[59].startswith("face_")  # face starts

    def test_yaml_composition_loads_correctly(self):
        """Verify the hybrid YAML composes correctly with all component definitions."""
        from skellytracker.old.composite_gpu_tracker.names_and_connections import (
            RTMO_BODY_17_DEFINITION,
            RTMO_HYBRID_DEFINITION,
        )

        assert RTMO_BODY_17_DEFINITION.num_tracked_points == 17
        assert len(RTMO_BODY_17_DEFINITION.tracked_points) == 17
        assert RTMO_BODY_17_DEFINITION.name == "rtmo_body_17"

        assert RTMO_HYBRID_DEFINITION.num_tracked_points == 127
        assert len(RTMO_HYBRID_DEFINITION.tracked_points) == 127
        assert RTMO_HYBRID_DEFINITION.name == "rtmo_hybrid"

        # Verify connection indices span all components
        conn_indices = RTMO_HYBRID_DEFINITION.connection_indices()
        assert len(conn_indices) > 0

    def test_annotator_no_crash_with_empty_observation(self):
        """Verify annotator handles empty observations gracefully."""
        from skellytracker.old.composite_gpu_tracker.composite_gpu_annotator import (
            CompositeGPUImageAnnotator,
        )
        from skellytracker.old.composite_gpu_tracker.composite_gpu_config import (
            CompositeGPUImageAnnotatorConfig,
        )

        config = CompositeGPUImageAnnotatorConfig()
        annotator = CompositeGPUImageAnnotator.create(config)

        # Empty observation
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = annotator.annotate_image(image, observation=None)
        assert annotated is image  # unchanged

    def test_roi_crop_utils(self):
        """Verify ROI crop utilities produce valid geometry."""
        from skellytracker.old.composite_gpu_tracker.roi_crop_utils import (
            compute_face_crop_params,
            compute_hand_crop_size,
            compute_square_roi,
            hand_bbox_diagonal,
            smooth_roi_params,
        )

        # Square ROI
        roi = compute_square_roi(center_x=320, center_y=240, size=200, image_w=640, image_h=480)
        assert roi.x >= 0
        assert roi.y >= 0
        assert roi.width > 0
        assert roi.height > 0
        assert roi.x + roi.width <= 640
        assert roi.y + roi.height <= 480

        # Hand crop size
        size = compute_hand_crop_size(arm_length=100.0, image_h=720)
        assert size > 0

        # Smoothing cold start
        result = smooth_roi_params(raw_cx=100.0, raw_cy=200.0, raw_size=150.0, prev_smoothed=None)
        assert result == (100.0, 200.0, 150.0)

        # Smoothing with previous
        result = smooth_roi_params(
            raw_cx=110.0, raw_cy=190.0, raw_size=155.0,
            prev_smoothed=(100.0, 200.0, 150.0), alpha=0.5,
        )
        assert result == (105.0, 195.0, 152.5)

        # Face crop from visible points
        pts = np.array([[300.0, 200.0], [340.0, 200.0], [320.0, 240.0]])
        params = compute_face_crop_params(visible_head_points=pts)
        assert params is not None
        center, crop_size = params
        assert crop_size > 0

        # Hand bbox diagonal
        landmarks = np.array([[100.0, 200.0, 0.0], [130.0, 230.0, 0.0]])
        diag = hand_bbox_diagonal(landmarks)
        assert diag > 0
