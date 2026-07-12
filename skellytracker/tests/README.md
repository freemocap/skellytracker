# Tests

## Running

```bash
# All tests
uv run pytest skellytracker/tests

# Single file
uv run pytest skellytracker/tests/test_keypoints.py

# Fail on skips (confirm environment is fully set up)
uv run pytest skellytracker/tests --fail-on-skip
```

## Test files

| File | Requires network | Requires onnxruntime |
|------|:---:|:---:|
| `test_keypoints.py` | | |
| `test_temporal_processing.py` | | |
| `test_precomputed_object_detector.py` | | |
| `test_aruco_detector.py` | | |
| `test_yolox_detector.py` (model-free tests) | | |
| `test_yolox_detector.py` (inference tests) | ✓ | ✓ |
| `test_charuco_detector.py` | ✓ | |
| `test_mediapipe_detectors.py` | ✓ | |
| `test_rtmpose_detectors.py` | ✓ | ✓ |
| `test_data_store.py` | ✓ | |
| `test_mediapipe_video.py` | ✓ | |
| `test_rtmpose_video.py` | ✓ | ✓ |
| `test_charuco_video.py` | ✓ | |
| `test_aruco_video.py` | ✓ | |
| `test_yolox_video.py` | ✓ | ✓ |

## Writing tests for a new detector

Each detector gets two test files: one for single-image behaviour and one for multi-frame behaviour on the test video.

### Single-image tests (`test_<name>_detector.py`)

Cover the detector's contract in isolation — no video, no temporal state. Use the `test_image` or `charuco_test_image` fixture from `conftest.py` for real images, or generate a synthetic image in the file (see `test_aruco_detector.py`).

Typical test class structure:

```python
@pytest.fixture(scope="module")
def detector(session) -> MyDetector:
    return MyDetector.create(MyDetectorConfig(), session)

class TestMyDetector:
    def test_detect_returns_correct_shape(self, detector, test_image): ...
    def test_visibility_in_range(self, detector, test_image): ...
    def test_detect_blank_image(self, detector): ...       # NaN / empty output
    def test_at_least_one_detection(self, detector, test_image): ...
    def test_connections(self): ...                        # usually ()
```

If the detector requires `onnxruntime`, add `pytest.importorskip("onnxruntime", ...)` at the top of the file (before any onnxruntime imports) — this skips the entire file cleanly when onnxruntime is absent. See `test_rtmpose_detectors.py` for the pattern.

### Video tests (`test_<name>_video.py`)

Run the detector frame-by-frame over the test recording to catch issues that only appear across real sequential images.

- Use the `test_video_path` fixture from `conftest.py` (session-scoped, skips if recording is unavailable).
- Copy the `_load_video_frames(path, n_frames)` helper from any existing video test file.
- Use a `class`-scoped `@classmethod` fixture to run inference once and share results across all tests in the class.
- Frame count guidelines: **20–30 frames** for detectors that should trigger early in the video (charuco board); **15–20 frames** for person-detection tests.

For detectors **without** temporal state (charuco, aruco, yolox), call `detector.detect(frame)` directly in the fixture loop. For detectors **with** temporal state (mediapipe, rtmpose), use `Tracker.process_image(frame, frame_number=i, state=state)` and thread the returned state through each call.

Minimal video test template:

```python
class TestMyDetectorVideo:
    @pytest.fixture(scope="class")
    @classmethod
    def video_results(cls, test_video_path, my_session):
        frames = _load_video_frames(test_video_path, _N_FRAMES)
        if not frames:
            pytest.skip("No frames read from test video")
        detector = MyDetector.create(MyDetectorConfig(), my_session)
        return [detector.detect(frame) for frame in frames]

    def test_output_shape_consistent(self, video_results): ...
    def test_at_least_one_detection(self, video_results): ...
    def test_visibility_in_range(self, video_results): ...
    def test_undetected_points_are_nan(self, video_results): ...
```

### What the test video contains

The test recording (`freemocap_test_data`) has synchronized videos from three cameras. The single-camera fixture (`test_video_path`) picks the first alphabetically.

- **Frames 0–~30**: a charuco board is visible — use these for charuco and aruco tests.
- **Throughout**: a person is present — use these for body pose and person-detection tests.

## Skips

Tests that need a network connection download two images from Figshare at session start. Downloaded images are cached at `~/.cache/skellytracker/test_images/` so subsequent runs don't re-download. Tests skip (rather than fail) when the download fails or `onnxruntime` is not installed.

Video-based tests (`test_data_store.py`, `test_mediapipe_video.py`, `test_rtmpose_video.py`) use the freemocap test recording. The session start checks `~/freemocap_data/recordings/freemocap_test_data/synchronized_videos/` first; if absent, it downloads and extracts from GitHub releases to `~/.cache/skellytracker/test_data/`. Tests skip when the recording is unavailable locally and cannot be downloaded.
