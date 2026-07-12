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

## Skips

Tests that need a network connection download two images from Figshare at session start. Downloaded images are cached at `~/.cache/skellytracker/test_images/` so subsequent runs don't re-download. Tests skip (rather than fail) when the download fails or `onnxruntime` is not installed.

Video-based tests (`test_data_store.py`, `test_mediapipe_video.py`, `test_rtmpose_video.py`) use the freemocap test recording. The session start checks `~/freemocap_data/recordings/freemocap_test_data/synchronized_videos/` first; if absent, it downloads and extracts from GitHub releases to `~/.cache/skellytracker/test_data/`. Tests skip when the recording is unavailable locally and cannot be downloaded.
