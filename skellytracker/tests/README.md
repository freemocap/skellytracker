# Tests

## Explicit CPU and CUDA reference validation

From the repository root, the normal entry points are:

```powershell
uv run --extra all-cuda poe test-reference
uv run --extra all-cuda poe benchmark-reference
```

These find the prepared `freemocap_test_data` recording in `~/freemocap_data/testing/`
automatically. The test command runs CPU and CUDA explicitly, plus MediaPipe checks.
The benchmark command compares CPU and CUDA across all cameras. If the environment
is already activated and configured with `all-cuda`, `poe test-reference` and
`poe benchmark-reference` work directly. The `--extra all-cuda` on `uv run` preserves
the inference dependencies when uv synchronizes the environment.

The commands below describe optional overrides, not required setup.

On an NVIDIA machine, `uv sync --locked --extra all-cuda` installs the GPU
runtime, which also supports CPU execution. Do not combine `all-cpu` and
`all-cuda` in the same environment. Installation is a separate environment step.

The YOLOX and RTMPose image/video session fixtures accept a provider matrix.
With no environment setting they retain automatic provider selection. To test
both explicitly in PowerShell, from the repository root:

```powershell
$env:SKELLYTRACKER_TEST_PROVIDERS = "cpu,cuda"
$env:SKELLYTRACKER_TEST_VIDEO_DIR = "C:\path\to\reference\synchronized_videos"
.venv/Scripts/python.exe -m pytest skellytracker/tests/test_rtmpose_video.py skellytracker/tests/test_yolox_video.py -v --fail-on-skip
```

These existing video tests use the first camera alphabetically and their usual
15/20-frame samples. They now check the requested provider is actually first in
each loaded model session. A failed CUDA initialization cannot be counted as a
successful CPU fallback. CUDA may still execute unsupported/shape operations on
CPU; these checks do not profile individual graph nodes. When a provider matrix
is explicitly requested, a missing runtime/provider fails test setup rather than
silently skipping it. MediaPipe uses its own backend and is not a CUDA ONNX test.

Reference assets are loaded only when a fixture needs them. A supplied video
directory is read directly, without downloading another recording. Image tests
still use the published reference images and their existing cache.

For a repeatable timing comparison across **all cameras**:

```powershell
.venv/Scripts/python.exe -m skellytracker.tests.benchmark_providers $env:SKELLYTRACKER_TEST_VIDEO_DIR --providers cpu cuda --frames 30 --repeats 3
```

The benchmark uses the same decoded initial frames, FP32 models and batch size 1
for both routes, with fresh tracking state for each camera/repetition. Setup and
runtime warm-up are reported separately; three additional tracker warm-up frames
are excluded from latency. Reports include per-camera/per-run mean, median, p95
and throughput, active session providers, and CPU/CUDA coordinate agreement.
This measures `Tracker.process_image`, including preprocessing and tracking, but
excludes decoding, annotation, and encoding. It is not a full application FPS
measurement, a batched multi-camera benchmark, or a ground-truth accuracy score.
Run on an otherwise idle machine; reverse provider order for a confirmation run
if thermal/load effects are suspected. No hardware-dependent speed threshold is
used as a correctness assertion.

This workload runs YOLOX-m and RTMW-x-l on every frame, without reusing tracked
crops. It measures individual camera images processed sequentially, not synchronized
three-camera frame sets. Applications that reuse crops between person detections
have a different workload. Reports include each model's ONNX thread settings;
CPU defaults to automatic threading with idle spinning disabled, while accelerator
host threading remains at one. `OnnxSessionConfig.intra_op_num_threads` can set a
smaller thread budget for applications running multiple CPU workers.

JSON and first-pass keypoint arrays go to ignored `.test-artifacts/provider-benchmark/`.
The report has `completed: true` only when all requested providers finish.
CPU-only package installation should also be covered in a separate environment
or CI job; CPU execution within the GPU package does not validate that packaging.

After testing, clear the overrides if returning to automatic selection:

```powershell
Remove-Item Env:SKELLYTRACKER_TEST_PROVIDERS
Remove-Item Env:SKELLYTRACKER_TEST_VIDEO_DIR
```

## Spine mapping regression (2026-09-24)

`test_spine_midpoint_mapping.py` loads the shipped RTMPose and MediaPipe mappings.
The intended spine chain is hip midpoint -> mean of both hips and shoulders ->
shoulder midpoint -> ear midpoint, with no anatomical offsets on those endpoints.
Previously RTMPose mapped chest_center to the shoulder midpoint (collapsing the
thoracic endpoints); MediaPipe used an anatomical offset instead. Both are fixed.
Clavicular offsets remain separate. Tests cover the midpoint values, transformed
poses, measured/constructed classification, and missing source measurements.

Focused validation: 24 mapping tests passed, plus Ruff. For mapping-only tests use
`--noconftest` to avoid this repository's session-wide image/video downloads.

Separate Forge follow-up, not fixed here: a read-only synthetic diagnostic with
straight vertical spine endpoints and the existing constructed thoracic landmarks
produced a thoracic rigid-fit tilt of 8.77 degrees and moved its origin about 16 mm.
Lumbar and cervical axes stayed vertical in that diagnostic. Forge currently fits
the thoracic pose to its off-axis landmarks as well as the spine endpoints; the
constructed landmarks can therefore alter the declared primary direction. Add a
Forge regression and preserve the intended spine endpoints/direction before
claiming the real-time spine issue is resolved. This is synthetic evidence, not a
measurement of the user's live scene. Existing prepared Parquet is unchanged and
contains the old mapping/fit; its previous hierarchy report is not a post-fix result.

## Running

```bash
# All tests
uv run pytest
# or: uv run poe test

# Exclude slow video tests (faster feedback for small changes)
uv run pytest -m "not video"
# or: uv run poe test-fast

# Only video tests
uv run pytest -m video

# Single file
uv run pytest skellytracker/tests/test_keypoints.py

# Fail on skips (confirm environment is fully set up)
uv run pytest --fail-on-skip
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

### Video tests (`test_<name>_video.py`, marked `video`)

Run the detector frame-by-frame over the test recording to catch issues that only appear across real sequential images.

- Use the `test_video_path` fixture from `conftest.py` (session-scoped, skips if recording is unavailable).
- Copy the `_load_video_frames(path, n_frames)` helper from any existing video test file.
- Use a `class`-scoped `@classmethod` fixture to run inference once and share results across all tests in the class.
- Frame count guidelines: **20–30 frames** for detectors that should trigger early in the video (charuco board); **15–20 frames** for person-detection tests.

For detectors **without** temporal state (charuco, aruco, yolox), call `detector.detect(frame)` directly in the fixture loop. For detectors **with** temporal state (mediapipe, rtmpose), use `Tracker.process_image(frame, frame_number=i, state=state)` and thread the returned state through each call.

All video test files must set `pytestmark = pytest.mark.video` at module level (after the imports) so they are excluded by `pytest -m "not video"`.

Minimal video test template:

```python
pytestmark = pytest.mark.video

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
