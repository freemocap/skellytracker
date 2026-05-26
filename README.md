# skellytracker

The tracking backend for freemocap. Collects different pose estimation tools and aggregates them using a consistent API built around the **Tracker → Detector / Annotator / Recorder** pattern.

## Quick start

```bash
# Install with GPU extras
uv sync --extra recommended

# Run the webcam demo (press h for controls)
python -m skellytracker

# Build the Rust native module (after changes to skellytracker-rust/)
uv run poe rebuild
```

## Trackers

| Tracker | Backend | Hotkey | Rust? | Notes |
|---------|---------|--------|-------|-------|
| MediaPipe Holistic | Python `mediapipe` (C++ TFLite) | `m` | ✅ Phase 1 | Reverse PyO3 bridge — 211-point full-body |
| RTMPose | ONNX Runtime (CUDA) | `r` | ✅ Phase 2 | 133-keypoint whole-body, ~25ms/frame GPU |
| Charuco | OpenCV ArUco | `c` | ✅ Complete | Board detection + calibration |
| BrightestPoint | OpenCV luminance | `b` | ✅ Complete | Simple blob tracker |
| CompositeGPU | ONNX Runtime (CUDA) | — | ⬜ Planned | Multi-model GPU pipeline |

**Demo hotkeys:** `p` toggles Rust ↔ Python for the current tracker. `h` shows full controls.

## Architecture

See [rearchitecture-docs/skellytracker-architecture/](rearchitecture-docs/skellytracker-architecture/) for the full Rust re-architecture documentation, including:

- [Tracker trait + PointCloud design](rearchitecture-docs/skellytracker-architecture/01-tracker-trait-architecture.md)
- [PyO3 bridge pattern](rearchitecture-docs/skellytracker-architecture/03-pyo3-bridge-pattern.md)
- [Hot-swappable backends](rearchitecture-docs/skellytracker-architecture/04-hot-swappable-backend.md)
- [Lessons learned (21 rules)](rearchitecture-docs/skellytracker-architecture/05-lessons-learned.md)
- Per-tracker translation docs (BrightestPoint, Charuco, RTMPose, MediaPipe)

## GPU setup

For help setting up your GPU for ONNX Runtime trackers, see the [GPU_SETUP_GUIDE](GPU_SETUP_GUIDE.md).

## Contributing

See [CLAUDE.md](CLAUDE.md) for development commands and architecture details.

Pull requests welcome — fork the repo, branch from `main`, add tests, lint with `ruff`, and open a PR.
