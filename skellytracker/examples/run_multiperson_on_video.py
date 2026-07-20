"""Run YOLOX → RTMPose multi-person tracking on a single video.

Unlike run_rtmpose_on_folder.py (single subject, cross-camera batching), this
tracks an arbitrary number of people within one camera's video, assigning
each a stable track ID across frames. Requires onnxruntime. Install with:
    uv sync --extra all-cpu       # CPU
    uv sync --extra all-trt       # NVIDIA GPU with TensorRT

Usage:
    python -m skellytracker.examples.run_multiperson_on_video \\
        /path/to/video.mp4 /path/to/output
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from tqdm import tqdm

import skellytracker.core.detectors.keypoint_detectors.rtmpose  # noqa: F401
import skellytracker.core.detectors.object_detectors.yolox  # noqa: F401
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.data_primitives import MultiPersonDataStore
from skellytracker.core.detectors.keypoint_detectors.rtmpose import RTMPoseDetectorConfig, RTMPoseKeypointDetector
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetector, YoloxPersonDetectorConfig
from skellytracker.core.sessions.onnx_session import OnnxSession, OnnxSessionConfig
from skellytracker.core.temporal_processing.multi_person_config import MultiPersonTrackingConfig
from skellytracker.core.tracker.multi_person_tracker import MultiPersonTracker
from skellytracker.core.tracker.person_track import PersonTrackState

_TRACK_COLORS = [
    (66, 135, 245), (245, 96, 66), (66, 245, 132), (245, 215, 66),
    (188, 66, 245), (66, 245, 224), (245, 66, 155),
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


def run_multiperson_on_video(
    video_path: Path,
    output_dir: Path,
    yolox_model: str = "yolox-m",
    rtmpose_model: str = "rtmw-x-l_256x192",
    max_persons: int = 6,
    annotated_video_path: Path | None = None,
) -> MultiPersonDataStore:
    session = OnnxSession.create(OnnxSessionConfig(
        batch_size=1,
        models=[
            YoloxPersonDetector.model_spec(yolox_model),
            RTMPoseKeypointDetector.model_spec(rtmpose_model),
        ],
    ))

    config = TrackerConfig(
        stages=[
            DetectionStageConfig(
                name="body",
                # max_detections=None keeps every person YOLOX finds each frame
                # (Tracker's single-person configs cap this at 1) — the object
                # detector runs every frame here, so no bbox_policy is set.
                object_detector=YoloxPersonDetectorConfig(max_detections=max_persons),
                keypoint_detectors=[RTMPoseDetectorConfig()],
            )
        ]
    )
    tracker = MultiPersonTracker.create(
        config, {"onnx": session}, MultiPersonTrackingConfig(min_hits=3, max_age=10)
    )

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer: cv2.VideoWriter | None = None
    if annotated_video_path is not None:
        annotated_video_path = Path(annotated_video_path)
        annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    store = MultiPersonDataStore()
    tracks: dict[int, PersonTrackState] = {}
    frame_number = 0

    try:
        for _ in tqdm(range(n_frames), desc=video_path.name, unit="frames"):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_ms = int(frame_number / fps * 1000)
            observation, tracks = tracker.process_image(frame, frame_number, tracks, timestamp_ms)
            store.add(observation)

            if video_writer is not None:
                for track_id, person_obs in observation.people.items():
                    color = _color_for_track(track_id)
                    bbox = person_obs.stages["body"].bounding_boxes
                    if bbox:
                        x1, y1, x2, y2 = int(bbox[0].x1), int(bbox[0].y1), int(bbox[0].x2), int(bbox[0].y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"id {track_id}", (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                video_writer.write(frame)

            frame_number += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        tracker.close()

    for track_id, arr in store.to_arrays().items():
        print(f"  track {track_id}: {arr.shape} ({arr.shape[0]} frames, {arr.shape[1]} keypoints)")
    store.save(output_dir)
    return store


if __name__ == "__main__":
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("multiperson_test.mp4")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("skellytracker_multiperson_output/")
    annotated = output / f"{video.stem}_annotated.mp4"

    run_multiperson_on_video(video, output, annotated_video_path=annotated)
