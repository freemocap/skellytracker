from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from skellytracker.trackers.vitpose_tracker.vitpose_annotator import VITPoseAnnotator
from skellytracker.trackers.vitpose_tracker.vitpose_detector import (
    VITPoseDetector,
    VITPoseDetectorConfig,
)


# ---------------------------
# Video discovery
# ---------------------------

def _natural_sort_key(p: Path) -> str:
    return p.as_posix()


def load_synced_videos(
    video_dir: str | Path,
    exts: Iterable[str] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    video_dir = Path(video_dir)
    vids = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in set(exts)]
    vids.sort(key=_natural_sort_key)
    if not vids:
        raise FileNotFoundError(f"No videos found in {video_dir} with extensions {exts}")
    return vids


# ============================================================
# Single-pass extraction + annotation per camera
# ============================================================

@dataclass(frozen=True)
class _ProcessWorkerArgs:
    video_path: str
    cam_index: int
    out_video_path: str
    n_frames: Optional[int]
    detector_config: Optional[VITPoseDetectorConfig]
    confidence_threshold: Optional[float]
    fill_with_nans: bool
    progress_queue: Optional[object]


def _process_one_camera(args: _ProcessWorkerArgs) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Single pass: detect, annotate, and write video simultaneously.
    
    Returns:
      cam_index
      cam_pts2d:     (F, K, 2) - xy coordinates only
      raw_keypoints: (F, K, 3) - includes confidence scores
    """
    detector = VITPoseDetector.create(args.detector_config)
    annotator = VITPoseAnnotator.create()
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        args.out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    try:
        n_frames = args.n_frames or 10**12

        pending_pts2d: list[np.ndarray] = []
        raw_kp_list: list[np.ndarray] = []

        frame_idx = 0

        while frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect
            obs = detector.detect(frame_number=frame_idx, image=frame)

            # Store keypoints
            pts2d = obs.to_2d_array(
                confidence_threshold=args.confidence_threshold,
                fill_with_nans=args.fill_with_nans,
            )
            pts2d = np.asarray(pts2d, dtype=np.float32)
            pending_pts2d.append(pts2d)
            raw_kp_list.append(np.asarray(obs.keypoints, dtype=np.float32))

            # Annotate and write immediately
            annotated = annotator.annotate_image(frame, obs)
            writer.write(annotated)

            if args.progress_queue is not None:
                args.progress_queue.put(1)

            frame_idx += 1

        if not pending_pts2d:
            empty_pts2d = np.zeros((0, 0, 2), dtype=np.float32)
            raw_keypoints = np.zeros((0, 0, 3), dtype=np.float32)
            return args.cam_index, empty_pts2d, raw_keypoints

        cam_pts2d = np.stack(pending_pts2d, axis=0)  # (F, K, 2)
        raw_keypoints = np.stack(raw_kp_list, axis=0)  # (F, K, 3)

        return args.cam_index, cam_pts2d, raw_keypoints

    finally:
        writer.release()
        cap.release()


def vitpose_process_synced_folder_mp(
    video_dir: str | Path,
    output_video_dir: str | Path,
    *,
    detector_config: VITPoseDetectorConfig | None = None,
    confidence_threshold: float | None = None,
    fill_with_nans: bool = True,
    enforce_equal_frame_counts: bool = False,
    use_min_frame_count: bool = True,
    num_workers: int | None = None,
    show_progress: bool = True,
):
    """
    Single-pass processing: extract 2D poses and write annotated videos simultaneously.
    
    Returns:
      points_2d:       (C, F, K, 2) - combined array for all cameras (xy only)
      raw_kp_list:     list length C, each (F, K, 3) - includes confidence
      video_paths:     list length C - input video paths
      out_video_paths: list length C - output annotated video paths
    """
    video_paths = load_synced_videos(video_dir)
    n_cams = len(video_paths)

    output_video_dir = Path(output_video_dir)
    output_video_dir.mkdir(parents=True, exist_ok=True)

    # Get frame counts
    frame_counts = []
    for p in video_paths:
        cap = cv2.VideoCapture(str(p))
        try:
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()

    if enforce_equal_frame_counts and len(set(frame_counts)) != 1:
        raise ValueError(f"Frame count mismatch across cameras: {frame_counts}")

    n_frames = None
    if use_min_frame_count:
        valid = [c for c in frame_counts if c > 0]
        n_frames = min(valid) if valid else None

    if num_workers is None:
        num_workers = min(n_cams, max(1, mp.cpu_count() - 1))

    ctx = mp.get_context("spawn")

    manager = ctx.Manager() if show_progress else None
    progress_queue = manager.Queue() if show_progress else None

    total = None if (n_frames is None) else (n_frames * n_cams)
    pbar = tqdm(total=total, desc="VITPose process (all cams)", unit="frame") if show_progress else None
    STOP = None

    def _progress_consumer():
        if progress_queue is None or pbar is None:
            return
        while True:
            msg = progress_queue.get()
            if msg is STOP:
                break
            pbar.update(int(msg))

    t = threading.Thread(target=_progress_consumer, daemon=True)
    if show_progress:
        t.start()

    out_video_paths = [
        output_video_dir / f"{p.stem}_vitpose_annotated.mp4"
        for p in video_paths
    ]

    worker_args = [
        _ProcessWorkerArgs(
            video_path=str(p),
            cam_index=i,
            out_video_path=str(out_video_paths[i]),
            n_frames=n_frames,
            detector_config=detector_config,
            confidence_threshold=confidence_threshold,
            fill_with_nans=fill_with_nans,
            progress_queue=progress_queue,
        )
        for i, p in enumerate(video_paths)
    ]

    results: list[Tuple[int, np.ndarray, np.ndarray]] = []
    try:
        with ctx.Pool(processes=num_workers) as pool:
            for cam_index, cam_pts2d, raw_kp in pool.imap_unordered(_process_one_camera, worker_args):
                results.append((cam_index, cam_pts2d, raw_kp))
    finally:
        if show_progress and progress_queue is not None:
            progress_queue.put(STOP)
        if show_progress:
            t.join(timeout=2)
        if pbar is not None:
            pbar.close()
        if manager is not None:
            manager.shutdown()

    results.sort(key=lambda x: x[0])

    cam_pts2d_list = [r[1] for r in results]
    raw_kp_list = [r[2] for r in results]

    # Enforce equal frame length across all outputs
    min_len = min(a.shape[0] for a in cam_pts2d_list) if cam_pts2d_list else 0
    cam_pts2d_list = [a[:min_len] for a in cam_pts2d_list]
    raw_kp_list = [a[:min_len] for a in raw_kp_list]

    points_2d = np.stack(cam_pts2d_list, axis=0)  # (C, F, K, 2)
    return points_2d, raw_kp_list, video_paths, out_video_paths


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    recording_root = Path(r"D:\2026-01-30-JTM")

    recordings_list = [recording_root/"2026-01-30_10-40-03_GMT-5_JTM_nih_1",
                       recording_root/"2026-01-30_10-57-13_GMT-5_JTM_nih_2",
                       recording_root/"2026-01-30_11-21-06_GMT-5_JTM_treadmill_1",
                       recording_root/"2026-01-30_11-32-56_GMT-5_JTM_treadmill_2"]

    for recording in recordings_list:

        tracker_name = "vitpose"
        path_to_recording_folder = Path(recording)
        path_to_synced_videos = path_to_recording_folder / "synchronized_videos"
        path_to_output_data = path_to_recording_folder / "output_data" / tracker_name

        path_to_save_2d_data = (
            path_to_output_data
            / "raw_data"
            / f"{tracker_name}_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
        )
        path_to_save_2d_data.parent.mkdir(parents=True, exist_ok=True)

        path_to_annotated_dir = path_to_recording_folder / "annotated_videos" / tracker_name

        # Single pass: extract + annotate together
        points_2d, raw_kp_list, video_paths, out_video_paths = vitpose_process_synced_folder_mp(
            path_to_synced_videos,
            path_to_annotated_dir,
            num_workers=1,
            show_progress=True,
        )
        
        # Save 2D data
        np.save(path_to_save_2d_data, points_2d)
        print(f"Saved 2D data: {points_2d.shape}")
        print(f"Annotated videos saved to: {path_to_annotated_dir}")
