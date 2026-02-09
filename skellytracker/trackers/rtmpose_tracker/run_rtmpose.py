from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import threading

from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import (
    RTMPoseDetector,
    RTMPoseDetectorConfig,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_annotator import RTMPoseImageAnnotator


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
# 1) EXTRACTION (MP): points_2d + raw keypoints/scores in memory
# ============================================================

@dataclass(frozen=True)
class _ExtractWorkerArgs:
    video_path: str
    cam_index: int
    n_frames: Optional[int]
    detector_config: Optional[RTMPoseDetectorConfig]
    confidence_threshold: Optional[float]
    fill_with_nans: bool
    progress_queue: Optional[object]  # Manager.Queue proxy


def _extract_one_camera(args: _ExtractWorkerArgs) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cam_index
      cam_pts2d:     (F, K, 2)
      raw_keypoints: (F, Pmax, K, 2) NaN-padded
      raw_scores:    (F, Pmax, K)    NaN-padded
      n_people:      (F,)
    """
    detector = RTMPoseDetector.create(args.detector_config)
    cap = cv2.VideoCapture(args.video_path)

    try:
        n_frames = args.n_frames or 10**12

        pending_pts2d: list[np.ndarray] = []
        raw_kp_list: list[np.ndarray] = []  # each (P,K,2)
        raw_sc_list: list[np.ndarray] = []  # each (P,K)
        n_people_list: list[int] = []

        frame_idx = 0
        Pmax = 0
        K: Optional[int] = None

        while frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break

            obs = detector.detect(frame_number=frame_idx, image=frame)

            # --- Derived analysis array (ABC) ---
            pts2d = obs.to_2d_array(
                confidence_threshold=args.confidence_threshold,
                fill_with_nans=args.fill_with_nans,
            )
            pts2d = np.asarray(pts2d, dtype=np.float32)  # (K,2)
            pending_pts2d.append(pts2d)

            # --- Raw outputs (for annotation) ---
            kp = np.asarray(obs.keypoints, dtype=np.float32)  # expected (P,K,2)
            sc = np.asarray(obs.scores, dtype=np.float32)     # expected (P,K)

            # robust to weird/no-detection outputs
            if kp.ndim != 3:
                kp = np.zeros((0, 0, 2), dtype=np.float32)
            if sc.ndim != 2:
                sc = np.zeros((0, 0), dtype=np.float32)

            P = int(kp.shape[0])
            if P > 0 and K is None:
                K = int(kp.shape[1])

            Pmax = max(Pmax, P)
            n_people_list.append(P)
            raw_kp_list.append(kp)
            raw_sc_list.append(sc)

            if args.progress_queue is not None:
                args.progress_queue.put(1)

            frame_idx += 1

        if not pending_pts2d:
            empty_pts2d = np.zeros((0, 0, 2), dtype=np.float32)
            raw_keypoints = np.zeros((0, 0, 0, 2), dtype=np.float32)
            raw_scores = np.zeros((0, 0, 0), dtype=np.float32)
            n_people = np.zeros((0,), dtype=np.int16)
            return args.cam_index, empty_pts2d, raw_keypoints, raw_scores, n_people

        cam_pts2d = np.stack(pending_pts2d, axis=0)  # (F,K,2)
        F, K2, _ = cam_pts2d.shape
        if K is None:
            # No detections ever; fall back to K from pts2d
            K = K2

        raw_keypoints = np.full((F, Pmax, K, 2), np.nan, dtype=np.float32)
        raw_scores = np.full((F, Pmax, K), np.nan, dtype=np.float32)
        n_people = np.asarray(n_people_list, dtype=np.int16)

        for f in range(F):
            P = int(n_people[f])
            if P <= 0:
                continue
            kp = raw_kp_list[f]
            sc = raw_sc_list[f]
            # fill up to min(K, kpK) just in case
            k_fill = min(K, kp.shape[1]) if kp.shape[0] > 0 else 0
            if k_fill > 0:
                raw_keypoints[f, :P, :k_fill, :] = kp[:, :k_fill, :]
                raw_scores[f, :P, :k_fill] = sc[:, :k_fill]

        return args.cam_index, cam_pts2d, raw_keypoints, raw_scores, n_people

    finally:
        cap.release()


def rtmpose_2d_from_synced_folder_mp(
    video_dir: str | Path,
    *,
    detector_config: RTMPoseDetectorConfig | None = None,
    confidence_threshold: float | None = None,
    fill_with_nans: bool = True,
    enforce_equal_frame_counts: bool = False,
    use_min_frame_count: bool = True,
    num_workers: int | None = None,
    show_progress: bool = True,
):
    """
    Returns:
      points_2d:     (C, F, K, 2)
      raw_kp_list:   list length C, each (F, Pmax_cam, K, 2)
      raw_sc_list:   list length C, each (F, Pmax_cam, K)
      n_people_list: list length C, each (F,)
      video_paths:   list length C
    """
    video_paths = load_synced_videos(video_dir)
    n_cams = len(video_paths)

    # frame counts in parent
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
    pbar = tqdm(total=total, desc="RTMPose extract (all cams)", unit="frame") if show_progress else None
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

    worker_args = [
        _ExtractWorkerArgs(
            video_path=str(p),
            cam_index=i,
            n_frames=n_frames,
            detector_config=detector_config,
            confidence_threshold=confidence_threshold,
            fill_with_nans=fill_with_nans,
            progress_queue=progress_queue,
        )
        for i, p in enumerate(video_paths)
    ]

    results: list[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        with ctx.Pool(processes=num_workers) as pool:
            for cam_index, cam_pts2d, raw_kp, raw_sc, n_people in pool.imap_unordered(_extract_one_camera, worker_args):
                results.append((cam_index, cam_pts2d, raw_kp, raw_sc, n_people))
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
    raw_kp_list    = [r[2] for r in results]
    raw_sc_list    = [r[3] for r in results]
    n_people_list  = [r[4] for r in results]

    # enforce equal frame length across cams for ALL outputs
    min_len = min(a.shape[0] for a in cam_pts2d_list) if cam_pts2d_list else 0
    cam_pts2d_list = [a[:min_len] for a in cam_pts2d_list]
    raw_kp_list    = [a[:min_len] for a in raw_kp_list]
    raw_sc_list    = [a[:min_len] for a in raw_sc_list]
    n_people_list  = [a[:min_len] for a in n_people_list]

    points_2d = np.stack(cam_pts2d_list, axis=0)  # (C,F,K,2)
    return points_2d, raw_kp_list, raw_sc_list, n_people_list, video_paths


# ============================================================
# 2) ANNOTATION (MP): write annotated videos per camera
# ============================================================

@dataclass(frozen=True)
class _AnnotWorkerArgs:
    cam_index: int
    video_path: str
    out_path: str
    keypoints: np.ndarray  # (F,Pmax,K,2)
    scores: np.ndarray     # (F,Pmax,K)
    n_people: np.ndarray   # (F,)
    n_frames: int
    progress_queue: Optional[object]


def _annotate_one_camera(args: _AnnotWorkerArgs) -> int:
    annotator = RTMPoseImageAnnotator.create()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        args.out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    try:
        for f_idx in range(args.n_frames):
            ok, frame = cap.read()
            if not ok:
                break

            P = int(args.n_people[f_idx])
            kp = args.keypoints[f_idx, :P, :, :]
            sc = args.scores[f_idx, :P, :]

            annotated = annotator.annotate_image_from_keypoints_and_scores(frame, kp, sc)
            writer.write(annotated)

            if args.progress_queue is not None:
                args.progress_queue.put(1)

        return args.cam_index

    finally:
        writer.release()
        cap.release()


def annotate_synced_videos_from_raw_mp(
    *,
    video_paths: list[Path],
    raw_kp_list: list[np.ndarray],
    raw_sc_list: list[np.ndarray],
    n_people_list: list[np.ndarray],
    out_dir: str | Path,
    num_workers: int | None = None,
    show_progress: bool = True,
) -> list[Path]:
    """
    Writes one annotated video per camera, multiprocessing per camera.
    Returns list of output video paths (in cam order).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cams = len(video_paths)
    if num_workers is None:
        num_workers = min(n_cams, max(1, mp.cpu_count() - 1))

    # shared frame count across cams (should already be equalized)
    n_frames = min(int(k.shape[0]) for k in raw_kp_list) if raw_kp_list else 0

    ctx = mp.get_context("spawn")
    manager = ctx.Manager() if show_progress else None
    progress_queue = manager.Queue() if show_progress else None

    total = n_frames * n_cams if show_progress else None
    pbar = tqdm(total=total, desc="Annotate videos (all cams)", unit="frame") if show_progress else None
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

    out_paths: list[Path] = []
    worker_args: list[_AnnotWorkerArgs] = []
    for cam_i, vid_path in enumerate(video_paths):
        out_path = out_dir / f"{Path(vid_path).stem}_rtmpose_raw_annotated.mp4"
        out_paths.append(out_path)
        worker_args.append(
            _AnnotWorkerArgs(
                cam_index=cam_i,
                video_path=str(vid_path),
                out_path=str(out_path),
                keypoints=raw_kp_list[cam_i][:n_frames],
                scores=raw_sc_list[cam_i][:n_frames],
                n_people=n_people_list[cam_i][:n_frames],
                n_frames=n_frames,
                progress_queue=progress_queue,
            )
        )

    try:
        with ctx.Pool(processes=num_workers) as pool:
            # we don't really need the returns, but it helps catch worker errors
            for _ in pool.imap_unordered(_annotate_one_camera, worker_args):
                pass
    finally:
        if show_progress and progress_queue is not None:
            progress_queue.put(STOP)
        if show_progress:
            t.join(timeout=2)
        if pbar is not None:
            pbar.close()
        if manager is not None:
            manager.shutdown()

    return out_paths


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    tracker_name = "rtmpose"

    recording_root = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")

    recordings_list = [recording_root/"sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
                       recording_root/"sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
                       recording_root/"sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
                       recording_root/"sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
                       recording_root/"sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"]

    for path_to_recording_folder in recordings_list:
        path_to_recording_folder = Path(path_to_recording_folder)
        path_to_synced_videos = path_to_recording_folder / "synchronized_videos"
        path_to_output_data = path_to_recording_folder / "output_data" / tracker_name

        path_to_save_2d_data = (
            path_to_output_data
            / "raw_data"
            / f"{tracker_name}_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
        )
        path_to_save_2d_data.parent.mkdir(parents=True, exist_ok=True)

        path_to_annotated_dir = path_to_recording_folder / "annotated_videos" / tracker_name
        path_to_annotated_dir.mkdir(parents=True, exist_ok=True)

        # 1) Extract
        points_2d, raw_kp_list, raw_sc_list, n_people_list, video_paths = rtmpose_2d_from_synced_folder_mp(
            path_to_synced_videos,
            num_workers=6,
            show_progress=True,
        )
        np.save(path_to_save_2d_data, points_2d)

        # 2) Annotate (MP per camera)
        annotate_synced_videos_from_raw_mp(
            video_paths=video_paths,
            raw_kp_list=raw_kp_list,
            raw_sc_list=raw_sc_list,
            n_people_list=n_people_list,
            out_dir=path_to_annotated_dir,
            num_workers=6,
            show_progress=True,
        )
