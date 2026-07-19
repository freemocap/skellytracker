from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field, replace

from numpy.typing import NDArray
import numpy as np

from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.data_primitives import BoundingBox, Keypoints
from skellytracker.core.detectors.detection_context import DetectionContext
from skellytracker.core.detectors.detector_base_classes import (
    KeypointDetector,
    ObjectDetector,
    build_keypoint_detector,
    build_object_detector,
)
from skellytracker.core.data_primitives.observation import StageObservation
from skellytracker.core.sessions.session import Session
from skellytracker.core.temporal_processing.bbox_policy import BBoxPolicy, predict_bbox_from_keypoints
from skellytracker.core.temporal_processing.bbox_smoothing import apply_bbox_ema
from skellytracker.core.temporal_processing.keypoint_filtering import (
    KalmanFilter,
    OneEuroFilter,
    make_keypoint_filter,
)
from skellytracker.core.temporal_processing.keypoint_reset_policy import KeypointResetPolicy
from skellytracker.core.tracker.tracker_state import KeypointSmoothingState, StageState


@dataclass
class DetectionStage:
    """Compositional unit of the detection pipeline.

    Binds one optional ObjectDetector with one or more KeypointDetectors.
    Child stages receive the parent's crop and keypoints as context and run
    their own detection subtree, enabling hierarchical top-down pipelines
    (e.g., body stage → face child stage).

    Temporal processing (bbox reuse policy, EMA smoothing, one-euro keypoint
    filtering) is configured per stage and applied inside run() using StageState.
    """

    name: str
    keypoint_detectors: list[KeypointDetector]
    object_detector: ObjectDetector | None = None
    children: list[DetectionStage] = field(default_factory=list)
    bbox_policy: BBoxPolicy = field(default_factory=BBoxPolicy)
    bbox_smoothing_alpha: float | None = None
    keypoint_filter: OneEuroFilter | KalmanFilter | None = None
    keypoint_reset_policy: KeypointResetPolicy = field(default_factory=KeypointResetPolicy)
    # Per-camera detector instances for non-ONNX (stateful) keypoint detectors.
    # Index matches keypoint_detectors; inner dict maps cam_id → detector instance.
    # Populated lazily in run_batch so each camera stream gets its own state.
    _cam_kp_detectors: list[dict[str, KeypointDetector]] = field(
        default_factory=list, init=False, repr=False
    )
    # Persistent thread pool for run_batch's per-camera parallelism (preprocess +
    # non-ONNX detection). Created lazily on first use and reused across frames —
    # avoids the OS thread spawn/teardown cost of a fresh pool every call. Sized
    # once from the first batch's camera count, which is assumed stable for the
    # life of this stage.
    _executor: concurrent.futures.ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._cam_kp_detectors = [{} for _ in self.keypoint_detectors]

    def _get_executor(self, max_workers: int) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        return self._executor

    def run(
        self,
        image: NDArray[np.uint8],
        state: StageState,
        parent_keypoints: Keypoints | None = None,
        context: DetectionContext | None = None,
    ) -> tuple[StageObservation, StageState]:
        """Run this stage and all child stages.

        Args:
            image: Full or parent-cropped image.
            state: Current temporal state for this stage.
            parent_keypoints: Keypoints from the parent stage, available for
                              computing crop regions (e.g., wrist → hand crop).

        Returns:
            (StageObservation, updated StageState)
        """
        frame_number = context.frame_number if context is not None else 0
        dt = _dt_from_context(context)

        # 1. Object detection with bbox reuse policy
        bbox_state = state.bbox_state
        detector_ran = False
        if self.object_detector is not None:
            if self.bbox_policy.should_redetect(frame_number, state):
                bboxes = self.object_detector.detect(image, context)
                bbox_state = type(bbox_state)(
                    smooth_bbox=bbox_state.smooth_bbox,
                    last_detection_frame=frame_number,
                    last_detected_bbox=bboxes[0] if bboxes else None,
                )
                detector_ran = True
            else:
                predicted = self.bbox_policy.predict_bbox(state)
                bboxes = [predicted] if predicted is not None else []
        else:
            h, w = image.shape[:2]
            bboxes = [BoundingBox.full_image(h, w)]

        raw_bbox = bboxes[0] if bboxes else None

        # 2. BBox smoothing (EMA)
        if raw_bbox is not None and self.bbox_smoothing_alpha is not None:
            smoothed_bbox, bbox_state = apply_bbox_ema(raw_bbox, bbox_state, self.bbox_smoothing_alpha)
        else:
            smoothed_bbox = raw_bbox
            if raw_bbox is not None:
                bbox_state = replace(bbox_state, smooth_bbox=raw_bbox)

        # 3. Keypoint detection — crop, detect, translate back to full-frame coords
        #
        # The crop bbox must be clipped to image bounds *before* use: to_crop()
        # clamps negative/out-of-bounds coords when slicing, so translating by the
        # original (unclamped) bbox origin would offset keypoints by the clamped
        # amount — a translation bug whenever the bbox extends past the frame edge.
        h, w = image.shape[:2]
        crop_bbox = smoothed_bbox.clipped(h, w) if smoothed_bbox is not None else None
        crop = crop_bbox.to_crop(image) if crop_bbox is not None else image
        all_keypoints: list[Keypoints] = []
        updated_kp_states: list[KeypointSmoothingState] = []
        updated_misses: list[int] = []
        updated_resets: list[int] = []

        for i, detector in enumerate(self.keypoint_detectors):
            kp_state = (
                state.keypoint_states[i]
                if i < len(state.keypoint_states)
                else KeypointSmoothingState()
            )
            kpts = detector.detect(crop, context)
            if crop_bbox is not None:
                kpts = kpts.translated(crop_bbox.x1, crop_bbox.y1)

            # 4. Reset detector's internal temporal state after a run of misses
            # (e.g. MediaPipe VIDEO-mode tracking getting silently stuck).
            # Checked before any confidence filtering — a raw zero-keypoint result
            # is what defines a miss here. consecutive_resets backs off the
            # effective threshold each time a reset fires with no real detection
            # in between, so a subject genuinely out of frame doesn't trigger a
            # reset every max_consecutive_misses frames forever; it's cleared by
            # any real (non-empty) detection.
            prev_misses = state.consecutive_misses[i] if i < len(state.consecutive_misses) else 0
            prev_resets = state.consecutive_resets[i] if i < len(state.consecutive_resets) else 0
            if kpts.n_valid == 0:
                misses = prev_misses + 1
                resets = prev_resets
            else:
                misses = 0
                resets = 0
            if self.keypoint_reset_policy.should_reset(misses, resets):
                detector.reset_temporal_state()
                misses = 0
                resets = resets + 1
            updated_misses.append(misses)
            updated_resets.append(resets)

            # 5. Keypoint smoothing (one-euro filter)
            if self.keypoint_filter is not None:
                kpts, kp_state = self.keypoint_filter.smooth(kpts, kp_state, dt)

            all_keypoints.append(kpts)
            updated_kp_states.append(kp_state)

        merged = Keypoints.concatenate(all_keypoints) if all_keypoints else None

        # 5b. Refresh the keypoint-tracked bbox every frame (detect or skip) from
        # this frame's actual keypoints — see BBoxSmoothingState.keypoint_tracked_bbox.
        if self.bbox_policy.keypoint_bbox_expansion is not None and merged is not None:
            fresh_tracked = predict_bbox_from_keypoints(
                merged,
                self.bbox_policy.keypoint_bbox_expansion,
                self.bbox_policy.keypoint_bbox_min_visibility,
            )
            if fresh_tracked is not None:
                bbox_state = replace(bbox_state, keypoint_tracked_bbox=fresh_tracked)

        # 6. Child stages receive the crop; children translate their own coords
        child_observations: dict[str, StageObservation] = {}
        updated_child_states: dict[str, StageState] = {}
        for child in self.children:
            child_state = state.child_states.get(child.name, StageState())
            child_obs, child_state = child.run(crop, child_state, parent_keypoints=merged, context=context)
            child_observations[child.name] = child_obs
            updated_child_states[child.name] = child_state

        obs = StageObservation(
            name=self.name,
            bounding_boxes=bboxes,
            keypoints=merged,
            children=child_observations,
            detector_ran=detector_ran,
        )
        updated_state = StageState(
            bbox_state=bbox_state,
            keypoint_states=updated_kp_states,
            child_states=updated_child_states,
            last_keypoints=merged,
            consecutive_misses=updated_misses,
            consecutive_resets=updated_resets,
        )
        return obs, updated_state

    def run_batch(
        self,
        images: dict[str, NDArray[np.uint8]],
        states: dict[str, StageState],
        context: DetectionContext | None = None,
    ) -> tuple[dict[str, StageObservation], dict[str, StageState]]:
        """Run this stage on N cameras simultaneously.

        For ONNX-backed detectors, all cameras are processed in a single batched
        ORT call. For non-ONNX detectors (MediaPipe, Charuco, ArUco), each camera
        is processed in a thread pool to exploit parallelism without holding the GIL.

        States for cameras not present in ``states`` default to empty StageState.

        Parameters
        ----------
        images:
            Mapping from camera ID to BGR image array (H, W, 3).
        states:
            Mapping from camera ID to current StageState. Keys may be a subset
            of images.keys() — missing cameras get a fresh StageState.
        context:
            Shared detection context (frame number, timestamp).

        Returns
        -------
        (per-camera StageObservation dict, per-camera updated StageState dict)
        """
        # Lazy import to avoid requiring onnxruntime in non-GPU environments
        try:
            from skellytracker.core.sessions.onnx_session import OnnxSession as _OnnxSession
        except ImportError:
            _OnnxSession = None  # type: ignore[assignment,misc]

        frame_number = context.frame_number if context is not None else 0
        dt = _dt_from_context(context)
        cam_ids = list(images.keys())

        # ── 1. Object detection ───────────────────────────────────────────────
        bboxes_per_cam: dict[str, list[BoundingBox]] = {}
        bbox_states_per_cam: dict[str, object] = {}  # BBoxSmoothingState per cam
        detector_ran_per_cam: dict[str, bool] = {}

        # For ONNX-batched detectors, synchronize redetection: if any camera
        # needs a fresh detection, redetect all cameras together.  This keeps
        # the batch size at exactly 0 or N (never a partial subset), which
        # avoids repeated JIT recompilation on CoreML and TRT.
        onnx_object_detector = (
            self.object_detector
            if self.object_detector is not None
            and _OnnxSession is not None
            and isinstance(self.object_detector.session, _OnnxSession)
            else None
        )
        any_needs_redetect = onnx_object_detector is not None and any(
            self.bbox_policy.should_redetect(frame_number, states.get(c, StageState()))
            for c in cam_ids
        )

        for cam_id in cam_ids:
            state = states.get(cam_id, StageState())
            bbox_state = state.bbox_state

            if self.object_detector is not None:
                if onnx_object_detector is not None:
                    # Batched ONNX path: redetect decision is synchronized above.
                    if any_needs_redetect:
                        bboxes_per_cam[cam_id] = None  # type: ignore[assignment]
                        bbox_states_per_cam[cam_id] = type(bbox_state)(
                            smooth_bbox=bbox_state.smooth_bbox,
                            last_detection_frame=frame_number,
                        )
                        detector_ran_per_cam[cam_id] = True
                    else:
                        predicted = self.bbox_policy.predict_bbox(state)
                        bboxes_per_cam[cam_id] = [predicted] if predicted is not None else []
                        bbox_states_per_cam[cam_id] = bbox_state
                        detector_ran_per_cam[cam_id] = False
                elif self.bbox_policy.should_redetect(frame_number, state):
                    bboxes_per_cam[cam_id] = self.object_detector.detect(images[cam_id], context)
                    bbox_states_per_cam[cam_id] = type(bbox_state)(
                        smooth_bbox=bbox_state.smooth_bbox,
                        last_detection_frame=frame_number,
                        last_detected_bbox=bboxes_per_cam[cam_id][0] if bboxes_per_cam[cam_id] else None,
                    )
                    detector_ran_per_cam[cam_id] = True
                else:
                    predicted = self.bbox_policy.predict_bbox(state)
                    bboxes_per_cam[cam_id] = [predicted] if predicted is not None else []
                    bbox_states_per_cam[cam_id] = bbox_state
                    detector_ran_per_cam[cam_id] = False
            else:
                h, w = images[cam_id].shape[:2]
                bboxes_per_cam[cam_id] = [BoundingBox.full_image(h, w)]
                bbox_states_per_cam[cam_id] = bbox_state
                detector_ran_per_cam[cam_id] = False

        # Batched ONNX object detection for cameras needing redetect
        if onnx_object_detector is not None:
            cams_needing_detect = [c for c in cam_ids if bboxes_per_cam[c] is None]
            if cams_needing_detect:
                tensors = {}
                metas = {}
                _t = time.perf_counter()
                pool = self._get_executor(len(cam_ids))
                fut_map = {cam_id: pool.submit(self.object_detector.preprocess, images[cam_id]) for cam_id in cams_needing_detect}
                for cam_id, fut in fut_map.items():
                    tensors[cam_id], metas[cam_id] = fut.result()
                if context is not None and context.timings is not None:
                    context.timings.stop(f"{self.name}.obj_preprocess", _t)
                model_name = self.object_detector.config.model_name
                _t = time.perf_counter()
                raw_batch = self.object_detector.session.run_batched(model_name, tensors)
                if context is not None and context.timings is not None:
                    context.timings.stop(f"{self.name}.obj_infer", _t)
                for cam_id in cams_needing_detect:
                    bboxes_per_cam[cam_id] = self.object_detector.postprocess(raw_batch[cam_id], metas[cam_id])
                    bbox_states_per_cam[cam_id] = replace(
                        bbox_states_per_cam[cam_id],
                        last_detected_bbox=bboxes_per_cam[cam_id][0] if bboxes_per_cam[cam_id] else None,
                    )

        # ── 2. BBox smoothing (EMA) per camera ───────────────────────────────
        smoothed_bboxes: dict[str, BoundingBox | None] = {}
        for cam_id in cam_ids:
            raw_bbox = bboxes_per_cam[cam_id][0] if bboxes_per_cam[cam_id] else None
            bbox_state = bbox_states_per_cam[cam_id]
            if raw_bbox is not None and self.bbox_smoothing_alpha is not None:
                smoothed, bbox_state = apply_bbox_ema(raw_bbox, bbox_state, self.bbox_smoothing_alpha)
                smoothed_bboxes[cam_id] = smoothed
            else:
                smoothed_bboxes[cam_id] = raw_bbox
                if raw_bbox is not None:
                    bbox_state = replace(bbox_state, smooth_bbox=raw_bbox)
            bbox_states_per_cam[cam_id] = bbox_state

        # ── 3. Compute crops per camera ───────────────────────────────────────
        # crop_bboxes are clipped to image bounds — to_crop() clamps internally
        # when slicing, so translating keypoints back by the unclamped bbox
        # origin would offset them whenever the bbox extends past the frame edge.
        crop_bboxes: dict[str, BoundingBox | None] = {}
        crops: dict[str, NDArray[np.uint8]] = {}
        for cam_id in cam_ids:
            sb = smoothed_bboxes[cam_id]
            h, w = images[cam_id].shape[:2]
            crop_bboxes[cam_id] = sb.clipped(h, w) if sb is not None else None
            crops[cam_id] = crop_bboxes[cam_id].to_crop(images[cam_id]) if crop_bboxes[cam_id] is not None else images[cam_id]

        # ── 4. Keypoint detection ─────────────────────────────────────────────
        # Collect per-detector results: list indexed by detector index, each a
        # dict from cam_id → (Keypoints, KeypointSmoothingState)
        all_detector_results: list[dict[str, tuple[Keypoints, KeypointSmoothingState]]] = []
        # Per-camera consecutive-miss/-reset counters, appended to in detector order
        # (index i) so misses_per_cam[cam_id][i] lines up with keypoint_detectors[i].
        misses_per_cam: dict[str, list[int]] = {cam_id: [] for cam_id in cam_ids}
        resets_per_cam: dict[str, list[int]] = {cam_id: [] for cam_id in cam_ids}

        def _update_misses(cam_id: str, detector: KeypointDetector, kpts: Keypoints, i: int) -> None:
            prior_state = states.get(cam_id, StageState())
            prev_misses = (
                prior_state.consecutive_misses[i]
                if i < len(prior_state.consecutive_misses)
                else 0
            )
            prev_resets = (
                prior_state.consecutive_resets[i]
                if i < len(prior_state.consecutive_resets)
                else 0
            )
            if kpts.n_valid == 0:
                misses = prev_misses + 1
                resets = prev_resets
            else:
                misses = 0
                resets = 0
            if self.keypoint_reset_policy.should_reset(misses, resets):
                detector.reset_temporal_state()
                misses = 0
                resets = resets + 1
            misses_per_cam[cam_id].append(misses)
            resets_per_cam[cam_id].append(resets)

        for i, detector in enumerate(self.keypoint_detectors):
            detector_results: dict[str, tuple[Keypoints, KeypointSmoothingState]] = {}

            if _OnnxSession is not None and isinstance(detector.session, _OnnxSession):
                # Batched ONNX path — preprocess all cameras in parallel (cv2/numpy release the GIL)
                tensors = {}
                metas = {}
                _t = time.perf_counter()
                pool = self._get_executor(len(cam_ids))
                fut_map = {cam_id: pool.submit(detector.preprocess, crops[cam_id]) for cam_id in cam_ids}
                for cam_id, fut in fut_map.items():
                    tensors[cam_id], metas[cam_id] = fut.result()
                if context is not None and context.timings is not None:
                    context.timings.stop(f"{self.name}.kp_preprocess", _t)
                model_name = detector.config.model_name
                _t = time.perf_counter()
                raw_batch = detector.session.run_batched(model_name, tensors)
                if context is not None and context.timings is not None:
                    context.timings.stop(f"{self.name}.kp_infer", _t)
                for cam_id in cam_ids:
                    kpts = detector.postprocess(raw_batch[cam_id], metas[cam_id])
                    cb = crop_bboxes[cam_id]
                    if cb is not None:
                        kpts = kpts.translated(cb.x1, cb.y1)
                    _update_misses(cam_id, detector, kpts, i)
                    state = states.get(cam_id, StageState())
                    kp_state = (
                        state.keypoint_states[i]
                        if i < len(state.keypoint_states)
                        else KeypointSmoothingState()
                    )
                    if self.keypoint_filter is not None:
                        kpts, kp_state = self.keypoint_filter.smooth(kpts, kp_state, dt)
                    detector_results[cam_id] = (kpts, kp_state)
            else:
                # Non-ONNX path — per-camera detector instances + thread pool.
                # Each camera needs its own detector so stateful backends (e.g.
                # MediaPipe VIDEO mode) can maintain independent timestamp streams.
                def _detect_one(cam_id: str, detector: KeypointDetector = detector, i: int = i) -> tuple[str, Keypoints, KeypointSmoothingState]:
                    cam_detectors = self._cam_kp_detectors[i]
                    if cam_id not in cam_detectors:
                        cam_detectors[cam_id] = type(detector).create(detector.config, detector.session)
                    cam_detector = cam_detectors[cam_id]
                    kpts = cam_detector.detect(crops[cam_id], context)
                    cb = crop_bboxes[cam_id]
                    if cb is not None:
                        kpts = kpts.translated(cb.x1, cb.y1)
                    _update_misses(cam_id, cam_detector, kpts, i)
                    state = states.get(cam_id, StageState())
                    kp_state = (
                        state.keypoint_states[i]
                        if i < len(state.keypoint_states)
                        else KeypointSmoothingState()
                    )
                    if self.keypoint_filter is not None:
                        kpts, kp_state = self.keypoint_filter.smooth(kpts, kp_state, dt)
                    return cam_id, kpts, kp_state

                executor = self._get_executor(len(cam_ids))
                futures = [executor.submit(_detect_one, cam_id) for cam_id in cam_ids]
                for fut in concurrent.futures.as_completed(futures):
                    cam_id, kpts, kp_state = fut.result()
                    detector_results[cam_id] = (kpts, kp_state)

            all_detector_results.append(detector_results)

        # ── 5. Assemble merged keypoints + kp_states per camera ──────────────
        merged_per_cam: dict[str, Keypoints | None] = {}
        kp_states_per_cam: dict[str, list[KeypointSmoothingState]] = {}

        for cam_id in cam_ids:
            all_kpts = []
            kp_states = []
            for det_results in all_detector_results:
                kpts, kp_state = det_results[cam_id]
                all_kpts.append(kpts)
                kp_states.append(kp_state)
            merged_per_cam[cam_id] = Keypoints.concatenate(all_kpts) if all_kpts else None
            kp_states_per_cam[cam_id] = kp_states

        # ── 5b. Refresh the keypoint-tracked bbox every frame (detect or skip) ──
        if self.bbox_policy.keypoint_bbox_expansion is not None:
            for cam_id in cam_ids:
                merged = merged_per_cam[cam_id]
                if merged is None:
                    continue
                fresh_tracked = predict_bbox_from_keypoints(
                    merged,
                    self.bbox_policy.keypoint_bbox_expansion,
                    self.bbox_policy.keypoint_bbox_min_visibility,
                )
                if fresh_tracked is not None:
                    bbox_states_per_cam[cam_id] = replace(
                        bbox_states_per_cam[cam_id], keypoint_tracked_bbox=fresh_tracked
                    )

        # ── 6. Child stages ───────────────────────────────────────────────────
        child_obs_per_cam: dict[str, dict[str, StageObservation]] = {c: {} for c in cam_ids}
        child_states_per_cam: dict[str, dict[str, StageState]] = {c: {} for c in cam_ids}

        for child in self.children:
            child_stage_states = {
                cam_id: states.get(cam_id, StageState()).child_states.get(child.name, StageState())
                for cam_id in cam_ids
            }
            child_obs_batch, child_states_batch = child.run_batch(crops, child_stage_states, context)
            for cam_id in cam_ids:
                child_obs_per_cam[cam_id][child.name] = child_obs_batch[cam_id]
                child_states_per_cam[cam_id][child.name] = child_states_batch[cam_id]

        # ── 7. Build output dicts ─────────────────────────────────────────────
        obs_out: dict[str, StageObservation] = {}
        states_out: dict[str, StageState] = {}

        for cam_id in cam_ids:
            obs_out[cam_id] = StageObservation(
                name=self.name,
                bounding_boxes=bboxes_per_cam[cam_id],
                keypoints=merged_per_cam[cam_id],
                children=child_obs_per_cam[cam_id],
                detector_ran=detector_ran_per_cam[cam_id],
            )
            states_out[cam_id] = StageState(
                bbox_state=bbox_states_per_cam[cam_id],
                keypoint_states=kp_states_per_cam[cam_id],
                child_states=child_states_per_cam[cam_id],
                last_keypoints=merged_per_cam[cam_id],
                consecutive_misses=misses_per_cam[cam_id],
                consecutive_resets=resets_per_cam[cam_id],
            )

        return obs_out, states_out

    def close(self) -> None:
        """Release resources owned by all detectors in this stage and its children."""
        if self.object_detector is not None:
            self.object_detector.close()
        for detector in self.keypoint_detectors:
            detector.close()
        for cam_detectors in self._cam_kp_detectors:
            for cam_detector in cam_detectors.values():
                cam_detector.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for child in self.children:
            child.close()

    def reset_temporal_state(self) -> None:
        """Reset internal temporal state on all detectors in this stage and its children."""
        if self.object_detector is not None:
            self.object_detector.reset_temporal_state()
        for detector in self.keypoint_detectors:
            detector.reset_temporal_state()
        # Close and clear per-camera instances so they're recreated fresh on the next
        # run_batch call (necessary for VIDEO-mode backends whose timestamp state
        # must restart from scratch between independent recordings).
        for cam_detectors in self._cam_kp_detectors:
            for cam_detector in cam_detectors.values():
                cam_detector.close()
            cam_detectors.clear()
        for child in self.children:
            child.reset_temporal_state()

    @classmethod
    def create(
        cls,
        config: DetectionStageConfig,
        sessions: dict[str, Session],
    ) -> DetectionStage:
        """Build a DetectionStage and its full subtree from config."""
        object_detector = (
            build_object_detector(config.object_detector, sessions)
            if config.object_detector is not None
            else None
        )
        keypoint_detectors = [
            build_keypoint_detector(kp_cfg, sessions)
            for kp_cfg in config.keypoint_detectors
        ]
        children = [
            DetectionStage.create(child_cfg, sessions)
            for child_cfg in config.children
        ]
        return cls(
            name=config.name,
            object_detector=object_detector,
            keypoint_detectors=keypoint_detectors,
            children=children,
            bbox_policy=BBoxPolicy.from_config(config.bbox_policy),
            bbox_smoothing_alpha=config.bbox_smoothing.alpha if config.bbox_smoothing is not None else None,
            keypoint_filter=(
                make_keypoint_filter(config.keypoint_smoothing)
                if config.keypoint_smoothing is not None
                else None
            ),
            keypoint_reset_policy=KeypointResetPolicy.from_config(config.keypoint_reset_policy),
        )


def _dt_from_context(context: DetectionContext | None) -> float:
    """Return a time-delta to use for the one-euro filter.

    Always returns 1.0 (one frame unit) for now. min_cutoff and beta are
    therefore in frames⁻¹ rather than Hz. Proper per-frame dt computation
    requires storing the previous timestamp in StageState, which is a future
    improvement.
    """
    return 1.0
