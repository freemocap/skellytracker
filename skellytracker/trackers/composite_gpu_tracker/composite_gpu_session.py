"""
CompositeGPUSession: a long-lived GPU inference session that composes
selectable body, hand, and face ONNX models under a single ORT context.

Architecture:
  - body_model  → RTMO (one-stage, 17 COCO body keypoints, no detector needed)
  - hand_model  → RTMPose hand ONNX (21 keypoints per hand, crop from body wrists)
  - face_model  → RTMPose face ONNX (106 keypoints, crop from body head landmarks)

Batch inference pipeline:
  1. Stack N images → body_model batched session.run → N × body keypoints + bboxes
  2. Fixed-size square ROI crops from body wrist/head landmarks
  3. Letterbox each crop → stack → hand/face batched session.run → SIMCC decode
  4. Merge body + right_hand + left_hand + face keypoints per image
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from skellytracker.utilities.gpu_utils.model_registry import (
    ModelSpec,
    resolve_model_path,
)
from skellytracker.utilities.gpu_utils.rtm_postprocessing import (
    get_simcc_maximum,
)
from skellytracker.utilities.gpu_utils.rtm_preprocessing import (
    rtmo_postprocess,
    rtmo_preprocess,
)
from skellytracker.utilities.gpu_utils.ort_session_utils import (
    ExecutionProviderName,
    build_tuned_ort_session,
    ensure_cuda_dlls_loaded,
    probe_supports_batch,
    resolve_provider,
    session_run_batched,
)
from skellytracker.trackers.composite_gpu_tracker.roi_crop_utils import (
    ROIBox,
    collect_visible_head_points,
    compute_face_crop_params,
    compute_square_roi,
    smooth_roi_params,
)
from skellytracker.trackers.composite_gpu_tracker.sub_model_spec import (
    TrackerPreset,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Config
# =============================================================================

def _default_engine_cache_dir() -> Path:
    return Path.home() / ".cache" / "skellytracker" / "trt_engines"


class CompositeGPUSessionConfig(BaseModel):
    """Configuration for a composable GPU pose session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_provider: ExecutionProviderName = "cuda"
    engine_cache_dir: Path = Field(default_factory=_default_engine_cache_dir)
    max_batch_size: int = 4
    fp16: bool = True
    on_provider_missing: str = "fallback"

    detect_hands: bool = True
    detect_face: bool = True

    # ------------------------------------------------------------------
    # Model specs (ONNX path, input size, keypoint count, preprocessing)
    # ------------------------------------------------------------------
    body_spec: ModelSpec = Field(default_factory=ModelSpec.rtmo_medium)
    hand_spec: ModelSpec = Field(default_factory=ModelSpec.rtmpose_hand)
    face_spec: ModelSpec = Field(default_factory=ModelSpec.rtmpose_face)

    # ------------------------------------------------------------------
    # ROI crop parameters (face crop size is derived dynamically from
    # head landmarks × face_roi_scale; hand crop size is derived from
    # the smoothed face crop × hand_roi_face_scale.)
    # ------------------------------------------------------------------
    hand_roi_face_scale: float = 1.5
    hand_roi_image_fraction: float = 0.2
    hand_roi_center_offset: float = 0.17
    hand_wrist_bias: float = 1.5

    # Body keypoint indices for ROI cropping (COCO 17 order)
    body_left_wrist_index: int = 9
    body_right_wrist_index: int = 10
    body_left_elbow_index: int = 7
    body_right_elbow_index: int = 8
    body_head_indices: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ROI smoothing
    roi_visibility_threshold: float = 0.3
    roi_smoothing: float = 0.7

    # Face crop scale
    face_roi_scale: float = 2.0

    # ==================================================================
    # Convenience constructors
    # ==================================================================

    @classmethod
    def preset(cls, tier: TrackerPreset | str) -> "CompositeGPUSessionConfig":
        """Return a config with all sub-model specs set to *tier*."""
        if isinstance(tier, str):
            tier = TrackerPreset(tier)

        if tier == TrackerPreset.light:
            return cls(
                body_spec=ModelSpec.rtmo_light(),
                hand_spec=ModelSpec.rtmpose_hand(),
                face_spec=ModelSpec.rtmpose_face(),
            )
        elif tier == TrackerPreset.medium:
            return cls(
                body_spec=ModelSpec.rtmo_medium(),
                hand_spec=ModelSpec.rtmpose_hand(),
                face_spec=ModelSpec.rtmpose_face(),
            )
        else:  # heavy
            return cls(
                body_spec=ModelSpec.rtmo_heavy(),
                hand_spec=ModelSpec.rtmpose_hand(),
                face_spec=ModelSpec.rtmpose_face(),
            )


# =============================================================================
# Session
# =============================================================================

@dataclass
class CompositeGPUSession:
    config: CompositeGPUSessionConfig
    _active_provider: ExecutionProviderName

    _body_session: ort.InferenceSession | None = None
    _hand_session: ort.InferenceSession | None = None
    _face_session: ort.InferenceSession | None = None

    _body_supports_batch: bool = False
    _hand_supports_batch: bool = False
    _face_supports_batch: bool = False

    # NMS thresholds for RTMO body postprocessing
    _body_nms_thr: float = 0.45
    _body_score_thr: float = 0.7

    # Per-side smoothed ROI center: (cx, cy) or None
    _smooth_left_hand_center: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _smooth_right_hand_center: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _smooth_face_roi: tuple[float, float, float] | None = field(default=None, init=False, repr=False)

    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, config: CompositeGPUSessionConfig | None = None) -> "CompositeGPUSession":
        config = config or CompositeGPUSessionConfig()
        if config.execution_provider in ("trt", "cuda"):
            ensure_cuda_dlls_loaded()
        active_provider = resolve_provider(
            requested=config.execution_provider,
            on_missing=config.on_provider_missing,  # type: ignore[arg-type]
        )
        config.engine_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Constructing CompositeGPUSession (provider={active_provider!r}, "
                    f"max_batch={config.max_batch_size}, hands={config.detect_hands}, face={config.detect_face})")

        session = cls(config=config, _active_provider=active_provider)
        session._build_body(active_provider)
        session._build_hands(active_provider)
        session._build_face(active_provider)

        if config.detect_hands and config.detect_face:
            session._executor = ThreadPoolExecutor(max_workers=2)

        session._warmup()
        return session

    def _build_body(self, provider: ExecutionProviderName) -> None:
        spec = self.config.body_spec
        body_onnx = str(resolve_model_path(spec.source))
        logger.info(f"RTMO body model: {body_onnx}")
        self._body_session = build_tuned_ort_session(
            onnx_path=body_onnx, provider=provider, engine_cache_dir=self.config.engine_cache_dir,
            fp16=self.config.fp16, log_label="rtmo_body",
            max_batch_size=self.config.max_batch_size,
        )
        self._body_supports_batch = probe_supports_batch(self._body_session, label="rtmo_body")

    def _build_hands(self, provider: ExecutionProviderName) -> None:
        if not self.config.detect_hands:
            return
        spec = self.config.hand_spec
        hand_onnx = str(resolve_model_path(spec.source))
        logger.info(f"Hand model: {hand_onnx}")
        self._hand_session = build_tuned_ort_session(
            onnx_path=hand_onnx, provider=provider,
            engine_cache_dir=self.config.engine_cache_dir,
            fp16=self.config.fp16, log_label="rtmpose_hand",
            max_batch_size=self.config.max_batch_size,
        )
        self._hand_supports_batch = probe_supports_batch(self._hand_session, label="rtmpose_hand")

    def _build_face(self, provider: ExecutionProviderName) -> None:
        if not self.config.detect_face:
            return
        spec = self.config.face_spec
        try:
            face_onnx = str(resolve_model_path(spec.source))
            logger.info(f"Face model: {face_onnx}")
        except Exception as e:
            logger.warning(f"Face model download failed ({e!r}); face disabled.")
            self.config.detect_face = False
            return

        self._face_session = build_tuned_ort_session(
            onnx_path=face_onnx, provider=provider,
            engine_cache_dir=self.config.engine_cache_dir,
            fp16=self.config.fp16, log_label="rtmpose_face",
            max_batch_size=self.config.max_batch_size,
        )
        self._face_supports_batch = probe_supports_batch(self._face_session, label="rtmpose_face")

    @property
    def active_provider(self) -> ExecutionProviderName:
        return self._active_provider

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ warmup

    def _warmup(self) -> None:
        """Run synthetic frames through the full pipeline once per batch size
        so that GPU JIT / cuDNN auto-tune / TRT compilation is paid up-front."""
        h, w = 480, 640
        synthetic = np.full((h, w, 3), 128, dtype=np.uint8)
        sizes = sorted({1, max(1, self.config.max_batch_size)})
        for batch_size in sizes:
            t0 = time.perf_counter()
            logger.info(f"Warmup starting (batch_size={batch_size}) ...")
            try:
                self.predict_batch([synthetic] * batch_size)
                elapsed = time.perf_counter() - t0
                logger.info(f"Warmup OK (batch_size={batch_size}, elapsed={elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.warning(
                    f"Warmup failed at batch_size={batch_size} "
                    f"(elapsed={elapsed:.1f}s, non-fatal): {e!r}"
                )

    # ------------------------------------------------------------------ inference

    def predict_single(self, image: NDArray[np.uint8]) -> dict[str, Any]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[NDArray[np.uint8]]) -> list[dict[str, Any]]:
        if not images:
            return []

        n = len(images)
        logger.info(f"predict_batch: {n} image(s)")

        body_results = self._run_body_batch(images)
        logger.info(f"predict_batch: body done ({body_results[0][0].shape[0]} person(s))")

        if self._executor is not None:
            logger.info("predict_batch: submitting hands + face to thread pool")
            fh = self._executor.submit(self._run_hands_batch, images, body_results)
            ff = self._executor.submit(self._run_face_batch, images, body_results)
            hand_kpts, right_rois, left_rois = fh.result()
            face_kpts, face_rois = ff.result()
            logger.info("predict_batch: hands + face futures resolved")
        else:
            hand_kpts, right_rois, left_rois = self._run_hands_batch(images, body_results)
            face_kpts, face_rois = self._run_face_batch(images, body_results)

        merged: list[dict[str, Any]] = []
        for i in range(n):
            merged.append({
                "body": body_results[i],
                "hands": hand_kpts[i],
                "face": face_kpts[i],
                "right_hand_roi": right_rois[i] if i < len(right_rois) else None,
                "left_hand_roi": left_rois[i] if i < len(left_rois) else None,
                "face_roi": face_rois[i] if i < len(face_rois) else None,
            })
        return merged

    # ------------------------------------------------------------------ body

    def _run_body_batch(self, images: list[NDArray[np.uint8]]) -> list[tuple[NDArray, NDArray]]:
        if self._body_session is None:
            return [_empty_body_result() for _ in images]

        t0 = time.perf_counter()
        spec = self.config.body_spec
        preprocessed = [rtmo_preprocess(img, spec.input_size, spec.mean, spec.std)
                        for img in images]
        logger.info(f"_run_body_batch: preprocessed {len(images)} image(s) "
                     f"in {(time.perf_counter() - t0)*1000:.0f}ms")

        t1 = time.perf_counter()
        batch = np.stack([p[0].transpose(2, 0, 1).astype(np.float32) for p in preprocessed], axis=0)
        batch = np.ascontiguousarray(batch)
        try:
            outputs = session_run_batched(self._body_session, batch)
        except Exception as e:
            logger.error(f"RTMO batched inference failed: {e!r}")
            return [_empty_body_result() for _ in images]
        logger.info(f"_run_body_batch: ONNX inference "
                     f"in {(time.perf_counter() - t1)*1000:.0f}ms")

        t2 = time.perf_counter()
        results: list[tuple[NDArray, NDArray]] = []
        for i in range(len(images)):
            outputs_i = [o[i:i + 1] for o in outputs]
            try:
                kpts, scores = rtmo_postprocess(
                    outputs_i,
                    ratio=preprocessed[i][1],
                    nms_thr=self._body_nms_thr,
                    score_thr=self._body_score_thr,
                )
                results.append((kpts, scores))
            except Exception as e:
                logger.warning(f"RTMO postprocess failed: {e!r}")
                results.append(_empty_body_result())
        logger.info(f"_run_body_batch: postprocess "
                     f"in {(time.perf_counter() - t2)*1000:.0f}ms "
                     f"(total={results[0][0].shape[0]} person(s))")
        return results

    # ------------------------------------------------------------------ hands

    def _run_hands_batch(
        self, images: list[NDArray[np.uint8]], body_results: list[tuple[NDArray, NDArray]],
    ) -> tuple[list[tuple[NDArray, NDArray]], list[ROIBox | None], list[ROIBox | None]]:
        t0 = time.perf_counter()
        n = len(images)
        spec = self.config.hand_spec
        if self._hand_session is None or not self.config.detect_hands:
            return ([_empty_hands_result(spec.num_keypoints) for _ in images],
                    [None] * n, [None] * n)

        model_sz = spec.input_size
        all_crops: list[tuple[int, NDArray, ROIBox, int]] = []

        for i, (image, (body_kpts, _)) in enumerate(zip(images, body_results)):
            if body_kpts.shape[0] == 0:
                continue
            body_xy = body_kpts[0]
            image_h, image_w = image.shape[:2]

            # Hand crop size from smoothed face ROI (prev frame), or image fraction
            if self._smooth_face_roi is not None:
                crop_sz = int(self._smooth_face_roi[2] * self.config.hand_roi_face_scale)
            else:
                crop_sz = int(min(image_w, image_h) * self.config.hand_roi_image_fraction)

            for side_flag, wrist_idx, elbow_idx in [
                (0, self.config.body_right_wrist_index, self.config.body_right_elbow_index),
                (1, self.config.body_left_wrist_index, self.config.body_left_elbow_index),
            ]:
                wrist_xy = body_xy[wrist_idx]
                elbow_xy = body_xy[elbow_idx]
                if np.isnan(wrist_xy).any() or np.isnan(elbow_xy).any():
                    continue

                # Project hand center past wrist along forearm direction.
                # Offset is proportional to crop size so the wrist sits ~1/3 in,
                # giving ~2/3 of the crop to the fingers.
                forearm = wrist_xy - elbow_xy
                flen = float(np.linalg.norm(forearm))
                if flen < 1.0:
                    continue
                forearm_dir = forearm / flen
                offset = crop_sz * self.config.hand_roi_center_offset
                hand_cx = wrist_xy[0] + forearm_dir[0] * offset
                hand_cy = wrist_xy[1] + forearm_dir[1] * offset

                # Smooth center
                prev = (self._smooth_right_hand_center if side_flag == 0
                        else self._smooth_left_hand_center)
                cx, cy, _ = smooth_roi_params(
                    raw_cx=float(hand_cx), raw_cy=float(hand_cy), raw_size=float(crop_sz),
                    prev_smoothed=(*prev, float(crop_sz)) if prev else None, alpha=0.7,
                )
                if side_flag == 0:
                    self._smooth_right_hand_center = (cx, cy)
                else:
                    self._smooth_left_hand_center = (cx, cy)

                roi = compute_square_roi(
                    center_x=int(cx), center_y=int(cy), size=crop_sz,
                    image_w=image_w, image_h=image_h,
                )
                crop = roi.crop_image(image)
                if crop.size > 0:
                    all_crops.append((i, crop, roi, side_flag))

        if not all_crops:
            return ([_empty_hands_result(spec.num_keypoints) for _ in images], [None] * n, [None] * n)

        # Letterbox each crop → batch → SIMCC decode
        preprocessed = [_simple_letterbox(crop, model_sz, spec.mean, spec.std)
                        for _, crop, _, _ in all_crops]
        batch = np.stack([p[0].transpose(2, 0, 1) for p in preprocessed], axis=0)
        batch = np.ascontiguousarray(batch)
        try:
            outputs = session_run_batched(self._hand_session, batch)
        except Exception as e:
            logger.error(f"Hand batched inference failed: {e!r}")
            return ([_empty_hands_result(spec.num_keypoints) for _ in images], [None] * n, [None] * n)

        simcc_x, simcc_y = outputs
        hand_results_per_image: list[list[tuple[NDArray, NDArray, int]]] = [[] for _ in images]

        for j, (img_idx, _, roi, side_flag) in enumerate(all_crops):
            sx, sy = simcc_x[j:j + 1], simcc_y[j:j + 1]
            locs, scores = get_simcc_maximum(sx, sy)
            kpts_model = locs / spec.simcc_split_ratio
            ratio = preprocessed[j][1]
            kpts = kpts_model / ratio
            kpts[:, :, 0] += float(roi.x)
            kpts[:, :, 1] += float(roi.y)

            if float(scores.mean()) < 0.15:
                continue
            hand_results_per_image[img_idx].append((kpts.astype(np.float64), scores.astype(np.float32), side_flag))

        # Merge per-image
        kpt_results: list[tuple[NDArray, NDArray]] = []
        right_rois: list[ROIBox | None] = []
        left_rois: list[ROIBox | None] = []
        n_kpt = spec.num_keypoints

        for img_idx in range(n):
            per_img = hand_results_per_image[img_idx]
            body_kpts_i = body_results[img_idx][0]

            r_kpts = np.full((1, n_kpt, 2), np.nan, dtype=np.float64)
            r_sc = np.zeros((1, n_kpt), dtype=np.float32)
            l_kpts = np.full((1, n_kpt, 2), np.nan, dtype=np.float64)
            l_sc = np.zeros((1, n_kpt), dtype=np.float32)

            for kpts, scores, sf in per_img:
                if kpts.shape[0] == 0:
                    continue
                if sf == 0:
                    r_kpts, r_sc = kpts, scores
                else:
                    l_kpts, l_sc = kpts, scores

            # Hand-overlap dedup
            if (not np.isnan(r_kpts[0, 0]).any() and not np.isnan(l_kpts[0, 0]).any()
                    and body_kpts_i.shape[0] > 0):
                rw, lw = r_kpts[0, 0], l_kpts[0, 0]
                if float(np.linalg.norm(rw - lw)) < 80.0:
                    body = body_kpts_i[0]
                    brw = body[self.config.body_right_wrist_index]
                    blw = body[self.config.body_left_wrist_index]
                    if not np.isnan(brw).any() and not np.isnan(blw).any():
                        mid = (rw + lw) / 2
                        if float(np.linalg.norm(mid - brw)) <= float(np.linalg.norm(mid - blw)):
                            l_kpts = np.full((1, n_kpt, 2), np.nan, dtype=np.float64)
                            l_sc = np.zeros((1, n_kpt), dtype=np.float32)
                        else:
                            r_kpts = np.full((1, n_kpt, 2), np.nan, dtype=np.float64)
                            r_sc = np.zeros((1, n_kpt), dtype=np.float32)

            # Wrist blending + anthropometry filter
            if body_kpts_i.shape[0] > 0:
                body = body_kpts_i[0].astype(np.float64)
                body_sc = body_results[img_idx][1][0]
                r_kpts, r_sc, body = _blend_and_validate_hand(
                    r_kpts, r_sc, body, body_sc,
                    self.config.body_right_wrist_index,
                    hand_wrist_bias=self.config.hand_wrist_bias,
                )
                l_kpts, l_sc, body = _blend_and_validate_hand(
                    l_kpts, l_sc, body, body_sc,
                    self.config.body_left_wrist_index,
                    hand_wrist_bias=self.config.hand_wrist_bias,
                )
                body_kpts_i[0, :, :] = body

            all_k = np.concatenate([r_kpts, l_kpts], axis=1)
            all_s = np.concatenate([r_sc, l_sc], axis=1)
            kpt_results.append((all_k, all_s))
            right_rois.append(_find_roi_for_side(all_crops, img_idx, 0))
            left_rois.append(_find_roi_for_side(all_crops, img_idx, 1))

        logger.info(f"_run_hands_batch: {len(all_crops)} crops → "
                     f"{(time.perf_counter() - t0)*1000:.0f}ms total")
        return kpt_results, right_rois, left_rois

    # ------------------------------------------------------------------ face

    def _run_face_batch(
        self, images: list[NDArray[np.uint8]], body_results: list[tuple[NDArray, NDArray]],
    ) -> tuple[list[tuple[NDArray, NDArray]], list[ROIBox | None]]:
        t0 = time.perf_counter()
        n = len(images)
        spec = self.config.face_spec
        if self._face_session is None or not self.config.detect_face:
            return ([_empty_face_result(spec.num_keypoints) for _ in images],
                    [None] * n)

        model_sz = spec.input_size
        all_crops: list[tuple[int, NDArray, ROIBox]] = []

        for i, (image, (body_kpts, body_scores)) in enumerate(zip(images, body_results)):
            if body_kpts.shape[0] == 0:
                continue
            body_xy = body_kpts[0]
            body_vis = body_scores[0]
            image_h, image_w = image.shape[:2]

            head_pts = collect_visible_head_points(
                body_xyz=np.column_stack([body_xy, np.zeros((body_xy.shape[0],))]),
                body_vis=body_vis, head_indices=self.config.body_head_indices,
                visibility_threshold=self.config.roi_visibility_threshold,
            )
            if head_pts is None:
                continue
            face_params = compute_face_crop_params(
                visible_head_points=head_pts, face_roi_scale=self.config.face_roi_scale,
            )
            if face_params is None:
                continue

            (raw_cx, raw_cy), raw_crop_sz = face_params
            # Head bbox width from visible points — used to bias the center
            # downward because the face extends ~40 % of its width below the
            # eye-line centre while the forehead is only ~20 % above.
            head_w = float(head_pts[:, 0].max() - head_pts[:, 0].min())
            raw_cy += head_w * 0.20
            # Clamp crop size: small enough for the model to localise the
            # face (RTMPose was trained with ~1.25× face-bbox padding),
            # large enough to contain the full jawline contour.
            raw_crop_sz = max(120.0, min(600.0, raw_crop_sz))
            cx, cy, smoothed_sz = smooth_roi_params(
                raw_cx=raw_cx, raw_cy=raw_cy, raw_size=raw_crop_sz,
                prev_smoothed=self._smooth_face_roi, alpha=0.7,
            )
            self._smooth_face_roi = (cx, cy, smoothed_sz)
            crop_sz_int = int(smoothed_sz)

            roi = compute_square_roi(
                center_x=int(cx), center_y=int(cy), size=crop_sz_int,
                image_w=image_w, image_h=image_h,
            )
            crop = roi.crop_image(image)
            if crop.size > 0:
                all_crops.append((i, crop, roi))

        if not all_crops:
            return ([_empty_face_result(spec.num_keypoints) for _ in images], [None] * n)

        # Letterbox → batch → SIMCC decode
        preprocessed = [_simple_letterbox(crop, model_sz, spec.mean, spec.std)
                        for _, crop, _ in all_crops]
        batch = np.stack([p[0].transpose(2, 0, 1) for p in preprocessed], axis=0)
        batch = np.ascontiguousarray(batch)
        try:
            outputs = session_run_batched(self._face_session, batch)
        except Exception as e:
            logger.error(f"Face batched inference failed: {e!r}")
            return ([_empty_face_result(spec.num_keypoints) for _ in images],
                    [None] * n)

        simcc_x, simcc_y = outputs
        face_results: list[tuple[NDArray, NDArray] | None] = [None] * n

        for j, (img_idx, _, roi) in enumerate(all_crops):
            sx, sy = simcc_x[j:j + 1], simcc_y[j:j + 1]
            locs, scores = get_simcc_maximum(sx, sy)
            kpts_model = locs / spec.simcc_split_ratio
            ratio = preprocessed[j][1]
            kpts = kpts_model / ratio
            kpts[:, :, 0] += float(roi.x)
            kpts[:, :, 1] += float(roi.y)
            face_results[img_idx] = (kpts.astype(np.float64), scores.astype(np.float32))

        face_rois_out: list[ROIBox | None] = []
        for img_idx in range(n):
            found = None
            for ci, _, roi in all_crops:
                if ci == img_idx:
                    found = roi; break
            face_rois_out.append(found)

        logger.info(f"_run_face_batch: {len(all_crops)} crops → "
                     f"{(time.perf_counter() - t0)*1000:.0f}ms total")
        return ([r if r is not None else _empty_face_result(spec.num_keypoints) for r in face_results], face_rois_out)


# =============================================================================
# Helpers
# =============================================================================

def _simple_letterbox(
    image: NDArray[np.uint8],
    target_size: tuple[int, int],
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[NDArray[np.float32], float]:
    import cv2
    h, w = image.shape[:2]
    th, tw = target_size
    ratio = min(th / h, tw / w)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.float32)
    canvas[:nh, :nw] = resized.astype(np.float32)
    if mean is not None and std is not None:
        canvas = (canvas - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return canvas, ratio


def _find_roi_for_side(
    all_crops: list[tuple[int, Any, ROIBox, int]], img_idx: int, side_flag: int,
) -> ROIBox | None:
    for ci, _, roi, sf in all_crops:
        if ci == img_idx and sf == side_flag:
            return roi
    return None


# Hand anthropometry constants (keypoint indices for the RTMPose 21-point hand).
_HAND_ROOT = 0
_HAND_FOREFINGER_MCP = 5   # index-finger knuckle (palm boundary)
_HAND_MIDDLE_TIP = 12       # middle-finger tip
_HAND_PINKY_MCP = 17        # pinky knuckle (palm boundary)


def _blend_and_validate_hand(
    kpts: NDArray[np.float64],
    scores: NDArray[np.float32],
    body: NDArray[np.float64],
    body_scores: NDArray[np.float32],
    body_wrist_idx: int,
    *,
    hand_wrist_bias: float = 1.5,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float64]]:
    """Validate hand geometry, then confidence-weighted blend with body wrist."""
    if np.isnan(kpts[0, _HAND_ROOT]).any():
        return kpts, scores, body

    # --- anthropometry filter (position-independent — validate first) ---
    valid = kpts[0]
    valid_mask = ~np.isnan(valid).any(axis=1)
    valid_pts = valid[valid_mask]

    if valid_mask.sum() < 4:
        return _empty_single_hand() + (body,)

    min_xy = valid_pts.min(axis=0)
    max_xy = valid_pts.max(axis=0)
    bbox_w = max_xy[0] - min_xy[0]
    bbox_h = max_xy[1] - min_xy[1]
    diag = float(np.linalg.norm([bbox_w, bbox_h]))
    if diag < 30.0 or diag > 280.0:
        return _empty_single_hand() + (body,)

    if bbox_w > 0 and bbox_h > 0:
        aspect = bbox_w / bbox_h
        if aspect < 0.25 or aspect > 4.0:
            return _empty_single_hand() + (body,)

    if (valid_mask[_HAND_MIDDLE_TIP] and valid_mask[_HAND_ROOT]
            and valid_mask[_HAND_FOREFINGER_MCP] and valid_mask[_HAND_PINKY_MCP]):
        finger_len = float(np.linalg.norm(
            valid[_HAND_MIDDLE_TIP] - valid[_HAND_ROOT]))
        palm_w = float(np.linalg.norm(
            valid[_HAND_FOREFINGER_MCP] - valid[_HAND_PINKY_MCP]))
        if palm_w > 0 and finger_len / palm_w < 1.2:
            return _empty_single_hand() + (body,)

    # --- wrist blending ---
    hand_root = kpts[0, _HAND_ROOT]
    hand_conf = float(scores[0, _HAND_ROOT])
    body_wrist = body[body_wrist_idx]
    body_conf = float(body_scores[body_wrist_idx])

    if np.isnan(body_wrist).any():
        return kpts, scores, body

    # Confidence-weighted blend — hand bias pulls toward hand estimate
    hw = hand_conf * hand_wrist_bias
    bw = body_conf
    blended = (body_wrist * bw + hand_root * hw) / (bw + hw)

    # Translate hand so its root lands at blended wrist
    offset = blended - hand_root
    kpts = kpts + offset.astype(np.float64)

    # Write blended wrist back into body
    body = body.copy()
    body[body_wrist_idx] = blended

    return kpts, scores, body


def _empty_single_hand(num_keypoints: int = 21) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    return (
        np.full((1, num_keypoints, 2), np.nan, dtype=np.float64),
        np.zeros((1, num_keypoints), dtype=np.float32),
    )


def _empty_body_result() -> tuple[NDArray, NDArray]:
    return (np.empty((0, 17, 2), dtype=np.float64),
            np.empty((0, 17), dtype=np.float32))


def _empty_hands_result(num_keypoints: int = 21) -> tuple[NDArray, NDArray]:
    return (np.empty((0, 2 * num_keypoints, 2), dtype=np.float64),
            np.empty((0, 2 * num_keypoints), dtype=np.float32))


def _empty_face_result(num_keypoints: int = 106) -> tuple[NDArray, NDArray]:
    return (np.empty((0, num_keypoints, 2), dtype=np.float64),
            np.empty((0, num_keypoints), dtype=np.float32))
