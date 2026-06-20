"""
RTMPoseSession: a long-lived, tuned ONNX inference session that wraps rtmlib's
Wholebody model. Designed to be hosted in a single dedicated GPU worker process
that serves multiple cameras via batched inference.

Why this exists separately from `RTMPoseDetector`:
- `RTMPoseDetector` was designed for one tracker per process (the webcam demo
  pattern). When N processes each construct one, they end up with N independent
  CUDA contexts on a single GPU, which serializes work + adds per-context
  overhead.
- This module owns a *single* tuned ORT session per process and exposes batched
  inference (`predict_batch`) so a centralized worker can fan in frames from N
  cameras and run one ONNX call per round.

What "tuned" means here:
- Explicit ORT `SessionOptions` (`ORT_ENABLE_ALL`, single-threaded ops — the
  GPU is the bottleneck, not CPU).
- Explicit provider list with options:
    * TensorRT EP with engine + timing cache enabled, FP16 on by default.
    * CUDA EP fallback with `cudnn_conv_algo_search="EXHAUSTIVE"`.
- A warmup pass at construction so the first real frame doesn't pay JIT /
  algo-search / TRT-compile cost.
"""
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from skellytracker.utilities.gpu_utils.model_registry import (
    MODEL_URLS,
    ModelSource,
    resolve_model_path,
)
from skellytracker.utilities.gpu_utils.rtm_postprocessing import (
    convert_coco_to_openpose,
    get_simcc_maximum,
    multiclass_nms,
)
from skellytracker.utilities.gpu_utils.rtm_preprocessing import (
    rtmpose_letterbox_postprocess,
    rtmpose_letterbox_preprocess,
    yolox_letterbox_preprocess,
)
from skellytracker.utilities.gpu_utils.ort_session_utils import (
    ExecutionProviderName,
    build_tuned_ort_session,
    cuda_device_free_bytes,
    cuda_device_total_bytes,
    ensure_cuda_dlls_loaded,
    probe_supports_batch,
    resolve_provider,
    select_best_cuda_device,
    session_run_batched,
)
from skellytracker.trackers.rtmpose_tracker._yolox_dynamic_batch import (
    PRENMS_BBOX_OUTPUT,
    PRENMS_CONF_OUTPUT,
    ensure_dynamic_batch,
    ensure_prenms_model,
)
from skellytracker.trackers.rtmpose_tracker.rtmpose_tracking_state import (
    PersonTrackingState,
    predict_bbox_from_tracking,
    should_run_detector,
    update_tracking_state,
)

logger = logging.getLogger(__name__)


def _default_engine_cache_dir() -> Path:
    return Path.home() / ".cache" / "skellytracker" / "trt_engines"


def _default_session_provider() -> ExecutionProviderName:
    import sys
    return "coreml" if sys.platform == "darwin" else "trt"


class RTMPoseSessionConfig(BaseModel):
    """Configuration for a tuned RTMPose ONNX session.

    Independent of the per-frame `RTMPoseDetectorConfig` so the session can be
    reused across detector instances if needed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["performance", "lightweight", "balanced"] = "balanced"
    execution_provider: ExecutionProviderName = Field(default_factory=_default_session_provider)
    engine_cache_dir: Path = Field(default_factory=_default_engine_cache_dir)
    max_batch_size: int = 4
    fp16: bool = True
    device_id: int | None = None
    # Used only to size the warmup batch — actual inputs can be any shape.
    warmup_image_shape: tuple[int, int] = (720, 1280)
    # Behavior when the requested provider isn't available at runtime.
    # "fallback": warn and drop down (trt -> cuda -> cpu). "raise": hard error.
    on_provider_missing: Literal["fallback", "raise"] = "fallback"
    # Keep only the N highest-confidence person detections from YOLOX.
    # None = keep all detections. Set to 1 for single-person use to prevent
    # background clutter from being tracked as additional skeletons.
    max_persons: int | None = 1
    # Ceiling (bytes) on the CUDA arena for each ORT sub-session. None = size it
    # automatically from the chosen device's free VRAM (see ARENA_VRAM_FRACTION)
    # so a full batch fits instead of being throttled to the legacy 2 GiB cap.
    gpu_mem_limit: int | None = None
    # Downscale factor for YOLOX input images (0 < scale ≤ 1.0). When < 1.0,
    # images are resized before the YOLOX person-detection pass, but the original
    # full-resolution images are still used for cropping in the RTMPose pass.
    # 0.5 = half resolution (good starting point). 1.0 = no downscaling (legacy).
    yolox_image_scale: float = 1.0

    # ---- Tracking-based YOLOX skip ----
    # When True, use a velocity-based motion model to predict person bounding
    # boxes between YOLOX detections. On frames where tracking confidence is
    # high and YOLOX has run recently, YOLOX is skipped entirely and RTMPose
    # runs directly on the predicted bbox. Mirrors MediaPipe's
    # tracking-confidence pattern, adapted for wall-clock time.
    enable_tracking_skip: bool = True
    # Mean keypoint confidence below which we force a full YOLOX re-detection.
    # RTMPose SIMCC scores are per-keypoint in [0, 1]; a mean below this
    # threshold indicates the pose estimate is degrading.
    min_tracking_confidence: float = 0.3
    # Minimum wall-clock seconds between full YOLOX re-detections. While
    # tracking confidence stays above threshold, YOLOX runs at most this
    # often. Set well above frame interval so tracking gets a chance.
    min_detection_interval_seconds: float = 5.0
    # How much to expand the keypoint-derived bbox for the next frame's crop.
    # Keep this small (5%) to prevent ratcheting — the crop just needs to
    # cover the person's movement between frames, not every possible pose.
    tracking_expansion_ratio: float = 0.05


# Fraction of the selected device's free VRAM to use as the per-session CUDA
# arena ceiling when gpu_mem_limit is auto-sized. < 1.0 leaves headroom for the
# driver, the other sub-sessions, and transient spikes. The three sub-sessions
# (yolox, rtmpose, yolox_prenms) share one device, but gpu_mem_limit is a ceiling
# rather than a reservation — only the heavy rtmpose session approaches it — so a
# generous per-session cap is safe as long as actual usage stays under free VRAM.
ARENA_VRAM_FRACTION = 0.85

# Mapping from RTMPoseSessionConfig.mode → (det_url_key, det_input_size,
# pose_url_key, pose_input_size).  Mirrors rtmlib's Wholebody.MODE.
WHOLEBODY_MODE_CONFIG: dict[str, tuple[str, tuple[int, int], str, tuple[int, int]]] = {
    "performance": (
        "yolox-m", (640, 640),
        "rtmw-x-l_384x288", (288, 384),
    ),
    "lightweight": (
        "yolox-tiny", (416, 416),
        "rtmw-l-m_256x192", (192, 256),
    ),
    "balanced": (
        "yolox-m", (640, 640),
        "rtmw-x-l_256x192", (192, 256),
    ),
}


@dataclass
class _PoseCrop:
    """Holds preprocessed RTMPose input + the metadata needed to postprocess it.
    Mirrors rtmlib's RTMPose.preprocess/postprocess split, but kept around so
    we can stack N crops into one batched session.run."""
    resized_img: NDArray  # (H, W, 3)
    center: NDArray
    scale: NDArray


@dataclass
class RTMPoseSession:
    config: RTMPoseSessionConfig
    _active_provider: ExecutionProviderName

    # Direct attribute storage (replaces rtmlib's Wholebody class)
    _det_session: ort.InferenceSession | None = None
    _pose_session: ort.InferenceSession | None = None
    _det_input_size: tuple[int, int] = (640, 640)
    _pose_input_size: tuple[int, int] = (288, 384)
    _det_nms_thr: float = 0.45
    _det_score_thr: float = 0.7
    _pose_mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    _pose_std: tuple[float, float, float] = (58.395, 57.12, 57.375)
    _pose_to_openpose: bool = False

    # True when the YOLOX ONNX model accepts batch > 1. Many checkpoints are
    # exported with a static batch=1 dim; we probe this once at construction and
    # skip the stack + batched run entirely for those models.
    _yolox_supports_batch: bool = False
    # Dedicated ORT session built from the stripped prenms ONNX (backbone+decode
    # only, no Squeeze/NMS). ORT's CUDA EP runs the full compiled graph even when
    # only a subset of outputs is requested, so we can't skip Squeeze by requesting
    # pre-NMS outputs from the main session. Instead, this session is physically
    # stripped of those nodes and used for batch>1 YOLOX inference.
    _yolox_prenms_session: ort.InferenceSession | None = None
    # When True, the pose session always receives single-crop (batch=1) inputs,
    # even when multiple people are detected. Set for CoreML because it
    # JIT-compiles a new kernel on the first call with each new batch shape,
    # causing a multi-second freeze whenever the detected-person count changes.
    _per_crop_pose: bool = False
    # Cap on how many YOLOX detections are forwarded to RTMPose.
    # None = no cap. Mirrors RTMPoseSessionConfig.max_persons.
    _max_persons: int | None = None
    # Scale factor for YOLOX input images. When < 1.0, images are downscaled
    # before the YOLOX person-detection pass, but the original full-resolution
    # images are still used for cropping person regions in the RTMPose pass.
    # This gives most of the speed benefit of lower resolution without losing
    # pose estimation accuracy (since crops are always resized to a fixed input
    # size regardless of source resolution).
    _yolox_image_scale: float = 1.0
    # Tracking-based YOLOX skip (see rtmpose_tracking_state.py).
    _tracking_enabled: bool = True
    _tracking_min_confidence: float = 0.3
    _tracking_min_detection_interval: float = 5.0
    _tracking_expansion_ratio: float = 0.05
    # Debug: stores the per-camera bboxes from the most recent predict call.
    # list[NDArray | None], one entry per input image. Populated by
    # predict_batch and predict_batch_with_tracking. Read by annotators.
    last_bboxes: list[NDArray] | None = None
    # Debug: for each entry in last_bboxes, True = from YOLOX, False = tracking.
    last_bboxes_from_detector: list[bool] | None = None

    @classmethod
    def create(cls, config: RTMPoseSessionConfig | None = None) -> "RTMPoseSession":
        config = config or RTMPoseSessionConfig()

        if config.execution_provider in ("trt", "cuda"):
            ensure_cuda_dlls_loaded()

        active_provider = resolve_provider(
            requested=config.execution_provider,
            on_missing=config.on_provider_missing,
        )

        # CoreML does not support dynamic batch dims (crashes with SIGSEGV) or
        # fp16 inputs. Override those settings when the resolved provider is CoreML.
        if active_provider == "coreml":
            if config.fp16:
                logger.info("CoreML provider selected: disabling fp16 (not supported by CoreML EP)")
                config = config.model_copy(update={"fp16": False})
            if config.max_batch_size > 1:
                logger.info("CoreML provider selected: forcing max_batch_size=1 (dynamic batch dims crash CoreML EP)")
                config = config.model_copy(update={"max_batch_size": 1})

        # Resolve which physical GPU to use. Do this once here so every sub-session
        # lands on the same device.
        device_id = config.device_id
        selected_free_bytes: int | None = None
        selected_total_bytes: int | None = None
        if device_id is None and active_provider in ("cuda", "trt"):
            logger.info("RTMPoseSession: device_id not specified -- auto-selecting best CUDA device")
            device_id, selected_free_bytes, selected_total_bytes = select_best_cuda_device()
        device_id = device_id if device_id is not None else 0
        selection_source = "user-specified" if config.device_id is not None else "auto-selected"

        # Size the CUDA arena ceiling from the card's TOTAL VRAM — a stable hardware
        # constant. Free VRAM is a volatile snapshot; baking it into the session cap
        # would risk setting the ceiling too low when measured during transient GPU
        # activity (Chrome, Electron, another model, etc.). Total VRAM never changes.
        #
        # The cap is a CEILING, not a reservation — ORT only allocates what inference
        # actually needs, growing toward the cap. So "too high" costs nothing.
        gpu_mem_limit = config.gpu_mem_limit
        if gpu_mem_limit is None and active_provider in ("cuda", "trt"):
            if selected_total_bytes is None or selected_total_bytes <= 0:
                selected_total_bytes = cuda_device_total_bytes(device_id)
            if selected_total_bytes and selected_total_bytes > 0:
                gpu_mem_limit = int(selected_total_bytes * ARENA_VRAM_FRACTION)
                logger.info(
                    "RTMPose: sizing CUDA arena ceiling to %.2f GiB "
                    "(%.0f%% of %.2f GiB total on device %d)",
                    gpu_mem_limit / 1024 ** 3, ARENA_VRAM_FRACTION * 100,
                    selected_total_bytes / 1024 ** 3, device_id,
                )
                # Warn if another process is using significant GPU memory right now.
                if selected_free_bytes is None or selected_free_bytes <= 0:
                    selected_free_bytes = cuda_device_free_bytes(device_id)
                if selected_free_bytes and selected_free_bytes > 0:
                    in_use_mib = (selected_total_bytes - selected_free_bytes) // (1024 * 1024)
                    if in_use_mib > 512:
                        logger.warning(
                            "RTMPose: ~%d MiB of GPU memory is in use by other processes. "
                            "Close GPU-heavy apps (browser, game, etc.) for best performance.",
                            in_use_mib,
                        )
        if gpu_mem_limit is None:
            # CPU/CoreML path, or VRAM query failed -- fall back to the legacy default.
            gpu_mem_limit = 2 * 1024 * 1024 * 1024
        logger.info(
            "\n"
            "  ╔══════════════════════════════════════════════════════════════╗\n"
            "  ║                  RTMPose SESSION STARTUP                     ║\n"
            "  ╠══════════════════════════════════════════════════════════════╣\n"
            "  ║  provider   : %-46s║\n"
            "  ║  device_id  : %-46s║\n"
            "  ║  mode       : %-46s║\n"
            "  ║  max_batch  : %-46s║\n"
            "  ║  fp16       : %-46s║\n"
            "  ║  arena_cap  : %-46s║\n"
            "  ╚══════════════════════════════════════════════════════════════╝",
            active_provider,
            f"{device_id}  ({selection_source})",
            config.mode,
            config.max_batch_size,
            config.fp16,
            f"{gpu_mem_limit / 1024 ** 3:.2f} GiB",
        )

        config.engine_cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve mode -> model URLs + input sizes
        det_key, det_input_size, pose_key, pose_input_size = WHOLEBODY_MODE_CONFIG[
            config.mode
        ]

        # Download both ONNX models
        det_url = MODEL_URLS[det_key]
        pose_url = MODEL_URLS[pose_key]
        det_onnx_raw = str(resolve_model_path(ModelSource(url=det_url)))
        pose_onnx_raw = str(resolve_model_path(ModelSource(url=pose_url)))

        # Build ORT sessions.
        # YOLOX path: rewrite the ONNX to declare a symbolic batch dim.
        # The full YOLOX ONNX has NMS baked in — build the full det_session
        # with CUDA EP even when TRT is requested.
        det_provider = "cuda" if active_provider == "trt" else active_provider
        det_onnx_path = str(ensure_dynamic_batch(det_onnx_raw))
        det_session = build_tuned_ort_session(
            onnx_path=det_onnx_path,
            provider=det_provider,
            engine_cache_dir=config.engine_cache_dir,
            fp16=config.fp16,
            log_label="yolox",
            max_batch_size=config.max_batch_size,
            trt_set_batch_profile=True,
            device_id=device_id,
            gpu_mem_limit=gpu_mem_limit,
        )
        pose_session = build_tuned_ort_session(
            onnx_path=pose_onnx_raw,
            provider=active_provider,
            engine_cache_dir=config.engine_cache_dir,
            fp16=config.fp16,
            log_label="rtmpose",
            max_batch_size=config.max_batch_size,
            device_id=device_id,
            gpu_mem_limit=gpu_mem_limit,
        )

        yolox_supports_batch = probe_supports_batch(det_session, label="yolox")

        # Build a dedicated pre-NMS session for batch>1 YOLOX inference.
        yolox_prenms_session: ort.InferenceSession | None = None
        det_dynbatch_path = ensure_dynamic_batch(det_onnx_raw)
        prenms_path = ensure_prenms_model(det_dynbatch_path)
        if prenms_path is not None:
            logger.info(
                "Building pre-NMS ORT session for batch>1 YOLOX inference (%s) on device_id=%d",
                prenms_path.name, device_id,
            )
            yolox_prenms_session = build_tuned_ort_session(
                onnx_path=str(prenms_path),
                provider=active_provider,
                engine_cache_dir=config.engine_cache_dir,
                fp16=config.fp16,
                log_label="yolox_prenms",
                trt_set_batch_profile=True,
                max_batch_size=config.max_batch_size,
                device_id=device_id,
                gpu_mem_limit=gpu_mem_limit,
            )

        session = cls(
            config=config,
            _active_provider=active_provider,
            _det_session=det_session,
            _pose_session=pose_session,
            _det_input_size=det_input_size,
            _pose_input_size=pose_input_size,
            _yolox_supports_batch=yolox_supports_batch,
            _yolox_prenms_session=yolox_prenms_session,
            _per_crop_pose=active_provider == "coreml",
            # CoreML JIT-compiles per batch shape — varying person counts cause multi-second  freezes
            _max_persons=config.max_persons,
            _yolox_image_scale=config.yolox_image_scale,
            _tracking_enabled=config.enable_tracking_skip,
            _tracking_min_confidence=config.min_tracking_confidence,
            _tracking_min_detection_interval=config.min_detection_interval_seconds,
            _tracking_expansion_ratio=config.tracking_expansion_ratio,
        )

        # Step 3: warmup. With TRT this is what triggers engine compilation; can
        # take 1–3 minutes on first run, milliseconds on subsequent runs (cache hit).
        session._warmup()
        return session

    @property
    def active_provider(self) -> ExecutionProviderName:
        return self._active_provider

    # ------------------------------------------------------------------ inference

    def predict_single(
            self,
            image: NDArray[np.uint8],
    ) -> tuple[NDArray, NDArray]:
        """Run inference on a single image. Backwards-compatible with the
        original `Wholebody.__call__` shape: returns `(keypoints, scores)` where
        keypoints is `(num_persons, 133, 2)` and scores is `(num_persons, 133)`."""
        results = self.predict_batch([image])
        return results[0]

    def predict_pose_from_bboxes(
            self,
            images: list[NDArray[np.uint8]],
            bboxes_per_image: list[NDArray],
    ) -> list[tuple[NDArray, NDArray]]:
        """Run RTMPose on pre-detected person bboxes, skipping YOLOX entirely.

        Args:
            images: one image per camera, in the same order as bboxes_per_image.
            bboxes_per_image: one (N, 4) bbox array per camera in xyxy format.

        Returns one (keypoints, scores) tuple per input image, in the same order.
        """
        if not images:
            return []
        return self._estimate_pose_batched(images, bboxes_per_image)

    def predict_batch_with_tracking(
        self,
        images: list[NDArray[np.uint8]],
        tracking_states: list[PersonTrackingState],
    ) -> tuple[list[tuple[NDArray, NDArray]], list[PersonTrackingState]]:
        """Batched inference with tracking-based YOLOX skip.

        For each camera, decides whether to run full YOLOX+RTMPose or skip
        YOLOX and use a tracking-predicted bbox. Batches YOLOX runs together
        and RTMPose runs together for efficiency — cameras that need YOLOX
        share one ONNX call, and all cameras share one RTMPose call.

        Args:
            images: One image per camera.
            tracking_states: Per-camera tracking state, same length as images.

        Returns:
            ``(results, updated_states)`` where *results* is a list of
            ``(keypoints, scores)`` tuples (one per input image), and
            *updated_states* are the new tracking states.
        """
        if not images:
            return [], []

        n = len(images)
        if len(tracking_states) != n:
            raise ValueError(
                f"images ({n}) and tracking_states ({len(tracking_states)}) "
                f"must have the same length"
            )

        # ---- Step 1: Decide per-camera strategy ----
        # Three groups:
        #   A: run full YOLOX + RTMPose (cold start, lost track, periodic refresh)
        #   B: skip YOLOX, use tracking-predicted bbox (good track)
        #   C: skip entirely (no detection last frame AND predicted bbox is None)

        needs_detector: list[bool] = []
        tracking_bboxes: list[NDArray | None] = []
        use_tracking: list[bool] = []

        for i, state in enumerate(tracking_states):
            h, w = images[i].shape[:2]

            pred_bbox = predict_bbox_from_tracking(
                state,
                expansion_ratio=self._tracking_expansion_ratio,
                image_width=w,
                image_height=h,
            )

            run_detector = should_run_detector(
                state,
                min_tracking_confidence=self._tracking_min_confidence,
                min_detection_interval=self._tracking_min_detection_interval,
                predicted_bbox=pred_bbox,
            )

            needs_detector.append(run_detector)
            tracking_bboxes.append(pred_bbox)
            use_tracking.append(not run_detector and pred_bbox is not None)

        # ---- Step 2: Run YOLOX only on cameras that need it ----
        yolo_indices = [i for i, nd in enumerate(needs_detector) if nd]
        yolo_images = [images[i] for i in yolo_indices]

        yolo_bboxes_map: dict[int, NDArray] = {}
        if yolo_images:
            yolo_results = self._detect_persons_batched(yolo_images)
            if self._max_persons is not None:
                yolo_results = [b[: self._max_persons] for b in yolo_results]
            for idx, bboxes in zip(yolo_indices, yolo_results):
                yolo_bboxes_map[idx] = bboxes
            logger.debug(
                f"predict_batch_with_tracking: YOLOX on {len(yolo_images)}/"
                f"{n} cams (indices={yolo_indices}), "
                f"tracking on {sum(use_tracking)}/{n} cams"
            )

        # ---- Step 3: Assemble combined bboxes for RTMPose ----
        combined_bboxes: list[NDArray] = []
        for i in range(n):
            if i in yolo_bboxes_map:
                combined_bboxes.append(yolo_bboxes_map[i])
            elif use_tracking[i] and tracking_bboxes[i] is not None:
                combined_bboxes.append(
                    np.asarray(tracking_bboxes[i], dtype=np.float64).reshape(1, 4)
                )
            else:
                combined_bboxes.append(np.empty((0, 4), dtype=np.float64))

        # ---- Step 4: Run RTMPose on all crops (batched) ----
        pose_results = self._estimate_pose_batched(images, combined_bboxes)

        # ---- Step 5: Update tracking states from keypoints ----
        updated_states: list[PersonTrackingState] = []
        for i, (state, (keypoints, scores)) in enumerate(
            zip(tracking_states, pose_results)
        ):
            h, w = images[i].shape[:2]
            bbox_from_detector = i in yolo_bboxes_map

            if len(keypoints) == 0 or len(scores) == 0:
                updated_states.append(PersonTrackingState())
            else:
                new_state = update_tracking_state(
                    state,
                    keypoints=keypoints,
                    scores=scores,
                    expansion_ratio=self._tracking_expansion_ratio,
                    from_detector=bbox_from_detector,
                    image_width=w,
                    image_height=h,
                )
                if use_tracking[i]:
                    new_state.consecutive_skips = state.consecutive_skips + 1
                updated_states.append(new_state)

        # Store bboxes for debug annotation.
        combined_bboxes_for_annot: list[NDArray | None] = [
            c if len(c) > 0 else None for c in combined_bboxes
        ]
        self.last_bboxes = combined_bboxes_for_annot
        self.last_bboxes_from_detector = [
            (i in yolo_bboxes_map) for i in range(n)
        ]

        return pose_results, updated_states

    def predict_batch(
            self,
            images: list[NDArray[np.uint8]],
    ) -> list[tuple[NDArray, NDArray]]:
        """Batched inference over N images. Returns one (keypoints, scores)
        tuple per input image, in the same order. Empty detections are returned
        as zero-length arrays.

        Implementation:
          1. (Optional) Downscale images for YOLOX person detection.
          2. YOLOX preprocess each image (per-image ratio + letterbox).
          3. Stack into (N, 3, H, W); one session.run for person detection.
          4. Scale bboxes back to original-image coordinates if downscaled.
          5. RTMPose preprocess each (image, bbox) crop — using ORIGINAL images.
          6. Stack into (M, 3, H_pose, W_pose); one session.run for pose.
          7. Distribute pose outputs back per-input-image and postprocess.

        Falls back to per-image session.run (still single CUDA context) for
        either stage if the ONNX model rejects the batched input."""
        if not images:
            return []

        # ---- Stage 1: YOLOX person detection (batched) ----
        scale = self._yolox_image_scale
        if scale < 1.0:
            yolo_images = [
                cv2.resize(
                    img,
                    (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
                for img in images
            ]
            bboxes_per_image = self._detect_persons_batched(yolo_images)
            # Scale bboxes from downscaled-image coords back to original-image
            # coords so the RTMPose crop (which uses the full-res originals) works.
            inv_scale = 1.0 / scale
            bboxes_per_image = [
                b * inv_scale if len(b) > 0 else b
                for b in bboxes_per_image
            ]
        else:
            bboxes_per_image = self._detect_persons_batched(images)

        # ---- Cap detections if max_persons is set ----
        # YOLOX returns boxes sorted by descending confidence, so [:N] keeps the
        # N most confident detections and discards lower-confidence false positives.
        if self._max_persons is not None:
            bboxes_per_image = [b[: self._max_persons] for b in bboxes_per_image]

        # ---- Stage 2: RTMPose keypoint estimation (batched over all crops) ----
        # Always uses the ORIGINAL full-resolution images for cropping — never
        # the downscaled YOLOX images. This is the key insight: YOLOX only needs
        # enough resolution to find person blobs, but the pose model benefits
        # from every pixel of detail in the crop region.
        pose_results = self._estimate_pose_batched(images, bboxes_per_image)

        # Debug: store bboxes for annotation.
        self.last_bboxes = [
            b if len(b) > 0 else None for b in bboxes_per_image
        ]
        self.last_bboxes_from_detector = [True] * len(images)

        return pose_results

    # ------------------------------------------------------------------ stage 1

    def _detect_persons_batched(
            self,
            images: list[NDArray[np.uint8]],
    ) -> list[NDArray]:
        """Run YOLOX over N images. Returns one bbox array per image.

        For rtmlib checkpoints the YOLOX ONNX has NMS baked in, which contains
        a Squeeze(axis=0) that only works for batch=1. When the dynamic-batch
        rewrite has exposed pre-NMS tensors (_yolox_has_prenms=True) we bypass
        that subgraph entirely for batch>1: request only the pre-NMS outputs
        (ORT skips Squeeze+NMS) and run Python NMS per image.

        For YOLOX models without baked-in NMS the standard batched path works."""

        # Fast path: model has static batch=1 or only one image — per-image
        # YOLOX path (includes baked-in NMS, works on CUDA).
        if not self._yolox_supports_batch or len(images) == 1:
            return [_single_image_yolox(
                img, self._det_session, self._det_input_size,
                self._det_nms_thr, self._det_score_thr,
            ) for img in images]

        preprocessed = [yolox_letterbox_preprocess(img, self._det_input_size)
                        for img in images]

        # Stack to (N, 3, H, W). All padded images share the same shape because
        # YOLOX letterboxes to a fixed model_input_size.
        batch = np.stack(
            [p.transpose(2, 0, 1) for p, _ in preprocessed], axis=0,
        ).astype(np.float32, copy=False)
        batch = np.ascontiguousarray(batch)

        # Preferred batched path: prenms session (backbone+decode, no Squeeze/NMS).
        # Required when the YOLOX ONNX has baked-in NMS whose Squeeze(axis=0)
        # only tolerates batch=1. This session is physically stripped of those nodes.
        if self._yolox_prenms_session is not None:
            input_name = self._yolox_prenms_session.get_inputs()[0].name
            try:
                bboxes_batch, conf_batch = self._yolox_prenms_session.run(
                    [PRENMS_BBOX_OUTPUT, PRENMS_CONF_OUTPUT], {input_name: batch},
                )
                return [
                    _yolox_postprocess_prenms(
                        bboxes_one=bboxes_batch[i],
                        conf_one=conf_batch[i],
                        ratio=preprocessed[i][1],
                        nms_thr=self._det_nms_thr,
                        score_thr=self._det_score_thr,
                    )
                    for i in range(len(images))
                ]
            except Exception as e:
                logger.warning(
                    f"YOLOX pre-NMS batched run failed ({e!r}); "
                    f"falling back to per-image inference."
                )
                return [_single_image_yolox(
                    img, self._det_session, self._det_input_size,
                    self._det_nms_thr, self._det_score_thr,
                ) for img in images]

        # Fallback batched path: YOLOX session without baked-in NMS (uncommon).
        try:
            outputs = session_run_batched(self._det_session, batch)
        except Exception as e:
            logger.warning(
                f"YOLOX batched session.run failed ({e!r}); "
                f"falling back to per-image inference."
            )
            return [_single_image_yolox(
                img, self._det_session, self._det_input_size,
                self._det_nms_thr, self._det_score_thr,
            ) for img in images]

        det_output = outputs[0]  # (N, num_anchors, C)

        bboxes_per_image: list[NDArray] = []
        for i, (_, ratio) in enumerate(preprocessed):
            per_image_output = det_output[i:i + 1]  # keep leading 1-batch dim
            bboxes_per_image.append(
                _yolox_postprocess_one(
                    outputs_one=per_image_output,
                    ratio=ratio,
                    model_input_size=self._det_input_size,
                    nms_thr=self._det_nms_thr,
                    score_thr=self._det_score_thr,
                )
            )
        return bboxes_per_image

    # ------------------------------------------------------------------ stage 2

    def _estimate_pose_batched(
            self,
            images: list[NDArray[np.uint8]],
            bboxes_per_image: list[NDArray],
    ) -> list[tuple[NDArray, NDArray]]:
        """Run RTMPose over the union of all (image, bbox) crops. Returns one
        (keypoints, scores) tuple per input image."""

        # CoreML JIT-compiles a new kernel the first time it sees each batch
        # shape. If the number of detected people changes frame-to-frame, CoreML
        # would freeze for several seconds on each new shape. Force batch=1 by
        # processing each crop individually.
        if self._per_crop_pose:
            return [
                _single_image_rtmpose(
                    images[i], bboxes=list(bboxes_per_image[i]),
                    session=self._pose_session, pose_input_size=self._pose_input_size,
                    pose_mean=self._pose_mean, pose_std=self._pose_std,
                    to_openpose=self._pose_to_openpose,
                )
                if len(bboxes_per_image[i]) > 0 else _empty_pose_result()
                for i in range(len(images))
            ]

        # Per-image crops + a flat list of all crops with provenance.
        crops: list[_PoseCrop] = []
        crop_ranges: list[tuple[int, int]] = []  # (start, end) into crops, per image
        for image, bboxes in zip(images, bboxes_per_image):
            start = len(crops)
            if len(bboxes) == 0:
                crop_ranges.append((start, start))
                continue
            for bbox in bboxes:
                resized_img, center, scale = rtmpose_letterbox_preprocess(
                    image, bbox=np.asarray(bbox, dtype=np.float64), model_input_size=self._pose_input_size,
                    mean=self._pose_mean, std=self._pose_std,
                )
                crops.append(_PoseCrop(
                    resized_img=resized_img.astype(np.float32, copy=False),
                    center=np.asarray(center, dtype=np.float64),
                    scale=np.asarray(scale, dtype=np.float64),
                ))
            crop_ranges.append((start, len(crops)))

        if not crops:
            return [_empty_pose_result() for _ in images]

        # Stack and run.
        batch = np.stack(
            [c.resized_img.transpose(2, 0, 1) for c in crops], axis=0,
        ).astype(np.float32, copy=False)
        batch = np.ascontiguousarray(batch)

        try:
            outputs = session_run_batched(self._pose_session, batch)
        except Exception as e:
            e_str = str(e)
            if "BFCArena" in e_str or "Available memory" in e_str:
                raise MemoryError(
                    f"GPU Out of Memory in RTMPose batched pose inference: {e}"
                ) from e
            logger.warning(
                f"RTMPose batched session.run failed ({e!r}); "
                f"falling back to per-crop inference."
            )
            return [
                _single_image_rtmpose(
                    images[i], bboxes=list(bboxes_per_image[i]),
                    session=self._pose_session, pose_input_size=self._pose_input_size,
                    pose_mean=self._pose_mean, pose_std=self._pose_std,
                    to_openpose=self._pose_to_openpose,
                )
                if len(bboxes_per_image[i]) > 0 else _empty_pose_result()
                for i in range(len(images))
            ]

        # outputs from RTMPose are typically [simcc_x, simcc_y] each of shape (M, K, ...)
        simcc_x, simcc_y = outputs[0], outputs[1]

        results: list[tuple[NDArray, NDArray]] = []
        for i, (start, end) in enumerate(crop_ranges):
            if end == start:
                results.append(_empty_pose_result())
                continue
            kpts_per_person: list[NDArray] = []
            scores_per_person: list[NDArray] = []
            for ci in range(start, end):
                sx = simcc_x[ci:ci + 1]
                sy = simcc_y[ci:ci + 1]
                kpts, scr = _rtmpose_decode_one(
                    simcc_x=sx,
                    simcc_y=sy,
                    center=crops[ci].center,
                    scale=crops[ci].scale,
                    model_input_size=self._pose_input_size,
                )
                kpts_per_person.append(kpts)
                scores_per_person.append(scr)

            keypoints = np.concatenate(kpts_per_person, axis=0)
            scores = np.concatenate(scores_per_person, axis=0)
            if self._pose_to_openpose:
                keypoints, scores = convert_coco_to_openpose(keypoints, scores)
            results.append((keypoints, scores))
        return results

    # ------------------------------------------------------------------ warmup

    def _warmup(self) -> None:
        h, w = self.config.warmup_image_shape
        synthetic = np.full((h, w, 3), 128, dtype=np.uint8)
        # Hit both extremes of the TRT optimization profile / cuDNN algo cache
        # so the first real frame at either size doesn't pay re-search cost.
        # Dedup if max_batch_size == 1 (degenerate config).
        sizes = sorted({1, max(1, self.config.max_batch_size)})

        # TRT compiles engines lazily on the first session.run() call inside
        # predict_batch(). Detect first-run and show a prominent warning + live
        # elapsed-time ticker so users know it's working and not hung.
        is_trt = self._active_provider == "trt"
        has_cached_engine = is_trt and any(self.config.engine_cache_dir.glob("**/*.engine"))
        if is_trt and not has_cached_engine:
            logger.warning(
                f"\n"
                f"  ╔══════════════════════════════════════════════════════════════╗\n"
                f"  ║         TensorRT FIRST-RUN ENGINE COMPILATION                ║\n"
                f"  ╠══════════════════════════════════════════════════════════════╣\n"
                f"  ║  TRT is compiling your models to native GPU kernels.         ║\n"
                f"  ║  This happens ONCE and is cached for all future runs.        ║\n"
                f"  ║  Expected time: 1–5 minutes. Do not close the process.       ║\n"
                f"  ╚══════════════════════════════════════════════════════════════╝"
            )

        stop_event = threading.Event()

        def _tick() -> None:
            start = time.perf_counter()
            while not stop_event.wait(timeout=20):
                elapsed = time.perf_counter() - start
                m, s = divmod(int(elapsed), 60)
                if is_trt and not has_cached_engine:
                    logger.info(f"  TRT compiling ... {m}m {s:02d}s elapsed (please wait)")

        ticker = threading.Thread(target=_tick, daemon=True)
        ticker.start()
        t0 = time.perf_counter()

        warmed: list[int] = []
        for batch_size in sizes:
            try:
                self.predict_batch([synthetic] * batch_size)
            except Exception as e:
                logger.warning(
                    f"RTMPoseSession warmup at batch_size={batch_size} failed "
                    f"(non-fatal): {e!r}"
                )
                continue
            warmed.append(batch_size)

        stop_event.set()
        ticker.join(timeout=2)
        elapsed_s = time.perf_counter() - t0

        if warmed:
            logger.info(
                f"RTMPoseSession warmup OK on {self._active_provider!r} "
                f"(batch_sizes={warmed}, image_shape={(h, w)}, elapsed={elapsed_s:.1f}s)"
            )


# ============================================================================
# Free functions (kept module-level so they are easy to test in isolation)
# ============================================================================
# Provider, session-building, and batched-inference utilities are imported from
# skellytracker.trackers.gpu_utils.ort_session_utils (shared with CompositeGPUSession).

_WHOLEBODY_NUM_KPT = 133  # RTMPose wholebody keypoint count


def _empty_pose_result() -> tuple[NDArray, NDArray]:
    """Empty `(keypoints, scores)` matching rtmlib's shape conventions for
    'no detection': zero-person leading dim, correct keypoint count so callers
    that inspect shape[1] (e.g. draw_skeleton) still see a valid skeleton type."""
    return (
        np.empty((0, _WHOLEBODY_NUM_KPT, 2), dtype=np.float64),
        np.empty((0, _WHOLEBODY_NUM_KPT), dtype=np.float32),
    )


# ============================================================================
# rtmlib post-processing helpers, lifted to module scope so they can run on a
# single output slice rather than the (1, ...) batch the rtmlib classes assume.
# ============================================================================


def _yolox_postprocess_prenms(
        *,
        bboxes_one: NDArray,
        conf_one: NDArray,
        ratio: float,
        nms_thr: float,
        score_thr: float,
) -> NDArray:
    """Postprocess pre-NMS YOLOX tensors for a single image.

    Used by the pre-NMS bypass path when the YOLOX ONNX has NMS baked in.

    Args:
        bboxes_one: (num_anchors, 4) decoded bboxes in model-input pixel coords,
                    x1y1x2y2 format. The ONNX decode subgraph runs before TopK
                    so these are already in xyxy (not cxcywh).
        conf_one:   (num_anchors,) combined obj*cls confidence per anchor.
        ratio:      letterbox scale factor (model_size / image_size). Divide
                    model-coord boxes by this to get image-coord boxes.
        nms_thr:    IoU threshold for NMS.
        score_thr:  Confidence threshold for NMS pre-filter.

    Returns:
        (num_persons, 4) bboxes in image pixel coords, x1y1x2y2.
    """
    bboxes_img = bboxes_one / ratio  # model coords → image coords
    scores_2d = conf_one[:, np.newaxis]  # (num_anchors, 1) for multiclass_nms
    dets, _ = multiclass_nms(bboxes_img, scores_2d, nms_thr=nms_thr, score_thr=score_thr)
    if dets is None:
        return np.empty((0, 4), dtype=np.float64)
    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls_inds = dets[:, 5]
    keep = (final_scores > 0.3) & (final_cls_inds == 0)
    return final_boxes[keep]


def _yolox_postprocess_one(
        *,
        outputs_one: NDArray,
        ratio: float,
        model_input_size: tuple[int, int],
        nms_thr: float,
        score_thr: float,
) -> NDArray:
    """Single-image YOLOX postprocess. `outputs_one` has shape (1, num_anchors, C).
    Mirrors `rtmlib.YOLOX.postprocess` but operates on one image slice."""
    if outputs_one.shape[-1] == 4:
        grids: list[NDArray] = []
        expanded_strides: list[NDArray] = []
        strides = [8, 16, 32]
        hsizes = [model_input_size[0] // s for s in strides]
        wsizes = [model_input_size[1] // s for s in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids_ = np.concatenate(grids, 1)
        expanded_strides_ = np.concatenate(expanded_strides, 1)
        outputs_one[..., :2] = (outputs_one[..., :2] + grids_) * expanded_strides_
        outputs_one[..., 2:4] = np.exp(outputs_one[..., 2:4]) * expanded_strides_

        predictions = outputs_one[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets, _keep = multiclass_nms(
            boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr,
        )
        if dets is None:
            return np.empty((0, 4), dtype=np.float64)
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        isscore = final_scores > 0.3
        iscat = final_cls_inds == 0
        isbbox = [i and j for (i, j) in zip(isscore, iscat)]
        return final_boxes[isbbox]

    if outputs_one.shape[-1] == 5:
        # ONNX with NMS module already applied.
        boxes = outputs_one[0, :, :4]
        scores = outputs_one[0, :, 4]
        boxes /= ratio
        return boxes[scores > 0.3]

    raise RuntimeError(f"Unexpected YOLOX output shape: {outputs_one.shape}")


def _single_image_yolox(
        image: NDArray[np.uint8],
        session: ort.InferenceSession | None,
        input_size: tuple[int, int],
        nms_thr: float,
        score_thr: float,
) -> NDArray:
    """Per-image YOLOX: preprocess → session.run → postprocess.

    Replaces ``rtmlib.YOLOX.__call__``.
    """
    if session is None:
        return np.empty((0, 4), dtype=np.float64)

    padded, ratio = yolox_letterbox_preprocess(image, input_size)
    inp = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32))
    outputs = session.run(None, {session.get_inputs()[0].name: inp})
    return _yolox_postprocess_one(
        outputs_one=outputs[0],
        ratio=ratio,
        model_input_size=input_size,
        nms_thr=nms_thr,
        score_thr=score_thr,
    )


def _single_image_rtmpose(
        image: NDArray[np.uint8],
        bboxes: list[NDArray],
        session: ort.InferenceSession | None,
        pose_input_size: tuple[int, int],
        pose_mean: tuple[float, float, float],
        pose_std: tuple[float, float, float],
        to_openpose: bool = False,
) -> tuple[NDArray, NDArray]:
    """Per-image RTMPose: preprocess each bbox → session.run → postprocess.

    Replaces ``rtmlib.RTMPose.__call__``.
    """
    if session is None or len(bboxes) == 0:
        return _empty_pose_result()

    if len(bboxes) == 1 and bboxes[0].shape == (4,):
        bbox_list: list[NDArray] = [bboxes[0]]
    else:
        bbox_list = list(bboxes)

    kpts_list: list[NDArray] = []
    scores_list: list[NDArray] = []
    for bbox in bbox_list:
        resized, center, scale = rtmpose_letterbox_preprocess(
            image, bbox=np.asarray(bbox, dtype=np.float64), model_input_size=pose_input_size,
            mean=pose_mean, std=pose_std,
        )
        inp = np.ascontiguousarray(resized.transpose(2, 0, 1)[None].astype(np.float32))
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        sx, sy = outputs[0], outputs[1]
        kpts, scr = rtmpose_letterbox_postprocess(
            simcc_x=sx, simcc_y=sy,
            center=center, scale=scale,
            model_input_size=pose_input_size,
            simcc_split_ratio=2.0,
        )
        kpts_list.append(kpts)
        scores_list.append(scr)

    keypoints = np.concatenate(kpts_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    if to_openpose:
        keypoints, scores = convert_coco_to_openpose(keypoints, scores)
    return keypoints, scores


def _rtmpose_decode_one(
        *,
        simcc_x: NDArray,
        simcc_y: NDArray,
        center: NDArray,
        scale: NDArray,
        model_input_size: tuple[int, int],
        simcc_split_ratio: float = 2.0,
) -> tuple[NDArray, NDArray]:
    """Single-crop RTMPose decode. Mirrors `rtmlib.RTMPose.postprocess`."""
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = locs / simcc_split_ratio
    keypoints = keypoints / np.asarray(model_input_size) * scale
    keypoints = keypoints + center - scale / 2
    return keypoints.astype(np.float64), scores.astype(np.float32)
