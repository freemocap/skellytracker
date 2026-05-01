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
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from rtmlib import Wholebody
from rtmlib.tools.object_detection.post_processings import multiclass_nms
from rtmlib.tools.pose_estimation.post_processings import convert_coco_to_openpose, get_simcc_maximum

from skellytracker.trackers.rtmpose_tracker._yolox_dynamic_batch import (
    PRENMS_BBOX_OUTPUT,
    PRENMS_CONF_OUTPUT,
    ensure_dynamic_batch,
    ensure_prenms_model,
)

logger = logging.getLogger(__name__)

ExecutionProviderName = Literal["trt", "cuda", "cpu"]


def _default_engine_cache_dir() -> Path:
    return Path.home() / ".cache" / "skellytracker" / "trt_engines"


class RTMPoseSessionConfig(BaseModel):
    """Configuration for a tuned RTMPose ONNX session.

    Independent of the per-frame `RTMPoseDetectorConfig` so the session can be
    reused across detector instances if needed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["performance", "lightweight", "balanced"] = "balanced"
    execution_provider: ExecutionProviderName = "trt"
    engine_cache_dir: Path = Field(default_factory=_default_engine_cache_dir)
    max_batch_size: int = 4
    fp16: bool = True
    # Used only to size the warmup batch — actual inputs can be any shape.
    warmup_image_shape: tuple[int, int] = (720, 1280)
    # Behavior when the requested provider isn't available at runtime.
    # "fallback": warn and drop down (trt -> cuda -> cpu). "raise": hard error.
    on_provider_missing: Literal["fallback", "raise"] = "fallback"


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
    wholebody: Wholebody
    _active_provider: ExecutionProviderName
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

    @classmethod
    def create(cls, config: RTMPoseSessionConfig | None = None) -> "RTMPoseSession":
        config = config or RTMPoseSessionConfig()

        # rtmlib's `device` arg takes a string in {"cuda", "cpu", "rocm", "mps"}.
        # We resolve TRT down to "cuda" for the rtmlib container and override the
        # session afterwards (TRT has no separate rtmlib code path).
        rtmlib_device = "cuda" if config.execution_provider in ("trt", "cuda") else "cpu"
        if rtmlib_device == "cuda":
            _ensure_cuda_dlls_loaded()

        active_provider = _resolve_provider(
            requested=config.execution_provider,
            on_missing=config.on_provider_missing,
        )

        config.engine_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Constructing RTMPoseSession (mode={config.mode!r}, "
            f"requested_provider={config.execution_provider!r}, "
            f"active_provider={active_provider!r}, "
            f"max_batch_size={config.max_batch_size}, fp16={config.fp16})"
        )

        # Step 1: let rtmlib construct Wholebody normally. This downloads the
        # ONNX files and builds a basic ORT session per sub-model.
        wholebody = Wholebody(
            to_openpose=False,
            mode=config.mode,
            backend="onnxruntime",
            device=rtmlib_device,
        )

        # Step 2: replace each sub-model's session with a tuned one so we get
        # provider options, TRT engine cache, and explicit SessionOptions.
        # The .onnx_model attribute on each BaseTool is the absolute path to
        # the (already-downloaded) model file.
        # YOLOX path: rewrite the ONNX to declare a symbolic batch dim before
        # opening the ORT session. rtmlib's checkpoints ship with static batch=1,
        # which would otherwise force serial per-image inference for N cameras.
        #
        # The full YOLOX ONNX has NMS baked in, which contains a TopK node whose
        # K exceeds TRT's compile-time limit (~3840). Build the full det_session
        # with CUDA EP even when TRT is requested — it is only used as a per-image
        # fallback (N=1 or prenms unavailable). The prenms session below gets TRT.
        det_provider = "cuda" if active_provider == "trt" else active_provider
        det_session = _build_tuned_ort_session(
            onnx_path=wholebody.det_model.onnx_model,
            provider=det_provider,
            engine_cache_dir=config.engine_cache_dir,
            fp16=config.fp16,
            log_label="yolox",
            make_dynamic_batch=True,
            max_batch_size=config.max_batch_size,
        )
        wholebody.det_model.session = det_session
        wholebody.pose_model.session = _build_tuned_ort_session(
            onnx_path=wholebody.pose_model.onnx_model,
            provider=active_provider,
            engine_cache_dir=config.engine_cache_dir,
            fp16=config.fp16,
            log_label="rtmpose",
            make_dynamic_batch=False,
            max_batch_size=config.max_batch_size,
        )

        # Probe whether YOLOX accepts batch > 1. ONNX exports often have a fixed
        # batch=1 dimension; detect it once here so _detect_persons_batched can
        # skip the doomed stack + session.run entirely (no warning spam).
        yolox_supports_batch = _probe_supports_batch(det_session, label="yolox")

        # Build a dedicated pre-NMS session for batch>1 YOLOX inference.
        # ORT's CUDA EP runs the full compiled graph even when only a subset of
        # outputs is requested, so we can't skip Squeeze_axis0 by requesting pre-NMS
        # outputs from the main session. Instead we use onnx.utils.extract_model to
        # create a stripped backbone-only ONNX and build a separate session from it.
        yolox_prenms_session: ort.InferenceSession | None = None
        det_dynbatch_path = ensure_dynamic_batch(wholebody.det_model.onnx_model)
        prenms_path = ensure_prenms_model(det_dynbatch_path)
        if prenms_path is not None:
            logger.info(
                f"Building pre-NMS ORT session for batch>1 YOLOX inference "
                f"({prenms_path.name})"
            )
            yolox_prenms_session = _build_tuned_ort_session(
                onnx_path=str(prenms_path),
                provider=active_provider,
                engine_cache_dir=config.engine_cache_dir,
                fp16=config.fp16,
                log_label="yolox_prenms",
                make_dynamic_batch=False,
                trt_set_batch_profile=True,  # prenms ONNX already has dynamic batch
                max_batch_size=config.max_batch_size,
            )

        session = cls(
            config=config,
            wholebody=wholebody,
            _active_provider=active_provider,
            _yolox_supports_batch=yolox_supports_batch,
            _yolox_prenms_session=yolox_prenms_session,
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

    def predict_batch(
            self,
            images: list[NDArray[np.uint8]],
    ) -> list[tuple[NDArray, NDArray]]:
        """Batched inference over N images. Returns one (keypoints, scores)
        tuple per input image, in the same order. Empty detections are returned
        as zero-length arrays.

        Implementation:
          1. YOLOX preprocess each image (per-image ratio + letterbox).
          2. Stack into (N, 3, H, W); one session.run for person detection.
          3. RTMPose preprocess each (image, bbox) crop across all images.
          4. Stack into (M, 3, H_pose, W_pose); one session.run for pose.
          5. Distribute pose outputs back per-input-image and postprocess.

        Falls back to per-image session.run (still single CUDA context) for
        either stage if the ONNX model rejects the batched input."""
        if not images:
            return []

        # ---- Stage 1: YOLOX person detection (batched) ----
        bboxes_per_image = self._detect_persons_batched(images)

        # ---- Stage 2: RTMPose keypoint estimation (batched over all crops) ----
        return self._estimate_pose_batched(images, bboxes_per_image)

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
        det = self.wholebody.det_model

        # Fast path: model has static batch=1 or only one image — use rtmlib's
        # standard single-image YOLOX path (includes baked-in NMS, works on CUDA).
        if not self._yolox_supports_batch or len(images) == 1:
            return [det(img) for img in images]

        preprocessed = [det.preprocess(img) for img in images]

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
                        nms_thr=det.nms_thr,
                        score_thr=det.score_thr,
                    )
                    for i in range(len(images))
                ]
            except Exception as e:
                logger.warning(
                    f"YOLOX pre-NMS batched run failed ({e!r}); "
                    f"falling back to per-image inference."
                )
                return [det(img) for img in images]

        # Fallback batched path: YOLOX session without baked-in NMS (uncommon).
        try:
            outputs = _session_run_batched(det.session, batch)
        except Exception as e:
            logger.warning(
                f"YOLOX batched session.run failed ({e!r}); "
                f"falling back to per-image inference."
            )
            return [det(img) for img in images]

        det_output = outputs[0]  # (N, num_anchors, C)

        bboxes_per_image: list[NDArray] = []
        for i, (_, ratio) in enumerate(preprocessed):
            per_image_output = det_output[i:i + 1]  # keep leading 1-batch dim
            bboxes_per_image.append(
                _yolox_postprocess_one(
                    outputs_one=per_image_output,
                    ratio=ratio,
                    model_input_size=det.model_input_size,
                    nms_thr=det.nms_thr,
                    score_thr=det.score_thr,
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
        pose = self.wholebody.pose_model

        # Materialize mean/std as arrays once (rtmlib does this lazily inside
        # preprocess but mutates self.mean each time — we stay out of that path).
        mean = np.asarray(pose.mean, dtype=np.float32) if pose.mean is not None else None
        std = np.asarray(pose.std, dtype=np.float32) if pose.std is not None else None

        # Per-image crops + a flat list of all crops with provenance.
        crops: list[_PoseCrop] = []
        crop_ranges: list[tuple[int, int]] = []  # (start, end) into crops, per image
        for image, bboxes in zip(images, bboxes_per_image):
            start = len(crops)
            if len(bboxes) == 0:
                # rtmlib's behavior: when no bbox, use the whole image as one bbox.
                # But for centralized inference we'd rather return an empty
                # detection than silently invent a person — keeps the per-camera
                # output honest. Aggregator handles "no skeleton this frame".
                crop_ranges.append((start, start))
                continue
            for bbox in bboxes:
                resized_img, center, scale = pose.preprocess(image, bbox)
                if mean is not None and std is not None:
                    # `pose.preprocess` already did the normalization — it
                    # mutates self.mean/std so we just trust its output.
                    pass
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
            outputs = _session_run_batched(pose.session, batch)
        except Exception as e:
            logger.warning(
                f"RTMPose batched session.run failed ({e!r}); "
                f"falling back to per-crop inference."
            )
            return [
                self.wholebody.pose_model(images[i], bboxes=list(bboxes_per_image[i]))
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
                    model_input_size=pose.model_input_size,
                )
                kpts_per_person.append(kpts)
                scores_per_person.append(scr)

            keypoints = np.concatenate(kpts_per_person, axis=0)
            scores = np.concatenate(scores_per_person, axis=0)
            if pose.to_openpose:
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


def _resolve_provider(
        *,
        requested: ExecutionProviderName,
        on_missing: Literal["fallback", "raise"],
) -> ExecutionProviderName:
    """Pick the actual EP to use given what's available. Falls back trt -> cuda
    -> cpu unless `on_missing="raise"`."""
    available = set(ort.get_available_providers())
    needs = {
        "trt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if needs[requested] in available:
        return requested
    if on_missing == "raise":
        raise RuntimeError(
            f"Requested execution_provider={requested!r} but ONNX Runtime "
            f"only sees providers={sorted(available)}. Install onnxruntime-gpu "
            f"(and a TensorRT-enabled build for trt) to enable GPU execution."
        )
    # Fallback chain.
    fallback_order: list[ExecutionProviderName] = ["trt", "cuda", "cpu"]
    start = fallback_order.index(requested)
    for candidate in fallback_order[start:]:
        if needs[candidate] in available:
            if candidate != requested:
                logger.warning(
                    f"Requested execution_provider={requested!r} not available "
                    f"({sorted(available)}); falling back to {candidate!r}."
                )
            return candidate
    # Should be unreachable — CPU EP is always available.
    raise RuntimeError(f"No supported ONNX Runtime providers found: {sorted(available)}")


def _validate_engine_cache(engine_cache_dir: Path) -> None:
    """Delete stale TRT engine files when TRT or ORT version has changed.

    Writes a cache_manifest.json recording the current versions. On the next call,
    if stored versions differ from current, all .engine and .timing files are
    deleted so ORT/TRT will recompile fresh engines.
    """
    import json
    manifest_path = engine_cache_dir / "cache_manifest.json"

    try:
        import tensorrt as _trt
        current_trt = _trt.__version__
    except Exception:
        current_trt = "unavailable"
    current = {"trt_version": current_trt, "ort_version": ort.__version__}

    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text())
        except Exception:
            stored = {}
        if stored != current:
            logger.warning(
                f"TRT engine cache version mismatch "
                f"(stored={stored}, current={current}). "
                f"Deleting stale engines in {engine_cache_dir} — "
                f"they will be recompiled on this run."
            )
            for stale in engine_cache_dir.glob("*.engine"):
                stale.unlink(missing_ok=True)
            for stale in engine_cache_dir.glob("*.timing"):
                stale.unlink(missing_ok=True)

    manifest_path.write_text(json.dumps(current, indent=2))


def _build_tuned_ort_session(
        *,
        onnx_path: str,
        provider: ExecutionProviderName,
        engine_cache_dir: Path,
        fp16: bool,
        log_label: str,
        make_dynamic_batch: bool = False,
        max_batch_size: int = 8,
        trt_set_batch_profile: bool | None = None,
) -> ort.InferenceSession:
    """Construct an ORT session with explicit SessionOptions + provider options.

    When `make_dynamic_batch=True`, the ONNX model is first rewritten so its
    leading input/output axes are symbolic. This is needed for YOLOX checkpoints
    shipped with static batch=1.

    `trt_set_batch_profile`: when True, sets the TRT optimization profile
    (min=1, opt=max_batch_size, max=max_batch_size) even if `make_dynamic_batch`
    is False. Required for sessions whose ONNX already has a symbolic batch dim
    (e.g. the prenms session, built from the already-dynamic-batched YOLOX ONNX).
    Defaults to `make_dynamic_batch` when not set.
    """
    if trt_set_batch_profile is None:
        trt_set_batch_profile = make_dynamic_batch

    if make_dynamic_batch:
        onnx_path = str(ensure_dynamic_batch(onnx_path))

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # We expect this session to be pegged on the GPU. CPU-side parallelism per
    # session would just cause cache thrash with the camera-node + aggregator
    # processes that share the same cores.
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # TensorRT engines compiled against a static-batch graph cannot be reused
    # for a dynamic-batch graph (and vice versa). Bucket dynamic-batch caches
    # under a versioned subdir so the two never collide. This applies both when
    # we rewrote the ONNX (make_dynamic_batch) and when we set a TRT batch
    # profile on an already-dynamic ONNX (trt_set_batch_profile).
    if trt_set_batch_profile:
        engine_cache_dir = engine_cache_dir / "dynbatch_v1"
        engine_cache_dir.mkdir(parents=True, exist_ok=True)

    if provider == "trt":
        _validate_engine_cache(engine_cache_dir)

    providers: list[tuple[str, dict] | str] = []
    if provider == "trt":
        trt_options: dict[str, Any] = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(engine_cache_dir),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(engine_cache_dir),
            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GiB
        }
        if trt_set_batch_profile:
            trt_options.update(
                _trt_dynamic_batch_profile(
                    onnx_path=onnx_path,
                    max_batch_size=max(1, max_batch_size),
                )
            )
        providers.append(("TensorrtExecutionProvider", trt_options))
        # Always include CUDA + CPU as fallback EPs in the provider list. If
        # TRT can't compile a subgraph, ORT falls through to CUDA.
        providers.append((
            "CUDAExecutionProvider",
            _cuda_provider_options(),
        ))
        providers.append("CPUExecutionProvider")
    elif provider == "cuda":
        providers.append((
            "CUDAExecutionProvider",
            _cuda_provider_options(),
        ))
        providers.append("CPUExecutionProvider")
    else:  # cpu
        providers.append("CPUExecutionProvider")

    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    logger.info(f"Building tuned ORT session for {log_label!r} with providers={provider_names}")

    if provider == "trt":
        logger.info(
            f"  TRT: building {log_label!r} session "
            f"(engine cache: {engine_cache_dir}) — "
            f"first-run TRT compilation can take 1-5 minutes; "
            f"subsequent runs load from cache instantly."
        )

    t0 = time.perf_counter()
    session = ort.InferenceSession(
        path_or_bytes=onnx_path,
        sess_options=sess_options,
        providers=providers,
    )
    elapsed_s = time.perf_counter() - t0
    actual = session.get_providers()
    logger.info(f"  {log_label!r} session ready in {elapsed_s:.1f}s (active providers: {actual})")
    if provider == "trt" and elapsed_s > 30:
        logger.info(
            f"  TRT engine for {log_label!r} compiled and cached to {engine_cache_dir} — "
            f"next run will load in seconds."
        )
    return session


def _trt_dynamic_batch_profile(
        *,
        onnx_path: str,
        max_batch_size: int,
) -> dict[str, str]:
    """Build TensorRT optimization profile shape strings for a YOLOX ONNX with
    a symbolic batch axis.

    TRT requires explicit min/opt/max shapes when any input dim is dynamic.
    We pin H/W from the ONNX (YOLOX letterboxes to a fixed model_input_size)
    and let batch range from 1 to `max_batch_size`.
    """
    model = onnx.load(onnx_path)
    inp = model.graph.input[0]
    name = inp.name
    dims = inp.type.tensor_type.shape.dim
    # YOLOX is (batch, channels, H, W). Pull C/H/W from the static dims.
    fixed_shape = []
    for i, d in enumerate(dims):
        if i == 0:
            continue  # batch — handled separately
        if d.HasField("dim_value") and d.dim_value > 0:
            fixed_shape.append(int(d.dim_value))
        else:
            # Shouldn't happen for YOLOX, but bail rather than guess.
            logger.warning(
                f"TRT profile: input dim {i} of {name!r} is non-static; "
                f"skipping dynamic-batch profile."
            )
            return {}
    fixed_str = "x".join(str(x) for x in fixed_shape)
    min_str = f"{name}:1x{fixed_str}"
    opt_str = f"{name}:{max_batch_size}x{fixed_str}"
    max_str = f"{name}:{max_batch_size}x{fixed_str}"
    logger.info(
        f"TRT optimization profile for {name!r}: "
        f"min={min_str.split(':')[1]}, opt={opt_str.split(':')[1]}, "
        f"max={max_str.split(':')[1]}"
    )
    return {
        "trt_profile_min_shapes": min_str,
        "trt_profile_opt_shapes": opt_str,
        "trt_profile_max_shapes": max_str,
    }


def _cuda_provider_options() -> dict:
    return {
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": True,
        # 2 GiB. Enough for both YOLOX + RTMPose under either model size.
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
    }


def _probe_supports_batch(session: ort.InferenceSession, label: str = "") -> bool:
    """Return True if the session's first input has a dynamic (or > 1) batch dim.

    ONNX exports from rtmlib's humanart YOLOX checkpoint ship with a static
    batch=1 first dimension. Attempting session.run with N > 1 raises
    `InvalidArgument: Got invalid dimensions for input: index: 0 Got: N Expected: 1`.
    We detect this once at construction so callers can skip the doomed path."""
    try:
        first_input_shape = session.get_inputs()[0].shape
    except Exception:
        return True  # can't probe — optimistic
    if not first_input_shape:
        return True
    batch_dim = first_input_shape[0]
    # Dynamic batch: the dim is a string like "batch" or "N".
    if isinstance(batch_dim, str):
        return True
    # Static dim: must be > 1 to support any real batching.
    supports = int(batch_dim) > 1
    if not supports:
        logger.debug(
            f"{label!r} ONNX model has static batch_size={batch_dim}; "
            f"per-image inference will be used (single CUDA context still applies)."
        )
    return supports


def _session_run_batched(session: ort.InferenceSession, batch: NDArray) -> list[NDArray]:
    """Run an ORT session with a (N,3,H,W) batched input. Wraps the same
    boilerplate that rtmlib's `BaseTool.inference` does for the single-image case."""
    sess_input_name = session.get_inputs()[0].name
    sess_output_names = [o.name for o in session.get_outputs()]
    return session.run(sess_output_names, {sess_input_name: batch})


_WHOLEBODY_NUM_KPT = 133  # RTMPose wholebody keypoint count


def _empty_pose_result() -> tuple[NDArray, NDArray]:
    """Empty `(keypoints, scores)` matching rtmlib's shape conventions for
    'no detection': zero-person leading dim, correct keypoint count so callers
    that inspect shape[1] (e.g. draw_skeleton) still see a valid skeleton type."""
    return (
        np.empty((0, _WHOLEBODY_NUM_KPT, 2), dtype=np.float64),
        np.empty((0, _WHOLEBODY_NUM_KPT), dtype=np.float32),
    )


def _ensure_cuda_dlls_loaded() -> None:
    """Layered Windows DLL discovery so cuDNN's lazy LoadLibrary calls succeed.
    Delegates to the existing helper in `rtmpose_detector` to avoid duplicating
    the three-layer fix described there."""
    if sys.platform != "win32":
        ort.preload_dlls()
        return
    from skellytracker.trackers.rtmpose_tracker.rtmpose_detector import (
        _make_nvidia_pip_dlls_discoverable_on_windows,
    )
    _make_nvidia_pip_dlls_discoverable_on_windows()
    ort.preload_dlls()


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
