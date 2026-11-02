"""Bounded warm CPU detector timings; does not measure camera/UI/application latency."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from skellytracker.core.detectors.keypoint_detectors.dlc import (
    DlcDetector,
    DlcDetectorConfig,
)
from skellytracker.core.sessions.onnx_model_spec import OnnxSessionConfig
from skellytracker.core.sessions.onnx_session import OnnxSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 20:
        raise ValueError("Use 5–20 repetitions for this bounded diagnostic")
    cases = []
    for bundle, fixture in (
        ("onnx-full-002", "full_tail"),
        ("onnx-crop", "offset_crop"),
    ):
        sidecar = args.artifact_root / bundle / "dlc_resnet50_gn.yaml"
        config = DlcDetectorConfig(model_name=str(sidecar), confidence_threshold=0.0)
        spec = DlcDetector.model_spec(config.model_name)
        session = OnnxSession.create(
            OnnxSessionConfig(
                batch_size=3,
                models=[spec],
                execution_provider="cpu",
                fp16=False,
                intra_op_num_threads=2,
            )
        )
        try:
            detector = DlcDetector.create(config, session)
            reference = np.load(args.artifact_root / "reference" / f"{fixture}.npz")
            for batch_size in (1, 2, 3) if fixture == "full_tail" else (1,):
                images = {
                    str(i): np.ascontiguousarray(reference["rgb_images"][i, ..., ::-1])
                    for i in range(batch_size)
                }
                measurements = []
                retained = None
                for iteration in range(args.repeats + 2):
                    start = time.perf_counter_ns()
                    prepared = {
                        key: detector.preprocess(image) for key, image in images.items()
                    }
                    after_preprocess = time.perf_counter_ns()
                    outputs = session.run_batched(
                        config.model_name,
                        {key: value[0] for key, value in prepared.items()},
                    )
                    after_inference = time.perf_counter_ns()
                    decoded = {
                        key: detector.postprocess(outputs[key], prepared[key][1])
                        for key in images
                    }
                    after_decode = time.perf_counter_ns()
                    if retained is not None:
                        previous, previous_xyz, previous_scores = retained
                        np.testing.assert_array_equal(previous.xyz, previous_xyz)
                        np.testing.assert_array_equal(
                            previous.visibility, previous_scores
                        )
                        if np.shares_memory(previous.xyz, decoded["0"].xyz):
                            raise AssertionError(
                                "Successive observations share mutable coordinates"
                            )
                    retained = (
                        decoded["0"],
                        decoded["0"].xyz.copy(),
                        decoded["0"].visibility.copy(),
                    )
                    if iteration >= 2:
                        measurements.append(
                            np.array(
                                [
                                    after_preprocess - start,
                                    after_inference - after_preprocess,
                                    after_decode - after_inference,
                                    after_decode - start,
                                ]
                            )
                            / 1e6
                        )
                values = np.stack(measurements)
                cases.append(
                    {
                        "fixture": fixture,
                        "batch_size": batch_size,
                        "image_shape": list(images["0"].shape),
                        "warmups": 2,
                        "repeats": args.repeats,
                        "artifact_sha256": spec.expected_sha256,
                        "input_sha256": hashlib.sha256(
                            reference["rgb_images"].tobytes()
                        ).hexdigest(),
                        "milliseconds": {
                            name: {
                                "p50": float(np.percentile(values[:, index], 50)),
                                "p95": float(np.percentile(values[:, index], 95)),
                            }
                            for index, name in enumerate(
                                (
                                    "preprocess",
                                    "inference_batch",
                                    "decode",
                                    "detector_total",
                                )
                            )
                        },
                        "samples_ms": values.tolist(),
                        "output_retention": "passed",
                    }
                )
        finally:
            session.close()
    report = {
        "provider": "CPUExecutionProvider",
        "onnxruntime": ort.__version__,
        "cpu": platform.processor(),
        "intra_op_threads": 2,
        "scope": "Detector preprocessing, inference and decoding; excludes camera, Tracker stage bookkeeping, transport and UI",
        "cases": cases,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "provider": report["provider"],
                "cases": [
                    {key: value for key, value in case.items() if key != "samples_ms"}
                    for case in cases
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
