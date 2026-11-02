"""Check a converted ONNX artifact against pinned CPU DLC reference tensors."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


def check_onnx(
    model_path: Path,
    references: Path,
    report_path: Path,
    provider: str,
    use_tf32: bool = False,
    tracker_runtime: bool = False,
) -> None:
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested provider unavailable: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(report_path.with_suffix(""))
    provider_options = {}
    if provider == "CUDAExecutionProvider":
        if tracker_runtime and sys.platform == "win32":
            from skellytracker.core.sessions.onnx_session import (
                _load_nvidia_dlls_on_windows,
            )

            _load_nvidia_dlls_on_windows()
        ort.preload_dlls()
        provider_options = {"use_tf32": "1" if use_tf32 else "0"}
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[provider],
        provider_options=[provider_options],
    )
    session.disable_fallback()
    if session.get_providers()[0] != provider:
        raise RuntimeError("Requested provider was not selected")
    input_info = session.get_inputs()[0]
    started = time.perf_counter()
    report = {
        "provider": provider,
        "onnxruntime_version": ort.__version__,
        "cases": {},
        "provider_options": session.get_provider_options(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_dtype": "float32",
        "use_tf32_requested": use_tf32,
        "tracker_runtime_loader": tracker_runtime,
        "process_id": os.getpid(),
    }
    try:
        for path in sorted(references.glob("*.npz")):
            with np.load(path, allow_pickle=False) as reference:
                inputs = reference["normalized_inputs"]
                if list(inputs.shape[2:]) != list(input_info.shape[2:]):
                    continue
                expected = reference["keypoint_values"][:, 0]
                for order_name, order in (
                    ("native", np.arange(len(inputs))),
                    ("reversed", np.arange(len(inputs))[::-1]),
                ):
                    actual = session.run(None, {input_info.name: inputs[order]})[0]
                    target = expected[order]
                    error = np.abs(actual - target)
                    report["cases"][f"{path.stem}_{order_name}"] = {
                        "batch_size": len(inputs),
                        "normalized_input_sha256": hashlib.sha256(
                            inputs[order].tobytes()
                        ).hexdigest(),
                        "position_max_abs": float(error[..., :2].max()),
                        "confidence_max_abs": float(error[..., 2].max()),
                        "position_p95_abs": float(np.quantile(error[..., :2], 0.95)),
                    }
                    np.testing.assert_allclose(
                        actual[..., :2], target[..., :2], atol=0.01, rtol=0
                    )
                    np.testing.assert_allclose(
                        actual[..., 2], target[..., 2], atol=1e-4, rtol=0
                    )
        if not report["cases"]:
            raise ValueError("No matching reference inputs for this model")
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text())
        counts = Counter(
            (
                event.get("args", {}).get("provider"),
                event.get("args", {}).get("op_name"),
            )
            for event in events
            if event.get("cat") == "Node" and event.get("args", {}).get("provider")
        )
        report["executed_nodes"] = [
            {"provider": key[0], "operation": key[1], "count": count}
            for key, count in sorted(counts.items())
        ]
        report["profile_path"] = str(profile_path)
        report["elapsed_seconds"] = time.perf_counter() - started
        if provider == "CUDAExecutionProvider" and not any(
            node["provider"] == provider
            and node["operation"] in {"Conv", "ConvTranspose"}
            for node in report["executed_nodes"]
        ):
            report["failure"] = report.get(
                "failure", "No CUDA convolution execution recorded"
            )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del session
    if report.get("failure"):
        raise RuntimeError(report["failure"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--use-tf32", action="store_true")
    parser.add_argument("--tracker-runtime", action="store_true")
    args = parser.parse_args()
    check_onnx(
        args.model,
        args.references,
        args.report,
        args.provider,
        args.use_tf32,
        args.tracker_runtime,
    )
