"""Compare an independent inference model against saved DLC tensors on CPU."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from model_reconstruction import load_model


def check_parity(
    config: Path, checkpoint: Path, references: Path, report_path: Path
) -> None:
    torch.set_num_threads(4)
    model = load_model(config, checkpoint)
    report = {"device": "cpu", "cases": {}}
    with torch.inference_mode():
        for path in sorted(references.glob("*.npz")):
            with np.load(path, allow_pickle=False) as fixture:
                inputs = torch.from_numpy(fixture["normalized_inputs"])
                heatmaps, offsets = model.raw_outputs(inputs)
                keypoints = model(inputs).numpy()
                values = {
                    "heatmaps": heatmaps.numpy(),
                    "location_refinements": offsets.numpy(),
                    "keypoint_values": keypoints[:, None],
                }
                results = {}
                for name, actual in values.items():
                    expected = fixture[name]
                    difference = np.abs(actual - expected)
                    results[name] = {
                        "max_abs": float(difference.max()),
                        "p95_abs": float(np.quantile(difference, 0.95)),
                    }
                    if name == "keypoint_values":
                        np.testing.assert_allclose(
                            actual[..., :2], expected[..., :2], atol=0.01, rtol=0
                        )
                        np.testing.assert_allclose(
                            actual[..., 2], expected[..., 2], atol=1e-4, rtol=0
                        )
                    else:
                        np.testing.assert_allclose(
                            actual, expected, atol=1e-5, rtol=1e-4
                        )
                report["cases"][path.stem] = results
    imported_forbidden = [
        name
        for name in sys.modules
        if name.split(".")[0] in {"deeplabcut", "dlclive", "skellyclicker"}
    ]
    if imported_forbidden:
        raise RuntimeError(
            f"Forbidden converter dependencies imported: {imported_forbidden}"
        )
    report["forbidden_imports"] = imported_forbidden
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    check_parity(args.config, args.checkpoint, args.references, args.report)
