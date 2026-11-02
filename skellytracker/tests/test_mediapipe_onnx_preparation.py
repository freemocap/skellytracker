"""CPU-only verification of pinned candidate model preparation; no downloads."""

from pathlib import Path
import os

import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.model_preparation import (
    CANDIDATE_HASHES,
    prepare_candidate_model,
)


@pytest.mark.parametrize("pack_outputs", [False, True])
@pytest.mark.parametrize("filename", list(CANDIDATE_HASHES))
def test_prepared_candidate_matches_independent_raw_frames(
    filename, tmp_path, pack_outputs
):
    import onnxruntime as ort
    import shutil

    root = Path(
        os.environ.get(
            "SKELLYTRACKER_HAND_CANDIDATE_DIR", ".test-artifacts/w4/upstream"
        )
    )
    source = root / filename
    if not source.exists():
        pytest.skip("Pinned ONNX candidate artifact not supplied")
    copied = tmp_path / filename
    shutil.copyfile(source, copied)
    prepared = prepare_candidate_model(copied, pack_outputs=pack_outputs)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    raw_session = ort.InferenceSession(
        str(copied), sess_options=options, providers=["CPUExecutionProvider"]
    )
    batched_session = ort.InferenceSession(
        str(prepared), sess_options=options, providers=["CPUExecutionProvider"]
    )
    assert batched_session.get_providers() == ["CPUExecutionProvider"]
    raw_shape = raw_session.get_inputs()[0].shape
    generator = np.random.default_rng(20261101)
    for count in (1, 3, 2, 1):
        images = generator.uniform(0, 1, (count, *raw_shape[1:])).astype(np.float32)
        expected = [raw_session.run(None, {"input_1": frame[None]}) for frame in images]
        actual = batched_session.run(
            None, {"input_1": images.transpose(0, 3, 1, 2).copy()}
        )
        if pack_outputs:
            packed_expected = []
            for result in expected:
                if len(result) == 2:
                    packed_expected.append([np.concatenate(result, axis=-1)])
                else:
                    xyz = result[0].reshape(1, -1, 3)
                    world = result[3].reshape(1, -1, 3)
                    presence = np.broadcast_to(
                        result[1].reshape(1, 1, 1), (*xyz.shape[:2], 1)
                    )
                    handedness = np.broadcast_to(
                        result[2].reshape(1, 1, 1), presence.shape
                    )
                    packed_expected.append(
                        [np.concatenate((xyz, presence, handedness, world), axis=-1)]
                    )
            expected = packed_expected
        for output_index, output in enumerate(actual):
            reference = np.concatenate(
                [result[output_index] for result in expected], axis=0
            )
            np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-4)
    assert prepare_candidate_model(copied, pack_outputs=pack_outputs) == prepared


def test_candidate_preparation_rejects_unknown_bytes(tmp_path):
    source = tmp_path / "unknown.onnx"
    source.write_bytes(b"not the inspected model")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_candidate_model(source)
