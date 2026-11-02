import numpy as np
import pytest
from skellytracker.core.detectors.keypoint_detectors.mediapipe_onnx.hand_geometry import (
    PalmDetection,
    palm_anchor_centers,
    decode_single_palm,
    prepare_aligned_hand,
)
from skellytracker.core.detectors.metadata import ImageTransform
from skellytracker.core.sidecar.model import InputSpec, ResizeSpec, CustomNormalization


def input_spec():
    return InputSpec(
        name="input_1",
        dtype={"fp32": "float32"},
        layout="NCHW",
        normalization=CustomNormalization(
            mode="custom", color_format="rgb", scale=1 / 255
        ),
        resize=ResizeSpec(
            method="stretch", target_size=(224, 224), interpolation="area"
        ),
    )


def test_palm_anchor_order_and_dense_decode():
    anchors = palm_anchor_centers((192, 192))
    assert anchors.shape == (2016, 2)
    np.testing.assert_allclose(anchors[:2], np.full((2, 2), 0.5 / 24))
    np.testing.assert_allclose(anchors[-6:], np.full((6, 2), 11.5 / 12))
    assert not anchors.flags.writeable
    values = np.zeros((len(anchors), 18), dtype=np.float32)
    logits = np.full((len(anchors), 1), -100, dtype=np.float32)
    values[10, 2:4] = (20, 40)
    logits[10] = 100
    palm = decode_single_palm(
        values, logits, (192, 192), ImageTransform(2.0, 2.0, 4.0, 8.0), 0.5
    )
    assert palm is not None
    expected_center = (anchors[10] * 192 - (4, 8)) / 2
    np.testing.assert_allclose(palm.box_corners.mean(axis=0), expected_center)
    np.testing.assert_allclose(palm.box_corners[1] - palm.box_corners[0], (10, 20))
    logits[:] = -100
    assert (
        decode_single_palm(values, logits, (192, 192), ImageTransform(1.0, 1.0), 0.5)
        is None
    )


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
@pytest.mark.parametrize("origin", [(180, 140), (12, 15)])
def test_alignment_round_trip_and_input_ownership(angle, origin):
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = (
        np.array([[0, 20], [-15, 0], [0, -10], [12, 0], [20, 10], [-20, 15], [-15, 10]])
        @ rotation.T
        + origin
    )
    palm = PalmDetection(
        np.stack((points.min(axis=0) - 5, points.max(axis=0) + 5)), points, 0.9
    )
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[..., 2] = 255
    original_image = image.copy()
    original_points = points.copy()
    prepared = prepare_aligned_hand(image, palm, input_spec(), (224, 224))
    assert prepared is not None
    tensor, transform = prepared
    assert (
        tensor.shape == (3, 224, 224)
        and tensor.dtype == np.float32
        and tensor.flags.c_contiguous
    )
    assert tensor[0].max() == 1 and tensor[2].max() == 0  # BGR converted to RGB once.
    xyz = np.column_stack(
        (np.array([[0, 0], [112, 112], [223, 223]], dtype=float), [0, 1, -2])
    )
    retained = xyz.copy()
    restored = transform.restore_keypoints(xyz)
    inverse = np.linalg.inv(transform.model_to_crop)
    recovered = restored[:, :2] @ inverse[:2, :2].T + inverse[:2, 2]
    np.testing.assert_allclose(recovered, xyz[:, :2], atol=1e-4)
    np.testing.assert_array_equal(xyz, retained)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(points, original_points)
    restored[:] = 0
    np.testing.assert_array_equal(xyz, retained)


def test_empty_crop_and_degenerate_orientation_do_not_invent_hand():
    palm = PalmDetection(np.array([[0.0, 0.0], [10.0, 10.0]]), np.ones((7, 2)), 0.9)
    assert (
        prepare_aligned_hand(
            np.zeros((0, 0, 3), dtype=np.uint8), palm, input_spec(), (224, 224)
        )
        is None
    )
    assert (
        prepare_aligned_hand(
            np.zeros((20, 20, 3), dtype=np.uint8), palm, input_spec(), (224, 224)
        )
        is None
    )


def test_alignment_preprocessing_matches_pinned_reference():
    import hashlib
    import importlib.util
    import os
    from pathlib import Path

    root = Path(
        os.environ.get(
            "SKELLYTRACKER_HAND_CANDIDATE_DIR", ".test-artifacts/w4/upstream"
        )
    )
    source = root / "mp_handpose.py"
    artifact = root / "handpose_estimation_mediapipe_2023feb.onnx"
    if not source.exists() or not artifact.exists():
        pytest.skip("Pinned upstream source and candidate artifact not supplied")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "7d67bfc645f8807496af26b8f393e084d72113cf5293a520b0e9905e9a331aa5"
    )
    specification = importlib.util.spec_from_file_location(
        "pinned_hand_reference", source
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    reference = module.MPHandPose(str(artifact))  # OpenCV's default CPU backend.
    generator = np.random.default_rng(3101)
    image = generator.integers(0, 256, (280, 370, 3), dtype=np.uint8)
    points = np.array(
        [
            [140, 170],
            [115, 135],
            [140, 120],
            [155, 130],
            [175, 150],
            [110, 165],
            [120, 155],
        ],
        dtype=np.float64,
    )
    corners = np.stack((points.min(axis=0) - 5, points.max(axis=0) + 5))
    palm = PalmDetection(corners, points, 0.9)
    expected = reference._preprocess(
        image, np.concatenate((corners.ravel(), points.ravel(), [0.9]))
    )[0]
    actual, _ = prepare_aligned_hand(image, palm, input_spec(), (224, 224))
    np.testing.assert_allclose(
        actual, expected[0].transpose(2, 0, 1), rtol=1e-6, atol=1e-7
    )
