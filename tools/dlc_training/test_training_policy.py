# ruff: noqa: S101
from copy import deepcopy

import pytest
from training_policy import configure_training_policy


def test_default_preserves_dlc_transforms_and_inference():
    pose = {
        "data": {
            "train": {
                "affine": {"rotation": 30},
                "crop_sampling": {"width": 448},
                "motion_blur": True,
                "hflip": True,
            },
            "inference": {"normalize_images": True},
        }
    }
    original = deepcopy(pose)
    configure_training_policy(pose, "dlc-default")
    assert pose["data"]["train"] == original["data"]["train"] | {"hflip": False}
    assert pose["data"]["inference"] == original["data"]["inference"]


def test_diagnostic_is_explicit_and_does_not_alias_inference():
    pose = {"data": {"train": {"affine": {}}, "inference": {"normalize_images": True}}}
    configure_training_policy(pose, "diagnostic-none")
    assert pose["data"]["train"] == {"normalize_images": True, "hflip": False}
    assert "hflip" not in pose["data"]["inference"]
    with pytest.raises(ValueError, match="Unknown augmentation"):
        configure_training_policy(pose, "typo")


def test_installed_dlc_default_builds_and_preserves_keypoint_identity():
    from pathlib import Path

    import numpy as np
    import yaml

    dlc = pytest.importorskip("deeplabcut")
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms

    config = yaml.safe_load(
        (
            Path(dlc.__file__).parent
            / "pose_estimation_pytorch/config/base/aug_default.yaml"
        ).read_text()
    )
    pose = {"data": config}
    configure_training_policy(pose, "dlc-default")
    transform = build_transforms(pose["data"]["train"])
    for _ in range(8):
        result = transform(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            keypoints=[[320.0, 240.0], [100.0, 100.0]],
            class_labels=[0, 1],
            bboxes=[],
            bbox_labels=[],
        )
        assert result["class_labels"] == [0, 1]
        assert result["image"].shape == (448, 448, 3)
        assert np.isfinite(result["image"]).all()
        assert np.asarray(result["keypoints"]).shape == (2, 2)
