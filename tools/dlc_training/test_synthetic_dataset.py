# ruff: noqa: S101
import json

import numpy as np
import pytest
from PIL import Image
from synthetic_dataset import dlc_pixel_coordinates, file_hash, read_dataset


def test_pixel_origin_conversion_matches_installed_augmentation():
    from albumentations.augmentations.geometric.functional import keypoint_hflip

    width, height = 12, 10
    renderer_xy = np.array([[[2.5, 3.5]]])
    converted, valid = dlc_pixel_coordinates(
        renderer_xy, np.ones((1, 1), bool), (width, height)
    )
    np.testing.assert_array_equal(converted, [[[2, 3]]])
    assert valid.all()
    flipped = keypoint_hflip((2.0, 3.0, 0.0, 1.0), height, width)
    np.testing.assert_array_equal(flipped[:2], [width - renderer_xy[0, 0, 0] - 0.5, 3])


def test_invalid_boundary_mask_preserves_input_and_has_no_visibility_argument():
    original = np.array([[[2.5, 3.5], [20.0, 4.0], [0.5, 0.5]]])
    saved = original.copy()
    values, valid = dlc_pixel_coordinates(
        original, np.array([[True, False, True]]), (12, 10)
    )
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(valid, [[True, False, False]])
    assert np.isnan(values[0, 1:]).all()


def fixture_dataset(tmp_path):
    images = [f"image_{index}.png" for index in range(3)]
    for name in images:
        Image.new("RGB", (12, 10)).save(tmp_path / name)
    np.savez(
        tmp_path / "labels.npz",
        image_names=images,
        feature_names=["fixed_feature"],
        projected_xy_pixels=np.full((3, 1, 2), 3.5),
        projection_valid=np.ones((3, 1), bool),
        visibility=np.zeros((3, 1), bool),
    )
    digest = file_hash(tmp_path / "labels.npz")
    manifest = {
        "label_sha256": digest,
        "pixel_convention": "origin at top-left image edge; pixel centres at half-integers",
        "feature_definition": {"feature_names": ["fixed_feature"]},
        "image_size": [12, 10],
        "image_sha256": {name: file_hash(tmp_path / name) for name in images},
    }
    membership = {
        "labels_sha256": digest,
        "splits": {
            name: {"image_indices": [index]}
            for index, name in enumerate(
                ("training", "evaluation", "excluded_guard_band")
            )
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "split_membership.json").write_text(json.dumps(membership))
    return manifest, membership


def test_occluded_finite_labels_and_exact_split_preserved(tmp_path):
    fixture_dataset(tmp_path)
    _, _, arrays, groups, coordinates, labeled = read_dataset(tmp_path)
    assert not arrays["visibility"].any()
    assert labeled.all()
    np.testing.assert_array_equal(coordinates, 3.0)
    assert groups["training"].tolist() == [0]
    assert groups["evaluation"].tolist() == [1]
    assert groups["excluded"].tolist() == [2]


def test_split_leakage_rejected(tmp_path):
    _, membership = fixture_dataset(tmp_path)
    membership["splits"]["evaluation"]["image_indices"] = [0]
    (tmp_path / "split_membership.json").write_text(json.dumps(membership))
    with pytest.raises(ValueError, match="partition"):
        read_dataset(tmp_path)


def test_production_split_keeps_reserved_final_out_of_training_selection(tmp_path):
    from synthetic_dataset import training_selection

    _, membership = fixture_dataset(tmp_path)
    old = membership["splits"]
    membership["splits"] = {
        "training": old["training"],
        "development": old["evaluation"],
        "final_evaluation": old["excluded_guard_band"],
    }
    (tmp_path / "split_membership.json").write_text(json.dumps(membership))
    _, _, _, groups, _, _ = read_dataset(tmp_path)
    validation_name, selected = training_selection(groups)
    assert validation_name == "development"
    assert selected.tolist() == [0, 1]
    assert groups["final_evaluation"].tolist() == [2]
    assert not np.intersect1d(selected, groups["final_evaluation"]).size


def test_image_mutation_rejected(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / "image_0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_dataset(tmp_path)


def test_training_guard_refuses_existing_execution_before_framework_import(tmp_path):
    from train_synthetic import validate_training_run

    (tmp_path / "training_execution.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="already started"):
        validate_training_run(tmp_path, 10, "cuda:0")


def test_training_guard_requires_explicit_device_and_positive_epochs(tmp_path):
    from train_synthetic import validate_training_run

    for epochs, device in ((0, "cpu"), (10, "auto"), (10, "cuda:-1")):
        with pytest.raises(ValueError, match="positive epochs"):
            validate_training_run(tmp_path, epochs, device)


def test_prepared_schedule_preserves_numeric_rates_through_all_milestones(tmp_path, monkeypatch):
    pytest.importorskip("deeplabcut")
    import torch
    from deeplabcut.core.config.utils import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.runners.schedulers import build_scheduler
    from train_synthetic import prepare

    source = tmp_path / "dataset"
    source.mkdir()
    fixture_dataset(source)
    output = tmp_path / "prepared"
    prepare(source, output, epochs=200)
    config = read_config_as_dict(output / "prepared_pose_config.yaml")
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], **config["runner"]["optimizer"]["params"])
    scheduler = build_scheduler(config["runner"]["scheduler"], optimizer)
    rates = []
    for _ in range(200):
        rate = optimizer.param_groups[0]["lr"]
        assert isinstance(rate, float) and rate > 0
        rates.append(rate)
        parameter.grad = torch.ones(1)
        optimizer.step()
        scheduler.step()
    assert rates[0] == 0.0005
    assert rates[90] == 0.0001
    assert rates[120] == 0.00001
    import deeplabcut
    from train_synthetic import train

    calls = []
    monkeypatch.setattr(deeplabcut, "train_network", lambda *args, **kwargs: calls.append(kwargs))
    train(output, 200, "cpu")
    assert calls[0]["batch_size"] == config["train_settings"]["batch_size"] == 8
    assert calls[0]["pytorch_cfg_updates"]["train_settings.seed"] == config["train_settings"]["seed"]
    execution = json.loads((output / "training_execution.json").read_text())
    assert execution["batch_size"] == 8


def test_cuda_statistics_initialize_context_before_allocator_access():
    from train_synthetic import initialize_cuda_statistics

    class LazyCudaContext:
        selected_device = None
        initialized = False
        measured_device = None

        def set_device(self, device):
            self.selected_device = device

        def init(self):
            assert self.selected_device == "cuda:0"
            self.initialized = True

        def reset_peak_memory_stats(self, device):
            if not self.initialized:
                raise RuntimeError("Invalid device argument")
            self.measured_device = device

    cuda = LazyCudaContext()
    with pytest.raises(RuntimeError, match="Invalid device argument"):
        cuda.reset_peak_memory_stats("cuda:0")
    initialize_cuda_statistics(cuda, "cuda:0")
    assert cuda.measured_device == "cuda:0"
