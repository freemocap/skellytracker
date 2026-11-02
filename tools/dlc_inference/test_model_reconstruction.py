import copy

import numpy as np
import pytest
import torch
from model_reconstruction import ModelContract, decode_heatmaps, read_contract


def supported_config(count=5):
    return {
        "model": {
            "backbone": {
                "type": "ResNet",
                "model_name": "resnet50_gn",
                "output_stride": 16,
                "freeze_bn_stats": False,
                "freeze_bn_weights": False,
            },
            "neck": None,
            "pose_model": None,
            "backbone_output_channels": 2048,
            "heads": {
                "bodypart": {
                    "type": "HeatmapHead",
                    "heatmap_config": {
                        "channels": [2048, count],
                        "kernel_size": [3],
                        "strides": [2],
                    },
                    "locref_config": {
                        "channels": [2048, 2 * count],
                        "kernel_size": [3],
                        "strides": [2],
                    },
                    "predictor": {
                        "type": "HeatmapPredictor",
                        "apply_sigmoid": False,
                        "clip_scores": True,
                        "location_refinement": True,
                        "locref_std": 7.2801,
                    },
                }
            },
        },
        "method": "bu",
        "metadata": {
            "bodyparts": [f"feature_{i}" for i in range(count)],
            "unique_bodyparts": [],
            "individuals": ["subject_0"],
        },
        "data": {"colormode": "RGB", "inference": {"normalize_images": True}},
    }


@pytest.mark.parametrize("count", [1, 4, 13])
def test_keypoint_count_is_derived(count):
    contract = read_contract(supported_config(count))
    assert len(contract.keypoint_names) == count  # noqa: S101 -- pytest assertion


@pytest.mark.parametrize(
    "alteration",
    ["backbone", "head_channels", "transform", "extra_model", "duplicate_names"],
)
def test_unsupported_contract_fails(alteration):
    config = copy.deepcopy(supported_config())
    if alteration == "backbone":
        config["model"]["backbone"]["model_name"] = "resnet50"
    elif alteration == "head_channels":
        config["model"]["heads"]["bodypart"]["locref_config"]["channels"][-1] = 8
    elif alteration == "transform":
        config["data"]["inference"]["resize"] = {"height": 256}
    elif alteration == "extra_model":
        config["model"]["extra_head"] = {}
    else:
        config["metadata"]["bodyparts"][1] = config["metadata"]["bodyparts"][0]
    with pytest.raises(ValueError):
        read_contract(config)


def test_decoder_preserves_first_maximum_and_interleaved_offsets():
    contract = ModelContract(("first_point", "second_point"), 16, 2, 2.0, False, True)
    heatmap = torch.zeros((1, 2, 2, 3))
    heatmap[0, 0, 0, 1] = 2.0
    heatmap[0, 0, 1, 2] = 2.0  # exact tie: first row-major peak wins
    heatmap[0, 1, 1, 0] = 0.7
    offsets = torch.zeros((1, 4, 2, 3))
    offsets[0, 0:2, 0, 1] = torch.tensor([1.0, -2.0])
    offsets[0, 2:4, 1, 0] = torch.tensor([-3.0, 4.0])
    actual = decode_heatmaps(heatmap, offsets, contract).numpy()
    np.testing.assert_allclose(actual, [[[14.0, 0.0, 1.0], [-2.0, 20.0, 0.7]]])


def test_decoder_does_not_mix_batch_rows():
    contract = ModelContract(("one_point",), 16, 2, 1.0, False, True)
    heatmap = torch.tensor([[[[0.1, 0.8]]], [[[0.9, 0.2]]]])
    offsets = torch.zeros((2, 2, 1, 2))
    actual = decode_heatmaps(heatmap, offsets, contract).numpy()
    np.testing.assert_allclose(actual, [[[12, 4, 0.8]], [[4, 4, 0.9]]])


@pytest.mark.parametrize(
    "extra", ["head_activation", "detector", "conditions", "autocast"]
)
def test_unhandled_inference_behavior_is_rejected(extra):
    config = supported_config()
    if extra == "head_activation":
        config["model"]["heads"]["bodypart"]["activation"] = "sigmoid"
    elif extra == "detector":
        config["detector"] = {"type": "unexpected"}
    elif extra == "conditions":
        config["inference"] = {"conditions": {"model": "other"}}
    else:
        config["inference"] = {"autocast": {"enabled": True}}
    with pytest.raises(ValueError):
        read_contract(config)
