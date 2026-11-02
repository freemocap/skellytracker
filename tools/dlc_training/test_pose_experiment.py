"""CPU regression checks for the controlled experiment's differentiable decoder."""
import pytest
import torch

from pose_experiment import atomic_checkpoint, decode_coordinates, decode_native_offsets, gradient_comparison, parser, select_objective
from pose_objective import linearized_pose_objective, pose_objective


def test_uniform_expectation_and_interleaved_offsets():
    heatmap = torch.zeros((1, 2, 2, 3), dtype=torch.float64, requires_grad=True)
    offsets = torch.zeros((1, 4, 2, 3), dtype=torch.float64)
    offsets[:, 0] = 1
    offsets[:, 3] = 2
    offsets.requires_grad_()
    result = decode_coordinates({"heatmap": heatmap, "locref": offsets}, 8, 2)
    torch.testing.assert_close(result, torch.tensor([[[14., 8.], [12., 12.]]], dtype=torch.float64))
    result.square().sum().backward()
    assert torch.isfinite(heatmap.grad).all() and heatmap.grad.abs().sum() > 0
    assert torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0


def test_coordinate_decoder_gradcheck():
    torch.manual_seed(3)
    heatmap = (torch.randn(1, 2, 2, 2, dtype=torch.float64) * .01).requires_grad_()
    offsets = torch.randn(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda h, o: decode_coordinates({"heatmap": h, "locref": o}, 8, 7.2801),
        (heatmap, offsets),
    )


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature"):
        decode_coordinates({}, 8, 7.2801, temperature=0)


def test_cli_requires_explicit_arm_and_device():
    with pytest.raises(SystemExit):
        parser().parse_args([])


def test_atomic_checkpoint_weights_only_roundtrip(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "snapshot.pt"
    atomic_checkpoint(path, model, optimizer, {"step": 0})
    loaded = torch.load(path, weights_only=True)
    assert loaded["metadata"]["step"] == 0
    assert "optimizer" in loaded and "rng" in loaded
    assert not path.with_suffix(".tmp").exists()


def test_objective_selection_preserves_nonlinear():
    assert select_objective("nonlinear", .01) is pose_objective
    selected = select_objective("linearized", .02)
    assert selected.func is linearized_pose_objective
    assert selected.keywords == {"damping": .02}
    with pytest.raises(ValueError):
        select_objective("linearized", 0)
    with pytest.raises(ValueError):
        select_objective("unknown", .01)


def test_objective_parser():
    args = parser().parse_args([
        "--project", "p", "--pose-config", "c", "--checkpoint", "m",
        "--geometry", "g", "--output", "o", "--device", "cpu", "--steps", "1",
        "--arm", "pose", "--pose-objective", "linearized", "--temperature", ".0025",
        "--pose-weight", ".001",
        "--coordinate-decoder", "native-offset",
    ])
    assert args.pose_objective == "linearized"
    assert args.pose_damping == .01
    assert args.temperature == .0025
    assert args.pose_weight == .001
    assert args.coordinate_decoder == "native-offset"


def test_experiment_requires_explicit_loss_configuration():
    with pytest.raises(SystemExit):
        parser().parse_args([
            "--project", "p", "--pose-config", "c", "--checkpoint", "m",
            "--geometry", "g", "--output", "o", "--device", "cpu",
            "--steps", "1", "--arm", "pose",
        ])


def test_gradient_diagnostics_does_not_change_backward():
    value = torch.tensor([2., 3.], requires_grad=True)
    original = value.square().sum()
    auxiliary = original * .25
    measured = gradient_comparison(original, auxiliary, [value])
    assert measured["gradient_cosine"] == pytest.approx(1.)
    assert measured["weighted_auxiliary_gradient_l2"] == pytest.approx(measured["original_gradient_l2"] * .25)
    assert value.grad is None
    (original + auxiliary).backward()
    torch.testing.assert_close(value.grad, value.detach() * 2.5)


def test_native_offset_forward_and_unused_offsets():
    heatmap = torch.zeros(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        heatmap[:, :, 2, 3] = 1
    offsets = torch.randn(2, 6, 4, 5, dtype=torch.float64, requires_grad=True)
    actual = decode_native_offsets({'heatmap': heatmap, 'locref': offsets}, 8, 7.2801)
    expected = torch.tensor([28., 20.]) + offsets.reshape(2, 3, 2, 4, 5)[:, :, :, 2, 3] * 7.2801
    torch.testing.assert_close(actual, expected)
    shifted = offsets.detach().clone()
    shifted[:, :, 0, 0] += 100
    torch.testing.assert_close(actual, decode_native_offsets({'heatmap': heatmap, 'locref': shifted}, 8, 7.2801))
    actual.square().sum().backward()
    assert heatmap.grad is None  # No straight-through derivative of peak selection.
    mask = torch.ones_like(offsets, dtype=torch.bool)
    mask[:, :, 2, 3] = False
    assert torch.count_nonzero(offsets.grad[mask]) == 0
    assert offsets.grad[:, :, 2, 3].abs().sum() > 0


def test_native_offset_gradcheck_and_tie_convention():
    h = torch.zeros(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    o = torch.randn(1, 4, 3, 4, dtype=torch.float64, requires_grad=True)
    # A tie selects the first row-major cell, matching torch.argmax in DLC.
    expected = 4 + o.reshape(1, 2, 2, 3, 4)[:, :, :, 0, 0] * 7.2801
    torch.testing.assert_close(decode_native_offsets({'heatmap': h, 'locref': o}, 8, 7.2801), expected)
    # Differentiate offsets only: a tie is not a differentiable heatmap location.
    assert torch.autograd.gradcheck(lambda x: decode_native_offsets({'heatmap': h, 'locref': x}, 8, 7.2801), (o,))


def test_native_offset_pose_gradient_reaches_selected_offsets():
    from test_pose_objective import example, pixels
    xyz, camera, rotation, translation, valid = example()
    target = pixels(xyz, camera, rotation, translation)
    count = len(xyz)
    heatmap = torch.zeros(1, count, 2, 2, dtype=torch.float64, requires_grad=True)
    offsets = torch.zeros(1, count, 2, 2, 2, dtype=torch.float64)
    # Native decoder gives DLC centres, converted back to image-edge coordinates.
    offsets[:, :, :, 0, 0] = (target + target.new_tensor([3., -2.]) - .5 - 4) / 7.2801
    offsets = offsets.reshape(1, count * 2, 2, 2).requires_grad_()
    xy = decode_native_offsets({'heatmap': heatmap, 'locref': offsets}, 8, 7.2801) + .5
    loss = linearized_pose_objective(xy, xyz, camera, rotation, translation, valid).loss
    loss.backward()
    assert heatmap.grad is None
    assert torch.isfinite(offsets.grad).all()
    assert offsets.grad[:, :, 0, 0].abs().sum() > 0
    assert torch.count_nonzero(offsets.grad[:, :, 1, :]) == 0
    assert torch.count_nonzero(offsets.grad[:, :, 0, 1]) == 0


def test_coordinate_decoder_rejects_unknown_choice():
    with pytest.raises(ValueError, match='Unknown coordinate decoder'):
        decode_coordinates({}, 8, 1, decoder='unknown')


def test_coordinate_decoder_cli_is_required():
    with pytest.raises(SystemExit):
        parser().parse_args([
            '--project', 'p', '--pose-config', 'c', '--checkpoint', 'm',
            '--geometry', 'g', '--output', 'o', '--device', 'cpu', '--steps', '1',
            '--arm', 'pose', '--pose-objective', 'linearized', '--temperature', '.0025',
            '--pose-weight', '.001',
        ])
