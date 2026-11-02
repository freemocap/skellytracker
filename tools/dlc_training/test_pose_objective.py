# ruff: noqa: S101
"""CPU-only numerical tests; no DLC import, model weights, camera, or GPU work."""
import pytest
import torch

from pose_objective import PoseFailure, _project_and_jacobian, _skew, pose_objective, linearized_pose_objective


def example(batch=1, dtype=torch.float64):
    geometry = torch.tensor([[-1., -.6, -.3], [1., -.6, -.3], [-1., .6, -.3],
                             [1., .6, -.3], [-.7, -.4, .5], [.7, -.4, .5],
                             [-.7, .4, .5], [.7, .4, .5]], dtype=dtype)
    rotation = torch.matrix_exp(_skew(torch.tensor([[.1, -.15, .04]], dtype=dtype))).repeat(batch, 1, 1)
    translation = torch.tensor([[.2, -.1, 7.]], dtype=dtype).repeat(batch, 1)
    camera = torch.tensor([[[700., 0., 320.], [0., 680., 240.], [0., 0., 1.]]], dtype=dtype).repeat(batch, 1, 1)
    mask = torch.ones((batch, len(geometry)), dtype=torch.bool)
    return geometry, camera, rotation, translation, mask


def pixels(geometry, camera, rotation, translation):
    points = geometry.unsqueeze(0) @ rotation.transpose(-1, -2) + translation[:, None, :]
    homogeneous = points @ camera.transpose(-1, -2)
    return homogeneous[..., :2] / homogeneous[..., 2:]


def shifted_pose(rotation, translation):
    twist = rotation.new_tensor([.025, -.035, .02]).expand(len(rotation), 3)
    moved_rotation = torch.matrix_exp(_skew(twist)) @ rotation
    moved_translation = translation + translation.new_tensor([.08, -.04, .12])
    return moved_rotation, moved_translation


def test_target_pixels_have_zero_loss_and_zero_gradient():
    geometry, camera, rotation, translation, mask = example()
    xy = pixels(geometry, camera, rotation, translation).requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.valid_samples.all()
    assert result.loss.item() < 1e-24
    result.loss.backward()
    assert torch.isfinite(xy.grad).all()
    assert xy.grad.abs().max() < 1e-12
    torch.testing.assert_close(result.fitted_rotation, rotation, atol=1e-10, rtol=0)
    torch.testing.assert_close(result.fitted_translation, translation, atol=1e-10, rtol=0)


def test_known_perturbation_recovery_and_normalized_displacement():
    geometry, camera, rotation, translation, mask = example()
    target_rotation, target_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, target_rotation, target_translation).requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.valid_samples.all()
    torch.testing.assert_close(result.fitted_rotation, target_rotation, atol=2e-5, rtol=0)
    torch.testing.assert_close(result.fitted_translation, target_translation, atol=2e-5, rtol=0)
    expected = ((geometry @ target_rotation[0].T + target_translation[0]) -
                (geometry @ rotation[0].T + translation[0])).square().sum(-1).mean() / torch.pdist(geometry).max().square()
    torch.testing.assert_close(result.loss, expected, atol=1e-7, rtol=1e-4)
    result.loss.backward()
    assert torch.isfinite(xy.grad).all()
    assert xy.grad.abs().max() > 1e-7


def test_pixel_gradcheck_through_all_pose_iterations():
    geometry, camera, rotation, translation, mask = example()
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda values: pose_objective(values, geometry, camera, rotation, translation, mask, iterations=5).loss,
        (xy,), eps=1e-4, atol=2e-6, rtol=3e-3,
    )


def test_analytic_projection_jacobian_matches_finite_difference():
    points = torch.tensor([[[.3, -.5, 4.], [1., .2, 6.]]], dtype=torch.float64)
    _, jacobian = _project_and_jacobian(points, 1e-6)
    def project(twist):
        moved = points @ torch.matrix_exp(_skew(twist[:3])).T + twist[3:]
        return moved[..., :2] / moved[..., 2:]
    automatic = torch.autograd.functional.jacobian(project, torch.zeros(6, dtype=torch.float64))
    torch.testing.assert_close(jacobian, automatic, atol=1e-12, rtol=1e-12)
    epsilon = 1e-6
    numerical = torch.stack([(project(epsilon * axis) - project(-epsilon * axis)) / (2 * epsilon)
                             for axis in torch.eye(6, dtype=torch.float64)], -1)
    torch.testing.assert_close(jacobian, numerical, atol=1e-9, rtol=1e-9)


def test_batched_variable_masks_match_individual_fits_and_masked_gradient_zero():
    geometry, camera, rotation, translation, mask = example(batch=3)
    mask[1, 4:] = False
    mask[2, :2] = False
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation).requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.valid_samples.all(), result.failure_codes
    for index in range(3):
        single = pose_objective(xy[index:index+1], geometry, camera[index:index+1], rotation[index:index+1], translation[index:index+1], mask[index:index+1])
        torch.testing.assert_close(single.per_sample_loss[0], result.per_sample_loss[index], rtol=1e-8, atol=1e-10)
    result.loss.backward()
    assert torch.isfinite(xy.grad).all()
    assert (xy.grad[~mask] == 0).all()


def test_insufficient_and_collinear_geometry_report_failure_not_zero_error():
    geometry, camera, rotation, translation, mask = example(batch=2)
    mask[0, 3:] = False
    mask[1] = False
    xy = pixels(geometry, camera, rotation, translation).requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert not result.valid_samples.any()
    assert (result.failure_codes == PoseFailure.INSUFFICIENT_POINTS).all()
    assert result.failure_counts['insufficient_points'].item() == 2
    assert torch.isnan(result.per_sample_loss).all()
    result.loss.backward()
    assert torch.isfinite(xy.grad).all() and (xy.grad == 0).all()
    line = geometry.clone(); line[:, 1:] = 0
    line[:, 0] = torch.arange(len(line), dtype=line.dtype)
    result = pose_objective(pixels(line, camera, rotation, translation), line, camera, rotation, translation, torch.ones_like(mask))
    assert (result.failure_codes == PoseFailure.UNOBSERVABLE_TARGET).all()


def test_units_equivalence_and_object_origin_equivalence():
    geometry, camera, rotation, translation, mask = example()
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation)
    baseline = pose_objective(xy, geometry, camera, rotation, translation, mask)
    scaled = pose_objective(xy, geometry*1000, camera, rotation, translation*1000, mask)
    torch.testing.assert_close(scaled.loss, baseline.loss, atol=1e-12, rtol=1e-9)
    torch.testing.assert_close(scaled.fitted_translation/1000, baseline.fitted_translation, atol=1e-10, rtol=1e-9)
    offset = geometry.new_tensor([10., -20., 30.])
    relocated = pose_objective(xy, geometry+offset, camera, rotation, translation-(rotation@offset), mask)
    torch.testing.assert_close(relocated.loss, baseline.loss, atol=1e-12, rtol=1e-8)


def test_nonfinite_predictions_and_singular_intrinsics_do_not_poison_valid_batch():
    geometry, camera, rotation, translation, mask = example(batch=3)
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation)
    xy[1, 0, 0] = torch.nan
    camera[2] = 0
    xy.requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.failure_codes.tolist() == [0, PoseFailure.NONFINITE_PREDICTION, PoseFailure.INVALID_TARGET]
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(xy.grad).all()
    assert xy.grad[0].abs().max() > 0
    assert (xy.grad[1:] == 0).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_affine_image_augmentation_and_non_square_intrinsics(dtype):
    geometry, camera, rotation, translation, mask = example(dtype=dtype)
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation)
    baseline = pose_objective(xy, geometry, camera, rotation, translation, mask)
    affine = geometry.new_tensor([[-.8, .1, 600.], [0., 1.2, 15.], [0., 0., 1.]])[None]
    augmented = torch.cat((xy, torch.ones_like(xy[..., :1])), -1) @ affine.transpose(-1,-2)
    result = pose_objective(augmented[..., :2], geometry, affine@camera, rotation, translation, mask)
    assert result.valid_samples.all()
    torch.testing.assert_close(result.loss, baseline.loss, atol=2e-7, rtol=1e-4)


def test_geometry_and_dtype_contract_errors_are_explicit():
    geometry, camera, rotation, translation, mask = example()
    xy = pixels(geometry, camera, rotation, translation)
    with pytest.raises(ValueError, match='diameter'):
        pose_objective(xy, geometry, camera, rotation, translation, mask, object_diameter=0)
    with pytest.raises(ValueError, match='float32 or float64'):
        pose_objective(xy.half(), geometry.half(), camera.half(), rotation.half(), translation.half(), mask)


def test_excessive_float32_update_does_not_poison_other_sample_gradients():
    geometry, camera, rotation, translation, mask = example(batch=2, dtype=torch.float32)
    moved_rotation, moved_translation = shifted_pose(rotation, translation)
    xy = pixels(geometry, camera, moved_rotation, moved_translation)
    xy[1] = 1e30
    xy.requires_grad_()
    result = pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.failure_codes.tolist() == [0, PoseFailure.EXCESSIVE_UPDATE]
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(xy.grad).all()
    assert xy.grad[0].abs().max() > 0
    assert (xy.grad[1] == 0).all()


def test_linearized_zero_error_and_exact_quadratic_scaling():
    geometry, camera, rotation, translation, mask = example()
    truth = pixels(geometry, camera, rotation, translation)
    initial = linearized_pose_objective(truth, geometry, camera, rotation, translation, mask)
    assert initial.valid_samples.all() and initial.loss < 1e-24
    direction = torch.randn(truth.shape, dtype=truth.dtype, generator=torch.Generator().manual_seed(8))
    baseline = linearized_pose_objective(truth+direction, geometry, camera, rotation, translation, mask)
    assert baseline.loss > 0
    for factor in [-16., -1., 2., 32.]:
        result = linearized_pose_objective(truth+direction*factor, geometry, camera, rotation, translation, mask)
        torch.testing.assert_close(result.loss, baseline.loss*factor**2, atol=1e-12, rtol=1e-9)
        torch.testing.assert_close(result.linearized_twist, baseline.linearized_twist*factor, atol=1e-12, rtol=1e-9)


def test_linearized_displacement_metric_matches_reported_loss():
    geometry, camera, rotation, translation, mask = example()
    xy = pixels(geometry, camera, *shifted_pose(rotation, translation))
    result = linearized_pose_objective(xy, geometry, camera, rotation, translation, mask)
    centered_rotated = ((geometry-geometry.mean(0))/torch.pdist(geometry).max())[None] @ rotation.transpose(-1,-2)
    twist = result.linearized_twist
    displacement = torch.cross(twist[:, None, :3].expand_as(centered_rotated), centered_rotated, dim=-1) + twist[:, None, 3:]
    torch.testing.assert_close(displacement.square().sum(-1).mean(), result.loss, atol=1e-12, rtol=1e-10)
    assert not hasattr(result, 'fitted_rotation')  # Do not present a tangent correction as an inference pose.


def test_linearized_gradcheck_and_positive_semidefinite_bounded_hessian():
    geometry, camera, rotation, translation, mask = example()
    camera = torch.eye(3, dtype=geometry.dtype)[None]
    truth = pixels(geometry, camera, rotation, translation)
    xy = (truth+.001).requires_grad_()
    objective = lambda values: linearized_pose_objective(values, geometry, camera, rotation, translation, mask).loss
    assert torch.autograd.gradcheck(objective, (xy,), eps=1e-6, atol=1e-7, rtol=1e-5)
    hessian = torch.autograd.functional.hessian(objective, xy).reshape(xy.numel(), xy.numel())
    torch.testing.assert_close(hessian, hessian.T, atol=1e-12, rtol=1e-10)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    assert eigenvalues.min() > -1e-10
    assert eigenvalues.max() <= 1/(2*.01*mask.sum()) + 1e-10


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_linearized_variable_masks_units_and_affine_calibration(dtype):
    geometry, camera, rotation, translation, mask = example(batch=3, dtype=dtype)
    mask[1, 4:] = False
    mask[2, :2] = False
    xy = pixels(geometry, camera, *shifted_pose(rotation, translation)).requires_grad_()
    result = linearized_pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.valid_samples.all()
    result.loss.backward()
    assert torch.isfinite(xy.grad).all() and (xy.grad[~mask] == 0).all()
    assert xy.grad[mask].abs().max() > 0
    scaled = linearized_pose_objective(xy, geometry*1000, camera, rotation, translation*1000, mask)
    torch.testing.assert_close(scaled.loss, result.loss, rtol=1e-4, atol=1e-7)
    affine = geometry.new_tensor([[-.8,.1,600.],[0.,1.2,15.],[0.,0.,1.]])[None]
    transformed = torch.cat((xy, torch.ones_like(xy[...,:1])), -1) @ affine.transpose(-1,-2)
    augmented = linearized_pose_objective(transformed[...,:2], geometry, affine@camera, rotation, translation, mask)
    torch.testing.assert_close(augmented.loss, result.loss, rtol=1e-4, atol=1e-7)
    for index in range(3):
        single = linearized_pose_objective(xy[index:index+1], geometry, camera[index:index+1], rotation[index:index+1], translation[index:index+1], mask[index:index+1])
        torch.testing.assert_close(single.per_sample_loss[0], result.per_sample_loss[index], rtol=1e-4, atol=1e-7)


def test_linearized_target_failures_and_extreme_prediction_isolation():
    geometry, camera, rotation, translation, mask = example(batch=4, dtype=torch.float32)
    xy = pixels(geometry, camera, *shifted_pose(rotation, translation))
    xy[1, 0, 0] = torch.nan
    camera[2] = 0
    xy[3] = 1e30
    xy.requires_grad_()
    result = linearized_pose_objective(xy, geometry, camera, rotation, translation, mask)
    assert result.failure_codes.tolist() == [0, PoseFailure.NONFINITE_PREDICTION, PoseFailure.INVALID_TARGET, PoseFailure.EXCESSIVE_UPDATE]
    result.loss.backward()
    assert torch.isfinite(xy.grad).all() and (xy.grad[1:] == 0).all()
    assert torch.isnan(result.per_sample_loss[1:]).all()
    mask[:] = False
    rejected = linearized_pose_objective(torch.zeros_like(xy, requires_grad=True), geometry, camera, rotation, translation, mask)
    assert not rejected.valid_samples.any() and rejected.loss == 0
    assert torch.isnan(rejected.linearized_twist).all()
    line = geometry.clone(); line[:, 1:] = 0; line[:,0] = torch.arange(len(line), dtype=line.dtype)
    line_result = linearized_pose_objective(torch.zeros_like(xy), line, camera, rotation, translation, torch.ones_like(mask))
    assert (line_result.failure_codes[[0,1,3]] == PoseFailure.UNOBSERVABLE_TARGET).all()
