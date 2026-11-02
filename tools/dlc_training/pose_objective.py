"""Optional supervised local pose objective for the isolated DLC training tools.

This is NOT an inference pose estimator: each fit starts at the known target pose.
It supervises the pose-consistent part of keypoint error and must accompany, not
replace, the original image-keypoint objective. No tracking runtime imports Torch.

Column vectors follow X_camera = R @ X_object + t, with positive camera Z forward.
Image coordinates and intrinsics must use the same pixel-origin convention.
Geometry and translation share arbitrary length units; diameter normalization
makes the objective invariant to changing those units together.

Differentiable PnP is motivated by [@chen2020Bpnp]; this implementation unrolls
local solves, not that paper's implicit differentiation. Batched linear solves
follow [@torch29LinearSolve].
"""
from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch import Tensor


class PoseFailure(IntEnum):
    NONE = 0
    INVALID_TARGET = 1
    INSUFFICIENT_POINTS = 2
    UNOBSERVABLE_TARGET = 3
    NONFINITE_PREDICTION = 4
    LINEAR_SOLVE_FAILED = 5
    INVALID_FITTED_DEPTH = 6
    NONFINITE_UPDATE = 7
    EXCESSIVE_UPDATE = 8


@dataclass(frozen=True)
class PoseObjectiveResult:
    loss: Tensor
    per_sample_loss: Tensor
    fitted_rotation: Tensor
    fitted_translation: Tensor
    valid_samples: Tensor
    failure_codes: Tensor
    valid_point_count: Tensor
    target_observability_ratio: Tensor

    @property
    def failure_counts(self) -> dict[str, Tensor]:
        """Device-side sample counts; logging can transfer these outside the loss."""
        return {failure.name.lower(): (self.failure_codes == failure).sum()
                for failure in PoseFailure if failure != PoseFailure.NONE}


@dataclass(frozen=True)
class LinearizedPoseObjectiveResult:
    """A supervised tangent correction, deliberately not a fitted rigid pose."""
    loss: Tensor
    per_sample_loss: Tensor
    linearized_twist: Tensor
    normalized_pose_correction: Tensor
    valid_samples: Tensor
    failure_codes: Tensor
    valid_point_count: Tensor
    target_observability_ratio: Tensor
    target_singular_values: Tensor

    @property
    def failure_counts(self) -> dict[str, Tensor]:
        return {failure.name.lower(): (self.failure_codes == failure).sum()
                for failure in PoseFailure if failure != PoseFailure.NONE}


def _skew(values: Tensor) -> Tensor:
    """Dense skew matrices: skew(p) @ omega == p cross omega."""
    x, y, z = values.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*values.shape[:-1], 3, 3)


def _project_and_jacobian(camera_points: Tensor, depth_floor: float) -> tuple[Tensor, Tensor]:
    # Positive-depth validity is checked separately; this floor only keeps rejected
    # samples numerically harmless in the shared batched calculation.
    depth = camera_points[..., 2].clamp_min(depth_floor)
    inverse = depth.reciprocal()
    x, y = camera_points[..., 0], camera_points[..., 1]
    zero = torch.zeros_like(inverse)
    projection = torch.stack((x * inverse, y * inverse), -1)
    perspective = torch.stack((inverse, zero, -x * inverse.square(),
                               zero, inverse, -y * inverse.square()), -1).reshape(*depth.shape, 2, 3)
    identity = torch.eye(3, dtype=camera_points.dtype, device=camera_points.device).expand(*depth.shape, 3, 3)
    # Left-camera twist: dP/d[omega,v] = [-skew(P), I]. The update below
    # uses the corresponding smooth retraction (exp(omega) R, exp(omega) t + v).
    return projection, perspective @ torch.cat((-_skew(camera_points), identity), -1)


def _geometry_contract(predicted_xy, object_xyz, camera_intrinsics, ground_truth_rotation, ground_truth_translation, valid_mask, object_diameter):
    if predicted_xy.ndim != 3 or predicted_xy.shape[-1] != 2:
        raise ValueError("predicted_xy must have shape [batch,keypoints,2]")
    batch, points, _ = predicted_xy.shape
    if not batch or not points:
        raise ValueError("At least one sample and one object point are required")
    expected = ((object_xyz, (points, 3)), (camera_intrinsics, (batch, 3, 3)),
                (ground_truth_rotation, (batch, 3, 3)), (ground_truth_translation, (batch, 3)),
                (valid_mask, (batch, points)))
    if any(tuple(value.shape) != shape for value, shape in expected):
        raise ValueError("Pose objective tensor axes do not match [B,K,2], [K,3], [B,3,3], [B,3], [B,K]")
    if predicted_xy.dtype not in (torch.float32, torch.float64):
        raise ValueError("Pose fitting requires float32 or float64, not mixed-precision half tensors")
    if any(value.device != predicted_xy.device for value, _ in expected):
        raise ValueError("All pose objective tensors must share a device")
    if any(value.dtype != predicted_xy.dtype for value, _ in expected[:-1]) or valid_mask.dtype != torch.bool:
        raise ValueError("Numerical tensors must share a dtype; valid_mask must be boolean")
    if not torch.isfinite(object_xyz).all():
        raise ValueError("Fixed object geometry must be finite")

    diameter = (torch.pdist(object_xyz).max() if points > 1 else object_xyz.new_zeros(())) if object_diameter is None else torch.as_tensor(object_diameter, dtype=object_xyz.dtype, device=object_xyz.device)
    if diameter.ndim != 0 or not torch.isfinite(diameter) or diameter <= 0:
        raise ValueError("Object diameter must be a finite positive scalar")
    return batch, points, diameter


def pose_objective(
    predicted_xy: Tensor,
    object_xyz: Tensor,
    camera_intrinsics: Tensor,
    ground_truth_rotation: Tensor,
    ground_truth_translation: Tensor,
    valid_mask: Tensor,
    *,
    object_diameter: float | Tensor | None = None,
    iterations: int = 8,
    damping: float = 1e-4,
    observability_rtol: float = 1e-6,
    depth_floor: float = 1e-6,
    max_update_magnitude: float = 1000.0,
) -> PoseObjectiveResult:
    """Fit predicted pixels locally from GT and penalize normalized pose displacement.

    Args:
        predicted_xy: [B,K,2], differentiable image coordinates (no argmax here).
        object_xyz: [K,3], fixed object points in the same identity order.
        camera_intrinsics: [B,3,3], invertible pixel projection matrices. An
            invertible image augmentation may already be composed into these.
        ground_truth_rotation: [B,3,3], object-to-camera proper rotations.
        ground_truth_translation: [B,3], in the geometry's length units.
        valid_mask: [B,K] bool, caller-selected target correspondences. Visibility
            is not inferred: projected occluded points may be included deliberately.

    For normalized camera points P, q(P)=(Px/Pz, Py/Pz), residual r=q-y and
    J=Dq(P)[-[P]x,I]. Each step solves
        (J.T J + damping * diag(J.T J)) delta = -J.T r.
    The returned auxiliary loss is mean_k ||(R_fit X_k+t_fit)
    -(R_gt X_k+t_gt)||^2 / diameter^2, averaged over accepted samples.
    It is neither pixel error nor an independently initialized PnP score.

    Invalid samples have NaN diagnostic outputs and explicit failure codes. If
    no samples are accepted, loss is differentiable zero and failure counts must
    be monitored by the caller; this is not a successful zero-error batch.
    The generous max_update_magnitude bounds each dimensionless twist component
    before matrix_exp; it rejects numerically unsafe local fits, not keypoints.
    """
    if (iterations < 1 or damping <= 0 or not 0 < observability_rtol < 1 or depth_floor <= 0
            or max_update_magnitude <= 0
            or not all(math.isfinite(value) for value in (damping, observability_rtol, depth_floor, max_update_magnitude))):
        raise ValueError("Iterations, damping, depth floor and observability threshold must be positive")
    batch, points, diameter = _geometry_contract(predicted_xy, object_xyz, camera_intrinsics, ground_truth_rotation, ground_truth_translation, valid_mask, object_diameter)
    center = object_xyz.mean(0)
    geometry = (object_xyz - center) / diameter
    identity3 = torch.eye(3, dtype=predicted_xy.dtype, device=predicted_xy.device).expand(batch, 3, 3)
    identity6 = torch.eye(6, dtype=predicted_xy.dtype, device=predicted_xy.device).expand(batch, 6, 6)
    finite_rotation = torch.isfinite(ground_truth_rotation).all(dim=(-2, -1))
    rotation_safe = torch.where(finite_rotation[:, None, None], ground_truth_rotation, identity3)
    valid_rotation = finite_rotation & ((rotation_safe.transpose(-1, -2) @ rotation_safe - identity3).abs().amax(dim=(-2, -1)) < 1e-3) & ((torch.linalg.det(rotation_safe) - 1).abs() < 1e-3)
    finite_translation = torch.isfinite(ground_truth_translation).all(-1)
    finite_camera = torch.isfinite(camera_intrinsics).all(dim=(-2, -1))
    camera_safe = torch.where(finite_camera[:, None, None], camera_intrinsics, identity3)
    # Probe invertibility before the differentiable solve, so invalid matrices do
    # not produce NaNs in the backward pass of otherwise valid batch members.
    _, camera_info = torch.linalg.inv_ex(camera_safe)
    target_valid = valid_rotation & finite_translation & finite_camera & (camera_info == 0)
    camera_safe = torch.where(target_valid[:, None, None], camera_safe, identity3)
    target_rotation = torch.where(target_valid[:, None, None], rotation_safe, identity3)
    target_translation = torch.where(target_valid[:, None], ground_truth_translation, torch.zeros_like(ground_truth_translation))
    target_center_translation = ((target_rotation @ center.expand(batch, 3).unsqueeze(-1)).squeeze(-1) + target_translation) / diameter
    target_points = geometry.unsqueeze(0) @ target_rotation.transpose(-1, -2) + target_center_translation[:, None, :]
    target_depth_valid = target_points[..., 2] > depth_floor
    correspondences = valid_mask & target_depth_valid
    counts = correspondences.sum(-1)
    finite_predictions = torch.isfinite(predicted_xy).all(-1)
    prediction_valid = (~valid_mask | finite_predictions).all(-1)
    safe_pixels = torch.where(finite_predictions[..., None], predicted_xy, torch.zeros_like(predicted_xy))
    homogeneous = torch.cat((safe_pixels, torch.ones_like(safe_pixels[..., :1])), -1)
    rays = torch.linalg.solve(camera_safe, homogeneous.transpose(-1, -2)).transpose(-1, -2)
    ray_valid = torch.isfinite(rays).all(-1) & (rays[..., 2].abs() > torch.finfo(rays.dtype).eps)
    prediction_valid = prediction_valid & (~valid_mask | ray_valid).all(-1)
    ray_denominator = torch.where(ray_valid, rays[..., 2], torch.ones_like(rays[..., 2]))
    normalized_xy = torch.where(ray_valid[..., None], rays[..., :2] / ray_denominator[..., None], torch.zeros_like(safe_pixels))
    _, target_jacobian = _project_and_jacobian(target_points, depth_floor)
    weighted_target_jacobian = (target_jacobian * correspondences[..., None, None]).reshape(batch, points * 2, 6)
    if points * 2 < 6:
        weighted_target_jacobian = torch.nn.functional.pad(weighted_target_jacobian, (0, 0, 0, 6 - points * 2))
    singular = torch.linalg.svdvals(weighted_target_jacobian)
    ratio = singular[..., -1] / singular[..., 0].clamp_min(torch.finfo(singular.dtype).eps)
    failure = torch.zeros(batch, dtype=torch.int64, device=predicted_xy.device)
    failure = torch.where(~target_valid, int(PoseFailure.INVALID_TARGET), failure)
    failure = torch.where((failure == 0) & (counts < 4), int(PoseFailure.INSUFFICIENT_POINTS), failure)
    failure = torch.where((failure == 0) & (ratio < observability_rtol), int(PoseFailure.UNOBSERVABLE_TARGET), failure)
    failure = torch.where((failure == 0) & ~prediction_valid, int(PoseFailure.NONFINITE_PREDICTION), failure)
    rotation, translation = target_rotation, target_center_translation
    for _ in range(iterations):
        active = failure == 0
        camera_points = geometry.unsqueeze(0) @ rotation.transpose(-1, -2) + translation[:, None, :]
        projection, jacobian = _project_and_jacobian(camera_points, depth_floor)
        weights = correspondences & active[:, None]
        jacobian = (jacobian * weights[..., None, None]).reshape(batch, points * 2, 6)
        residual = ((projection - normalized_xy) * weights[..., None]).reshape(batch, points * 2, 1)
        hessian = jacobian.transpose(-1, -2) @ jacobian
        regularizer = damping * hessian.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)
        system = hessian + torch.diag_embed(regularizer)
        system = torch.where(active[:, None, None], system, identity6)
        rhs = -(jacobian.transpose(-1, -2) @ residual)
        delta, info = torch.linalg.solve_ex(system, rhs)
        delta = delta.squeeze(-1)
        solve_good = info == 0
        finite_delta = torch.isfinite(delta).all(-1)
        failure = torch.where(active & ~solve_good, int(PoseFailure.LINEAR_SOLVE_FAILED), failure)
        failure = torch.where((failure == 0) & ~finite_delta, int(PoseFailure.NONFINITE_UPDATE), failure)
        failure = torch.where((failure == 0) & (delta.abs().amax(-1) > max_update_magnitude), int(PoseFailure.EXCESSIVE_UPDATE), failure)
        delta = torch.where((failure == 0)[:, None], delta, torch.zeros_like(delta))
        increment = torch.matrix_exp(_skew(delta[..., :3]))
        candidate_rotation = increment @ rotation
        candidate_translation = (increment @ translation.unsqueeze(-1)).squeeze(-1) + delta[..., 3:]
        candidate_points = geometry.unsqueeze(0) @ candidate_rotation.transpose(-1, -2) + candidate_translation[:, None, :]
        update_finite = torch.isfinite(candidate_points).all(dim=(-2, -1))
        depth_valid = (~correspondences | (candidate_points[..., 2] > depth_floor)).all(-1)
        failure = torch.where((failure == 0) & ~update_finite, int(PoseFailure.NONFINITE_UPDATE), failure)
        failure = torch.where((failure == 0) & ~depth_valid, int(PoseFailure.INVALID_FITTED_DEPTH), failure)
        rotation = torch.where((failure == 0)[:, None, None], candidate_rotation, rotation)
        translation = torch.where((failure == 0)[:, None], candidate_translation, translation)
    accepted = failure == 0
    fitted_points = geometry.unsqueeze(0) @ rotation.transpose(-1, -2) + translation[:, None, :]
    sample_loss = (fitted_points - target_points).square().sum(-1).mean(-1)
    loss = torch.where(accepted, sample_loss, torch.zeros_like(sample_loss)).sum() / accepted.sum().clamp_min(1)
    output_translation = diameter * translation - (rotation @ center.expand(batch, 3).unsqueeze(-1)).squeeze(-1)
    return PoseObjectiveResult(
        loss=loss,
        per_sample_loss=torch.where(accepted, sample_loss, torch.full_like(sample_loss, torch.nan)),
        fitted_rotation=torch.where(accepted[:, None, None], rotation, torch.full_like(rotation, torch.nan)),
        fitted_translation=torch.where(accepted[:, None], output_translation, torch.full_like(output_translation, torch.nan)),
        valid_samples=accepted, failure_codes=failure, valid_point_count=counts,
        target_observability_ratio=ratio,
    )


def linearized_pose_objective(
    predicted_xy: Tensor,
    object_xyz: Tensor,
    camera_intrinsics: Tensor,
    ground_truth_rotation: Tensor,
    ground_truth_translation: Tensor,
    valid_mask: Tensor,
    *,
    object_diameter: float | Tensor | None = None,
    damping: float = 0.01,
    observability_rtol: float = 1e-6,
    depth_floor: float = 1e-6,
) -> LinearizedPoseObjectiveResult:
    """Fixed-target-Jacobian quadratic pose-displacement supervision.

    This is a separate objective, not one iteration of ``pose_objective``. There
    is no fitted-pose exponential, changing Jacobian, or prediction-dependent
    depth rejection. The original nonlinear objective remains a diagnostic of
    the unsuccessful pilot; do not relabel its results as this objective.

    Geometry is centered and diameter-normalized. At the fixed target pose,
    G_k=[-[R X_centered_k]x,I] maps a tangent correction into normalized 3D point
    displacement; J_k=Dq(P_gt_k) G_k maps it into normalized image displacement.
    J and image residual r are divided by sqrt(number of valid correspondences).
    M=mean_k(G_k.T G_k) is the full-object displacement metric, including points
    not observed in this image. With M=L L.T, A=J L^(-T), solve
        z=(A.T A + damping I)^(-1) A.T r, delta=L^(-T)z, loss=z.T z.
    Thus damping is an absolute ridge penalty in normalized physical-displacement
    coordinates, NOT the nonlinear objective's diagonal-Hessian damping.

    For fixed affine image calibration this is a positive-semidefinite quadratic
    in predicted pixels. In normalized residual coordinates its Hessian norm is
    at most 1/(2*damping), since sigma^2/(sigma^2+damping)^2 <= 1/(4*damping).
    Weak pose directions are attenuated rather than amplified without bound.
    Its six-dimensional image subspace cannot supervise every keypoint error;
    retain the original DLC objective and evaluate with independent PnP.

    ``linearized_twist`` uses left rotation increments about the object centroid
    (radians) and independent camera-frame centroid translation (object diameters).
    It is a local supervised correction, not an independent pose estimate.
    Invalid samples receive NaN diagnostics and counted failures; an all-invalid
    batch returns differentiable zero only as an explicit masked-loss convention.
    """
    if (damping <= 0 or depth_floor <= 0 or not 0 < observability_rtol < 1
            or not all(math.isfinite(v) for v in (damping, depth_floor, observability_rtol))):
        raise ValueError("Damping, depth floor and observability threshold must be finite and positive")
    batch, points, diameter = _geometry_contract(predicted_xy, object_xyz, camera_intrinsics, ground_truth_rotation, ground_truth_translation, valid_mask, object_diameter)
    center = object_xyz.mean(0)
    geometry = (object_xyz - center) / diameter
    identity3 = torch.eye(3, dtype=predicted_xy.dtype, device=predicted_xy.device).expand(batch, 3, 3)
    identity6 = torch.eye(6, dtype=predicted_xy.dtype, device=predicted_xy.device).expand(batch, 6, 6)
    finite_rotation = torch.isfinite(ground_truth_rotation).all(dim=(-2, -1))
    rotation = torch.where(finite_rotation[:, None, None], ground_truth_rotation, identity3)
    proper_rotation = finite_rotation & ((rotation.transpose(-1, -2) @ rotation - identity3).abs().amax(dim=(-2, -1)) < 1e-3) & ((torch.linalg.det(rotation) - 1).abs() < 1e-3)
    finite_camera = torch.isfinite(camera_intrinsics).all(dim=(-2, -1))
    camera = torch.where(finite_camera[:, None, None], camera_intrinsics, identity3)
    _, camera_info = torch.linalg.inv_ex(camera)
    target_valid = proper_rotation & torch.isfinite(ground_truth_translation).all(-1) & finite_camera & (camera_info == 0)
    rotation = torch.where(target_valid[:, None, None], rotation, identity3)
    camera = torch.where(target_valid[:, None, None], camera, identity3)
    translation = torch.where(target_valid[:, None], ground_truth_translation, torch.zeros_like(ground_truth_translation))
    centered_translation = ((rotation @ center.expand(batch, 3).unsqueeze(-1)).squeeze(-1) + translation) / diameter
    rotated_geometry = geometry.unsqueeze(0) @ rotation.transpose(-1, -2)
    target_points = rotated_geometry + centered_translation[:, None, :]
    target_xy, camera_jacobian = _project_and_jacobian(target_points, depth_floor)
    correspondences = valid_mask & (target_points[..., 2] > depth_floor)
    counts = correspondences.sum(-1)
    normalizer = counts.clamp_min(1).sqrt().to(predicted_xy.dtype)
    # Convert the left-camera twist into independent centroid rotation/translation.
    rotation_jacobian = camera_jacobian[..., :3] + camera_jacobian[..., 3:] @ _skew(centered_translation)[:, None]
    jacobian = torch.cat((rotation_jacobian, camera_jacobian[..., 3:]), -1)
    jacobian = (jacobian * correspondences[..., None, None]).reshape(batch, points * 2, 6) / normalizer[:, None, None]
    displacement_jacobian = torch.cat((-_skew(rotated_geometry), identity3[:, None].expand(batch, points, 3, 3)), -1)
    metric = torch.einsum('bnji,bnjk->bik', displacement_jacobian, displacement_jacobian) / points
    cholesky, metric_info = torch.linalg.cholesky_ex(metric)
    metric_valid = (metric_info == 0) & torch.isfinite(cholesky).all(dim=(-2, -1))
    cholesky = torch.where(metric_valid[:, None, None], cholesky, identity6)
    whitening = torch.linalg.solve_triangular(cholesky.transpose(-1, -2), identity6, upper=True)
    physical_jacobian = jacobian @ whitening
    padded_jacobian = physical_jacobian if points * 2 >= 6 else torch.nn.functional.pad(physical_jacobian, (0, 0, 0, 6 - points * 2))
    singular = torch.linalg.svdvals(padded_jacobian)
    ratio = singular[..., -1] / singular[..., 0].clamp_min(torch.finfo(predicted_xy.dtype).eps)
    failure = torch.zeros(batch, dtype=torch.int64, device=predicted_xy.device)
    failure = torch.where(~target_valid, int(PoseFailure.INVALID_TARGET), failure)
    failure = torch.where((failure == 0) & (counts < 4), int(PoseFailure.INSUFFICIENT_POINTS), failure)
    failure = torch.where((failure == 0) & (~metric_valid | (ratio < observability_rtol)), int(PoseFailure.UNOBSERVABLE_TARGET), failure)
    finite_predictions = torch.isfinite(predicted_xy).all(-1)
    safe_pixels = torch.where(finite_predictions[..., None], predicted_xy, torch.zeros_like(predicted_xy))
    homogeneous = torch.cat((safe_pixels, torch.ones_like(safe_pixels[..., :1])), -1)
    rays = torch.linalg.solve(camera, homogeneous.transpose(-1, -2)).transpose(-1, -2)
    ray_valid = torch.isfinite(rays).all(-1) & (rays[..., 2].abs() > torch.finfo(predicted_xy.dtype).eps)
    prediction_valid = (~valid_mask | (finite_predictions & ray_valid)).all(-1)
    failure = torch.where((failure == 0) & ~prediction_valid, int(PoseFailure.NONFINITE_PREDICTION), failure)
    denominator = torch.where(ray_valid, rays[..., 2], torch.ones_like(rays[..., 2]))
    normalized_xy = torch.where(ray_valid[..., None], rays[..., :2] / denominator[..., None], torch.zeros_like(safe_pixels))
    active = failure == 0
    residual = ((normalized_xy - target_xy) * (correspondences & active[:, None])[..., None]).reshape(batch, points * 2, 1) / normalizer[:, None, None]
    physical_jacobian = physical_jacobian * active[:, None, None]
    system = physical_jacobian.transpose(-1, -2) @ physical_jacobian + damping * identity6
    rhs = physical_jacobian.transpose(-1, -2) @ residual
    correction, solve_info = torch.linalg.solve_ex(system, rhs)
    correction = correction.squeeze(-1)
    failure = torch.where((failure == 0) & (solve_info != 0), int(PoseFailure.LINEAR_SOLVE_FAILED), failure)
    failure = torch.where((failure == 0) & ~torch.isfinite(correction).all(-1), int(PoseFailure.NONFINITE_UPDATE), failure)
    # Check before squaring so an extreme rejected sample cannot poison backward.
    square_safe = correction.abs().amax(-1) < math.sqrt(torch.finfo(predicted_xy.dtype).max / 24)
    failure = torch.where((failure == 0) & ~square_safe, int(PoseFailure.EXCESSIVE_UPDATE), failure)
    accepted = failure == 0
    correction = torch.where(accepted[:, None], correction, torch.zeros_like(correction))
    sample_loss = correction.square().sum(-1)
    loss = sample_loss.sum() / accepted.sum().clamp_min(1)
    twist = (whitening @ correction.unsqueeze(-1)).squeeze(-1)
    return LinearizedPoseObjectiveResult(
        loss=loss,
        per_sample_loss=torch.where(accepted, sample_loss, torch.full_like(sample_loss, torch.nan)),
        linearized_twist=torch.where(accepted[:, None], twist, torch.full_like(twist, torch.nan)),
        normalized_pose_correction=torch.where(accepted[:, None], correction, torch.full_like(correction, torch.nan)),
        valid_samples=accepted, failure_codes=failure, valid_point_count=counts,
        target_observability_ratio=ratio, target_singular_values=singular,
    )
