"""Independent held-out PnP evaluation; never used in the training gradient.

Object-to-camera and SQPnP conventions: [@opencvPoseEvaluation413].
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def fit_predicted_pose(object_points, predicted_xy, confidence, intrinsics, image_size, threshold=.4):
    """Fit using predictions alone; no ground-truth pose or visibility input."""
    width, height = image_size
    selected = np.isfinite(predicted_xy).all(-1) & np.isfinite(confidence) & (confidence >= threshold)
    selected &= (predicted_xy[:, 0] >= 0) & (predicted_xy[:, 0] < width)
    selected &= (predicted_xy[:, 1] >= 0) & (predicted_xy[:, 1] < height)
    count = int(selected.sum())
    if count < 6:
        return None, count, 'insufficient_points'
    reference = np.ascontiguousarray(object_points[selected], dtype=np.float64)
    observed = np.ascontiguousarray(predicted_xy[selected], dtype=np.float64)
    if np.linalg.matrix_rank(reference - reference.mean(0)) < 2:
        return None, count, 'degenerate_geometry'
    try:
        success, rotation_vector, translation = cv2.solvePnP(reference, observed, intrinsics, None,
            flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            return None, count, 'solver_failed'
        rotation_vector, translation = cv2.solvePnPRefineLM(reference, observed, intrinsics, None,
            rotation_vector, translation)
        rotation = cv2.Rodrigues(rotation_vector)[0]
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation[:, 0]
        camera_points = reference @ rotation.T + translation[:, 0]
        if not np.isfinite(transform).all() or np.any(camera_points[:, 2] <= 0):
            return None, count, 'invalid_pose'
        return transform, count, 'ok'
    except cv2.error:
        return None, count, 'solver_error'


def pose_errors(fitted, target, object_points):
    """Return rotation degrees, origin translation and mean object displacement."""
    # Renderer poses are float32; project only the rotation comparison onto SO(3).
    u, _, vh = np.linalg.svd(target[:3, :3])
    target_rotation = u @ vh
    relative = fitted[:3, :3] @ target_rotation.T
    sine = np.linalg.norm([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                           relative[1, 0]-relative[0, 1]]) / 2
    cosine = np.clip((np.trace(relative)-1)/2, -1, 1)
    angle = float(np.rad2deg(np.arctan2(sine, cosine)))
    translation = float(np.linalg.norm(fitted[:3, 3]-target[:3, 3]))
    predicted_points = object_points @ fitted[:3, :3].T + fitted[:3, 3]
    target_points = object_points @ target[:3, :3].T + target[:3, 3]
    displacement = float(np.linalg.norm(predicted_points-target_points, axis=-1).mean())
    return angle, translation, displacement


def summarize(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return {'count': int(finite.size), 'mean': float(finite.mean()) if finite.size else None,
            'median': float(np.median(finite)) if finite.size else None,
            'p95': float(np.percentile(finite, 95)) if finite.size else None,
            'maximum': float(finite.max()) if finite.size else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'pose-config', 'checkpoint', 'geometry', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('DLClight', 'True')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    from torch.utils.data import DataLoader, Subset
    from deeplabcut.core.config.utils import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    from deeplabcut.pose_estimation_pytorch.task import Task
    from synthetic_metrics import evaluate_predictions
    from synthetic_dataset import file_hash

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_config_as_dict(args.pose_config)
    with np.load(args.geometry/'labels.npz', allow_pickle=False) as archive:
        geometry = {key: archive[key] for key in archive.files}
    manifest = json.loads((args.geometry/'manifest.json').read_text())
    membership = json.loads((args.geometry/'split_membership.json').read_text())
    development_indices = set(membership['splits']['development']['image_indices'])
    assert file_hash(args.geometry/'labels.npz') == manifest['label_sha256']
    assert config['metadata']['bodyparts'] == geometry['feature_names'].tolist()
    dataset = DLCLoader(args.project, trainset_index=0, shuffle=1).create_dataset(
        transform=build_transforms({'normalize_images': True}), mode='test', task=Task.BOTTOM_UP)
    if args.limit:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = PoseModel.build(deepcopy(config['model']), pretrained_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    model.to(args.device).eval()
    predictions = np.empty((len(dataset), len(geometry['feature_names']), 3), np.float32)
    source_indices = np.empty(len(dataset), np.int64)
    cursor = 0
    started = perf_counter()
    with torch.inference_mode():
        for batch in batches:
            indices = np.array([int(Path(path).stem.removeprefix('source_')) for path in batch['path']])
            assert all(int(index) in development_indices for index in indices)
            for path, index in zip(batch['path'], indices, strict=True):
                assert file_hash(Path(path)) == manifest['image_sha256'][str(geometry['image_names'][index])]
            output = model.get_predictions(model(batch['image'].to(args.device)))['bodypart']['poses']
            values = output[:, 0].cpu().numpy()
            predictions[cursor:cursor+len(indices)] = values
            source_indices[cursor:cursor+len(indices)] = indices
            cursor += len(indices)
    assert cursor == len(dataset)
    edge_xy, _, _, point_metrics = evaluate_predictions(predictions,
        geometry['projected_xy_pixels'][source_indices], geometry['projection_valid'][source_indices],
        geometry['visibility'][source_indices], geometry['feature_names'])
    fitted = np.full((len(dataset), 4, 4), np.nan)
    errors = np.full((len(dataset), 3), np.nan)
    counts = np.zeros(len(dataset), np.int32)
    reasons = []
    for index, source_index in enumerate(source_indices):
        pose, counts[index], reason = fit_predicted_pose(geometry['feature_positions_mm'], edge_xy[index],
            predictions[index, :, 2], geometry['camera_intrinsics'][source_index], manifest['image_size'])
        reasons.append(reason)
        if pose is not None:
            fitted[index] = pose
            errors[index] = pose_errors(pose, np.linalg.inv(geometry['camera_to_object'][source_index]),
                                       geometry['feature_positions_mm'])
    np.savez_compressed(args.output/'predictions.npz', predictions=predictions, source_indices=source_indices,
        fitted_object_to_camera=fitted, pose_errors=errors, selected_point_count=counts, failure_reason=reasons)
    report = {'checkpoint_sha256': file_hash(args.checkpoint), 'count': len(dataset),
        'pose_success_count': reasons.count('ok'), 'pose_coverage': reasons.count('ok')/len(dataset),
        'failure_counts': {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        'rotation_degrees': summarize(errors[:, 0]), 'translation_mm': summarize(errors[:, 1]),
        'object_displacement_mm': summarize(errors[:, 2]), 'keypoint_metrics': point_metrics,
        'pose_policy': 'SQPnP plus LM, predicted confidence >=0.4, >=6 in-image points; no GT initialization or visibility mask',
        'elapsed_seconds': perf_counter()-started, 'device': args.device, 'native_geometry': True}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != 'keypoint_metrics'}, indent=2))


if __name__ == '__main__':
    main()
