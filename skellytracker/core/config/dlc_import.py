"""Bounded DLC project discovery and content-addressed conversion reuse.

No training packages are imported here. Export architecture and inference
preprocessing validation remain in the independent optional exporter.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from statistics import median
from time import monotonic
from uuid import uuid4

from filelock import FileLock
import yaml

from skellytracker.core.config.dlc_conversion import DlcConversionRequest, convert_dlc_model, dlc_conversion_readiness
from skellytracker.core.tracker.tracker_presets import dlc_tracker_metadata
from skellytracker.core.sidecar.errors import SidecarError


class DlcInputRequired(ValueError):
    def __init__(self, message: str, fields: tuple[str, ...]):
        super().__init__(message)
        self.fields = fields


def _yaml(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f'DLC input file does not exist: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def latest_project_checkpoint(project: Path, pose: Path | None = None) -> Path:
    """Choose the greatest saved epoch within one configured training run."""
    data = _yaml(project)
    if pose is not None:
        directories = [pose.parent]
    else:
        iteration = data.get('iteration', 0)
        prefix = f"{data.get('Task', '')}{data.get('date', '')}-trainset"
        fractions = {int(float(value) * 100) for value in data.get('TrainingFraction', [])}
        parent = project.parent / 'dlc-models-pytorch' / f'iteration-{iteration}'
        directories = []
        if parent.is_dir():
            for folder in parent.iterdir():
                match = re.fullmatch(re.escape(prefix) + r'(\d+)shuffle(\d+)', folder.name)
                if match and int(match[1]) in fractions and (folder/'train'/'pytorch_config.yaml').is_file():
                    directories.append(folder/'train')
    if len(directories) != 1:
        raise DlcInputRequired('The project has no single matching training run. Choose a checkpoint from the intended shuffle.', ('checkpoint_path',))
    snapshots = []
    for path in directories[0].iterdir():
        match = re.fullmatch(r'snapshot-(best-)?(\d+)\.(pt|pth)', path.name)
        if match and path.is_file():
            snapshots.append((int(match[2]), not bool(match[1]), path))
    if not snapshots:
        raise DlcInputRequired('No numbered checkpoint exists in the project training run. Choose a checkpoint file.', ('checkpoint_path',))
    snapshots.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return snapshots[-1][2].resolve()


def resolve_dlc_request(request: DlcConversionRequest) -> DlcConversionRequest:
    pose = request.pose_config_path.expanduser().resolve() if request.pose_config_path else None
    project = request.project_config_path.expanduser().resolve() if request.project_config_path else None
    # Accept a project YAML supplied through the older pose-config picker too.
    if pose and 'model' not in _yaml(pose) and 'bodyparts' in _yaml(pose):
        if project and project != pose:
            raise ValueError('Two different project configuration files were supplied')
        project, pose = pose, None
    if request.checkpoint_path is None:
        if project is None:
            raise DlcInputRequired('Choose a DLC project to find its latest checkpoint.', ('project_config_path',))
        checkpoint = latest_project_checkpoint(project, pose)
    else:
        checkpoint = request.checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'DLC checkpoint does not exist: {checkpoint}')
    if pose is None:
        candidate = checkpoint.parent / 'pytorch_config.yaml'
        if not candidate.is_file():
            raise DlcInputRequired('The checkpoint has no companion pytorch_config.yaml. Choose its training configuration.', ('pose_config_path',))
        pose = candidate
    training = _yaml(pose)
    if project is None:
        # Only the declared project location is used; never pick an arbitrary nearby YAML.
        declared = (training.get('metadata') or {}).get('project_path')
        if declared:
            candidate = Path(declared).expanduser() / 'config.yaml'
            if candidate.is_file():
                project = candidate.resolve()
    data = _yaml(project) if project else {}
    if data.get('multianimalproject') or data.get('engine', 'pytorch') != 'pytorch':
        raise ValueError('This importer supports single-subject PyTorch DLC models, not multi-animal or TensorFlow projects')
    if data and data.get('bodyparts') != (training.get('metadata') or {}).get('bodyparts'):
        raise ValueError('Project keypoint names/order do not match the checkpoint training configuration')
    height, width = request.image_height, request.image_width
    if height is None or width is None:
        sizes = set()
        videos = data.get('video_sets') or {}
        for video in videos.values():
            crop = video.get('crop') if isinstance(video, dict) else None
            if crop is None:
                raise DlcInputRequired('Project video dimensions are not declared. What input width and height should this model use?', ('image_width', 'image_height'))
            coordinates = [int(value.strip()) for value in crop.split(',')] if isinstance(crop, str) else list(crop)
            if len(coordinates) != 4:
                raise ValueError('Project video crop must declare x1,x2,y1,y2')
            x1,x2,y1,y2 = coordinates
            sizes.add((y2-y1,x2-x1))
        if len(sizes) != 1:
            raise DlcInputRequired('The project does not declare one unambiguous video size. Choose the model input width and height.', ('image_width', 'image_height'))
        declared_height, declared_width = sizes.pop()
        height = height if height is not None else declared_height
        width = width if width is not None else declared_width
    return DlcConversionRequest(pose_config_path=pose, project_config_path=project, checkpoint_path=checkpoint,
                                image_height=height, image_width=width)


def _hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def conversion_identity(request: DlcConversionRequest) -> dict:
    default = Path(__file__).resolve().parents[3] / 'tools/dlc_inference/export_model.py'
    script = Path(os.environ.get('SKELLYTRACKER_DLC_EXPORT_SCRIPT', str(default)))
    return dict(version=1, checkpoint_sha256=_hash(request.checkpoint_path), config_sha256=_hash(request.pose_config_path),
                project_sha256=_hash(request.project_config_path) if request.project_config_path else None,
                input_size=[request.image_height, request.image_width], resize_method='fit', normalization='imagenet_rgb',
                exporter_sha256=_hash(script), reconstruction_sha256=_hash(script.with_name('model_reconstruction.py')))


def _cached(directory: Path, identity: dict) -> Path | None:
    try:
        record = json.loads((directory/'ready.json').read_text(encoding='utf8'))
        name = record['bundle']
        if record['identity'] != identity or len(name) != 32 or any(c not in '0123456789abcdef' for c in name):
            return None
        sidecar = directory/name/'dlc_resnet50_gn.yaml'
        if _hash(sidecar) != record['sidecar_sha256']:
            return None
        metadata = dlc_tracker_metadata(sidecar_path=sidecar)
        if metadata.model_sha256 != record['onnx_sha256'] or list(metadata.input_size) != identity['input_size']:
            return None
        return sidecar
    except (OSError, ValueError, KeyError, TypeError, SidecarError):
        return None


def prepare_model(request: DlcConversionRequest, cache_directory: Path, *, on_progress=None, python_executable: str | None = None) -> Path:
    def progress(stage: str, detail: str, **extra):
        if on_progress:
            on_progress(dict(stage=stage, detail=detail, estimated_remaining_seconds=None, **extra))
    progress('checking', 'Checking project, checkpoint and existing conversions')
    try:
        resolved = resolve_dlc_request(request)
    except DlcInputRequired as error:
        progress('needs_input', str(error), required_fields=list(error.fields))
        raise
    progress('checking', f'Using checkpoint {resolved.checkpoint_path.name}', resolved_checkpoint_path=str(resolved.checkpoint_path))
    identity = conversion_identity(resolved)
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory = cache_directory/key
    directory.mkdir(parents=True,exist_ok=True)
    progress('checking', 'Waiting for any conversion of this same checkpoint to finish')
    with FileLock(directory/'conversion.lock',timeout=650):
        cached = _cached(directory,identity)
        if cached:
            if on_progress:
                on_progress(dict(stage='ready',detail='Using the existing validated conversion',cache_hit=True,estimated_remaining_seconds=0))
            return cached.resolve()
        readiness = dlc_conversion_readiness(python_executable=python_executable)
        if not readiness['available']:
            raise ValueError(readiness['detail'])
        durations=[]
        for previous in list(cache_directory.glob('*/ready.json'))[-30:]:
            try:
                record=json.loads(previous.read_text(encoding='utf8'))
                if all(record['identity'][field] == identity[field] for field in ('input_size','exporter_sha256','reconstruction_sha256')):
                    durations.append(float(record['conversion_seconds']))
            except (OSError,ValueError,KeyError,TypeError):
                continue
        estimate=median(durations) if durations else None
        if on_progress:
            on_progress(dict(stage='converting',detail=f'Converting on CPU at {resolved.image_width} x {resolved.image_height}',
                             estimated_remaining_seconds=estimate,cache_hit=False))
        started=monotonic()
        bundle=uuid4().hex
        sidecar=convert_dlc_model(resolved,directory/bundle,python_executable=python_executable)
        progress('validating','Validating the converted model and artifact hashes')
        metadata=dlc_tracker_metadata(sidecar_path=sidecar)
        # Refuse publication if a training process changed the checkpoint during export.
        if conversion_identity(resolved) != identity:
            raise ValueError('The checkpoint or configuration changed during conversion. Select a stable checkpoint and retry.')
        record=dict(identity=identity,bundle=bundle,sidecar_sha256=_hash(sidecar),onnx_sha256=metadata.model_sha256,
                    conversion_seconds=monotonic()-started)
        pending=directory/f'ready-{uuid4().hex}.json'
        pending.write_text(json.dumps(record,indent=2),encoding='utf8')
        pending.replace(directory/'ready.json')
        if on_progress:
            on_progress(dict(stage='ready',detail='Conversion validated; ready to load',estimated_remaining_seconds=0,cache_hit=False))
        return sidecar.resolve()
