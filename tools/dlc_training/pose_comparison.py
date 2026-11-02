"""Sequential, resumable epoch orchestration for an explicitly frozen DLC experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def validate_config(config):
    for name in ('project', 'pose_config', 'checkpoint', 'geometry'):
        if not isinstance(config.get(name), str):
            raise ValueError(f'Missing path: {name}')
    if config.get('device') not in ('cpu', 'cuda:0'):
        raise ValueError('Explicit cpu or cuda:0 device required')
    for name in ('epochs', 'batch_size', 'seed_base'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'Positive integer required: {name}')
    if config['epochs'] > 30:
        raise ValueError('Maximum approved budget is 30 epochs per arm')
    expected = {'baseline': 'nonlinear', 'linearized-native': 'linearized', 'nonlinear-native': 'nonlinear'}
    arms = config.get('arms', [])
    if len(arms) != 3 or {arm.get('name') for arm in arms} != set(expected):
        raise ValueError('Require baseline, linearized-native and nonlinear-native exactly once')
    if arms[0]['name'] != 'baseline':
        raise ValueError('Baseline must run first in every epoch')
    for arm in arms:
        if arm.get('objective') != expected[arm['name']]:
            raise ValueError('Arm objective mismatch')
        for name in ('weight', 'damping', 'temperature'):
            value = arm.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name != 'weight' and value == 0):
                raise ValueError(f'Explicit finite arm parameter required: {name}')
        if (arm['name'] == 'baseline') != (arm['weight'] == 0):
            raise ValueError('Baseline weight must be zero; pose arms must have positive weights')
    safeguards = config.get('safeguards', {})
    for name in ('max_epoch_original_loss_ratio', 'max_epoch_mean_pose_loss', 'max_pose_failure_fraction'):
        value = safeguards.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Explicit positive safeguard required: {name}')
    if safeguards['max_pose_failure_fraction'] > 1:
        raise ValueError('Pose failure fraction must not exceed one')


def source_paths():
    here = Path(__file__).resolve().parent
    return [here / name for name in ('pose_comparison.py', 'pose_experiment.py', 'pose_objective.py',
                                     'pose_evaluation.py', 'synthetic_metrics.py', 'synthetic_dataset.py')]


def prepare(config, output, resume):
    validate_config(config)
    config = dict(config)
    for key in ('project', 'pose_config', 'checkpoint', 'geometry'):
        config[key] = str(Path(config[key]).resolve())
    geometry = Path(config['geometry'])
    membership = read_json(geometry / 'split_membership.json')
    training = membership['splits']['training']['image_indices']
    development = membership['splits']['development']['image_indices']
    if not training or not development or len(set(training)) != len(training) or set(training) & set(development):
        raise ValueError('Invalid training/development membership')
    inputs = [Path(config[key]) for key in ('project', 'pose_config', 'checkpoint')]
    inputs += [geometry / name for name in ('manifest.json', 'labels.npz', 'split_membership.json')]
    hashes = {str(path): digest(path) for path in inputs + source_paths()}
    frozen = {'config': config, 'input_hashes': hashes, 'training_count': len(training),
              'training_indices': training, 'development_count': len(development),
              'augmentation_policy': 'native image normalization only; no geometric or photometric augmentation',
              'steps_per_epoch': math.ceil(len(training) / config['batch_size'])}
    if output.exists():
        if not resume:
            raise ValueError('Output exists; explicit --resume required')
        if read_json(output / 'protocol.json') != frozen:
            raise ValueError('Frozen config, input or source hashes changed; refusing resume')
    else:
        if resume:
            raise ValueError('Cannot resume an absent experiment')
        output.mkdir(parents=True)
        write_json(output / 'protocol.json', frozen)
        snapshots = output / 'source_snapshot'
        snapshots.mkdir()
        for path in source_paths():
            (snapshots / path.name).write_bytes(path.read_bytes())
    return frozen


def command(config, arm, epoch, phase, checkpoint, destination, steps):
    script = Path(__file__).with_name('pose_experiment.py' if phase == 'train' else 'pose_evaluation.py')
    args = [sys.executable, str(script), '--project', config['project'], '--pose-config', config['pose_config'],
            '--checkpoint', str(checkpoint), '--geometry', config['geometry'], '--output', str(destination),
            '--device', config['device'], '--batch-size', str(config['batch_size'])]
    if phase == 'train':
        args += ['--steps', str(steps), '--arm', 'baseline' if arm['name'] == 'baseline' else 'pose',
                 '--seed', str(config['seed_base'] + epoch), '--pose-objective', arm['objective'],
                 '--pose-weight', str(arm['weight']), '--pose-damping', str(arm['damping']),
                 '--temperature', str(arm['temperature']), '--coordinate-decoder', 'native-offset',
                 '--gradient-diagnostics', '0', '--save-every', str(steps), '--final-checkpoint-only']
    return args


def execute(args, log_path):
    with log_path.open('w', encoding='utf8') as log:
        return subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def summarize_training(folder, expected_steps, expected_indices):
    manifest = read_json(folder / 'manifest.json')
    updates = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines() if line.strip()]
    if manifest.get('status') != 'complete' or manifest.get('completed_steps') != expected_steps or len(updates) != expected_steps:
        raise ValueError('Incomplete epoch requires inspection; no automatic restart')
    indices = [index for row in updates for index in row['source_indices']]
    if len(indices) != len(expected_indices) or set(indices) != set(expected_indices):
        raise ValueError('Epoch did not visit each training image exactly once')
    result = {}
    for name in ('original_loss', 'pose_loss', 'weighted_pose_loss', 'total_loss', 'gradient_norm'):
        values = [row[name] for row in updates]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Nonfinite update metric: {name}')
        result['mean_' + name] = statistics.mean(values)
    failures = sum(sum(row.get('failure_counts', {}).values()) for row in updates)
    result.update(updates=len(updates), samples=len(indices), pose_failure_samples=failures,
                  pose_failure_fraction=failures / len(indices),
                  training_seconds=manifest.get('elapsed_seconds'), loss_reduction='arithmetic mean of batch losses')
    return result


def run(config, output, resume=False, executor=execute):
    output = Path(output).resolve()
    frozen = prepare(config, output, resume)
    config = frozen['config']
    state_path = output / 'driver_status.json'
    records = read_json(output / 'epochs.json') if (output / 'epochs.json').exists() else []
    status = {'status': 'running', 'pid': os.getpid(), 'completed_epochs': records}
    write_json(state_path, status)
    try:
        for epoch in range(1, config['epochs'] + 1):
            for arm in config['arms']:
                arm_name = arm['name']
                folder = output / arm_name / f'epoch-{epoch:03d}'
                folder.mkdir(parents=True, exist_ok=True)
                previous = output / arm_name / f'epoch-{epoch-1:03d}' / 'train/snapshot-last.pt'
                checkpoint = Path(config['checkpoint']) if epoch == 1 else previous
                summary = None
                for phase in ('train', 'evaluation'):
                    if (output / 'PAUSE').exists():
                        status.update(status='paused', next_arm=arm_name, next_epoch=epoch, next_phase=phase)
                        write_json(state_path, status)
                        return status
                    destination = folder / phase
                    receipt = folder / (phase + '_receipt.json')
                    attempt = folder / (phase + '_attempt.json')
                    phase_input = checkpoint if phase == 'train' else folder / 'train/snapshot-last.pt'
                    if receipt.exists():
                        recorded = read_json(receipt)
                        if recorded['input_checkpoint_sha256'] != digest(phase_input):
                            raise ValueError('Completed operation input checkpoint changed')
                        for name, expected_hash in recorded['output_hashes'].items():
                            if digest(destination / name) != expected_hash:
                                raise ValueError('Completed operation output changed')
                    else:
                        if destination.exists() or attempt.exists():
                            raise ValueError(f'Partial operation requires inspection: {destination}')
                        for name, expected_hash in frozen['input_hashes'].items():
                            if digest(name) != expected_hash:
                                raise ValueError(f'Frozen input changed: {name}')
                        args = command(config, arm, epoch, phase, phase_input, destination, frozen['steps_per_epoch'])
                        status.update(status='running', arm=arm_name, epoch=epoch, phase=phase, command=args)
                        write_json(state_path, status)
                        write_json(attempt, {'command': args, 'input_checkpoint_sha256': digest(phase_input)})
                        started = time.perf_counter()
                        if executor(args, folder / (phase + '.log')) != 0:
                            raise RuntimeError(f'Child {phase} failed; inspect {folder}')
                        if phase == 'train':
                            summarize_training(destination, frozen['steps_per_epoch'], frozen['training_indices'])
                            artifacts = ('snapshot-last.pt', 'manifest.json', 'updates.jsonl')
                        else:
                            if read_json(destination / 'metrics.json')['count'] != frozen['development_count']:
                                raise ValueError('Evaluation did not cover the entire development set')
                            artifacts = ('metrics.json', 'predictions.npz')
                        write_json(receipt, {'input_checkpoint_sha256': digest(phase_input),
                                            'output_hashes': {name: digest(destination / name) for name in artifacts},
                                            'command': args, 'elapsed_seconds': time.perf_counter() - started})
                    if phase == 'train':
                        summary = summarize_training(destination, frozen['steps_per_epoch'], frozen['training_indices'])
                        same_arm = [row for row in records if row['arm'] == arm_name]
                        base = same_arm[0]['mean_original_loss'] if same_arm else summary['mean_original_loss']
                        limits = config['safeguards']
                        baseline = next((row for row in records if row['arm'] == 'baseline' and row['epoch'] == epoch), None)
                        if arm_name != 'baseline':
                            if baseline is None:
                                raise ValueError('Same-epoch completed baseline is required')
                            baseline_log = output / 'baseline' / f'epoch-{epoch:03d}' / 'train/updates.jsonl'
                            baseline_batches = [json.loads(line)['source_indices'] for line in baseline_log.read_text().splitlines() if line.strip()]
                            arm_batches = [json.loads(line)['source_indices'] for line in (destination / 'updates.jsonl').read_text().splitlines() if line.strip()]
                            if arm_batches != baseline_batches:
                                raise ValueError('Same-epoch training batch order or partition differs from baseline')
                            if summary['mean_original_loss'] > baseline['mean_original_loss'] * limits['max_epoch_original_loss_ratio']:
                                status['stopped_epoch'] = {'arm': arm_name, 'epoch': epoch, 'training': summary,
                                                           'reason': 'Original loss exceeds same-epoch baseline safeguard',
                                                           'evaluation_available': False}
                                write_json(state_path, status)
                                raise RuntimeError(f'Original loss exceeds same-epoch baseline safeguard: {arm_name} epoch {epoch}')
                        if (summary['mean_original_loss'] > base * limits['max_epoch_original_loss_ratio']
                                or (arm_name != 'baseline' and (summary['mean_pose_loss'] > limits['max_epoch_mean_pose_loss']
                                or summary['pose_failure_fraction'] > limits['max_pose_failure_fraction']))):
                            status['stopped_epoch'] = {'arm': arm_name, 'epoch': epoch, 'training': summary,
                                                       'reason': 'Configured original-loss, pose-loss or pose-failure safeguard',
                                                       'evaluation_available': False}
                            write_json(state_path, status)
                            raise RuntimeError(f'Configured divergence safeguard stopped all arms: {arm_name} epoch {epoch}')
                if not any(row['arm'] == arm_name and row['epoch'] == epoch for row in records):
                    metrics = read_json(folder / 'evaluation/metrics.json')
                    records.append({'arm': arm_name, 'epoch': epoch, **summary, 'evaluation': metrics,
                                    'checkpoint': str(folder / 'train/snapshot-last.pt'),
                                    'checkpoint_sha256': digest(folder / 'train/snapshot-last.pt')})
                    write_json(output / 'epochs.json', records)
        status.update(status='complete', completed_epochs=records)
    except BaseException as error:
        status.update(status='stopped_requires_inspection', error=repr(error), completed_epochs=records)
        write_json(state_path, status)
        raise
    write_json(state_path, status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(read_json(args.config), args.output, args.resume)
