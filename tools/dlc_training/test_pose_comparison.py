"""CPU-only orchestration tests; child commands are deterministic file-writing fakes."""
import json
from pathlib import Path

import pytest

from pose_comparison import run, validate_config


@pytest.fixture
def protocol(tmp_path):
    for name in ('project.yaml', 'pose.yaml', 'checkpoint.pt'):
        (tmp_path / name).write_text(name)
    geometry = tmp_path / 'geometry'
    geometry.mkdir()
    for name in ('manifest.json', 'labels.npz'):
        (geometry / name).write_text('{}')
    (geometry / 'split_membership.json').write_text(json.dumps({'splits': {
        'training': {'image_indices': [0, 1, 2]}, 'development': {'image_indices': [3, 4]}}}))
    return {'project': str(tmp_path / 'project.yaml'), 'pose_config': str(tmp_path / 'pose.yaml'),
            'checkpoint': str(tmp_path / 'checkpoint.pt'), 'geometry': str(geometry), 'device': 'cpu',
            'batch_size': 2, 'epochs': 2, 'seed_base': 12,
            'arms': [{'name': name, 'objective': objective, 'weight': weight, 'damping': .01, 'temperature': .0025}
                     for name, objective, weight in [('baseline', 'nonlinear', 0),
                     ('linearized-native', 'linearized', .001), ('nonlinear-native', 'nonlinear', .001)]],
            'safeguards': {'max_epoch_original_loss_ratio': 5, 'max_epoch_mean_pose_loss': 1,
                           'max_pose_failure_fraction': .2}}


def fake_child(args, log):
    options = dict(zip(args[2::2], args[3::2]))
    output = Path(options['--output'])
    output.mkdir()
    if '--steps' in options:
        rows = [{'step': index + 1, 'source_indices': indices, 'original_loss': .1,
                 'pose_loss': .01, 'weighted_pose_loss': .001, 'total_loss': .101,
                 'gradient_norm': .2, 'failure_counts': {}} for index, indices in enumerate(([0, 1], [2]))]
        (output / 'updates.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        (output / 'manifest.json').write_text(json.dumps({'status': 'complete', 'completed_steps': 2,
                                                         'elapsed_seconds': .1}))
        (output / 'snapshot-last.pt').write_text('checkpoint:' + str(output))
    else:
        (output / 'metrics.json').write_text(json.dumps({'count': 2, 'pose_coverage': 1}))
        (output / 'predictions.npz').write_text('fake test predictions')
    return 0


def test_interleaves_epochs_arms_and_resumes_completed(protocol, tmp_path):
    commands = []
    def execute(args, log):
        commands.append(args)
        return fake_child(args, log)
    output = tmp_path / 'result'
    assert run(protocol, output, executor=execute)['status'] == 'complete'
    assert len(commands) == 12
    train = commands[::2]
    assert [args[args.index('--seed') + 1] for args in train] == ['13'] * 3 + ['14'] * 3
    assert all(args[args.index('--steps') + 1] == '2' for args in train)
    assert 'epoch-001' in train[3][train[3].index('--checkpoint') + 1]
    run(protocol, output, resume=True, executor=lambda *_: pytest.fail('Completed phase was rerun'))


def test_pause_between_phases_preserves_checkpoint(protocol, tmp_path):
    output = tmp_path / 'result'
    def execute(args, log):
        result = fake_child(args, log)
        (output / 'PAUSE').touch()
        return result
    status = run(protocol, output, executor=execute)
    assert status['status'] == 'paused'
    assert status['next_phase'] == 'evaluation'
    assert (output / 'baseline/epoch-001/train/snapshot-last.pt').exists()
    (output / 'PAUSE').unlink()
    assert run(protocol, output, resume=True, executor=fake_child)['status'] == 'complete'


def test_partial_operation_requires_inspection(protocol, tmp_path):
    output = tmp_path / 'result'
    def failed(args, log):
        Path(args[args.index('--output') + 1]).mkdir()
        return 1
    with pytest.raises(RuntimeError):
        run(protocol, output, executor=failed)
    with pytest.raises(ValueError, match='Partial operation'):
        run(protocol, output, resume=True, executor=lambda *_: pytest.fail('Partial operation relaunched'))


def test_changed_inputs_reject_resume(protocol, tmp_path):
    output = tmp_path / 'result'
    run(protocol, output, executor=fake_child)
    Path(protocol['checkpoint']).write_text('changed')
    with pytest.raises(ValueError, match='hashes changed'):
        run(protocol, output, resume=True, executor=fake_child)


def test_explicit_budget_and_weights(protocol):
    protocol['epochs'] = 31
    with pytest.raises(ValueError, match='30'):
        validate_config(protocol)
    protocol['epochs'] = 1
    protocol['arms'][0]['weight'] = .1
    with pytest.raises(ValueError, match='Baseline weight'):
        validate_config(protocol)


def test_first_epoch_divergence_stops_against_baseline(protocol, tmp_path):
    def divergent(args, log):
        result = fake_child(args, log)
        output = Path(args[args.index('--output') + 1])
        if '--arm' in args and args[args.index('--arm') + 1] == 'pose':
            updates = output / 'updates.jsonl'
            rows = [json.loads(line) for line in updates.read_text().splitlines()]
            for row in rows:
                row['original_loss'] = 10
            updates.write_text('\n'.join(map(json.dumps, rows)))
        return result
    with pytest.raises(RuntimeError, match='same-epoch baseline'):
        run(protocol, tmp_path / 'result', executor=divergent)
    status = json.loads((tmp_path / 'result/driver_status.json').read_text())
    assert status['stopped_epoch']['training']['mean_original_loss'] == 10
    assert status['stopped_epoch']['arm'] == 'linearized-native'
    assert status['stopped_epoch']['evaluation_available'] is False


def test_exact_batch_order_required(protocol, tmp_path):
    def reordered(args, log):
        result = fake_child(args, log)
        output = Path(args[args.index('--output') + 1])
        if '--arm' in args and args[args.index('--arm') + 1] == 'pose':
            updates = output / 'updates.jsonl'
            rows = [json.loads(line) for line in updates.read_text().splitlines()]
            rows[0]['source_indices'] = [1, 0]
            updates.write_text('\n'.join(map(json.dumps, rows)))
        return result
    with pytest.raises(ValueError, match='batch order'):
        run(protocol, tmp_path / 'result', executor=reordered)


def test_baseline_first(protocol):
    protocol['arms'].reverse()
    with pytest.raises(ValueError, match='Baseline must run first'):
        validate_config(protocol)
