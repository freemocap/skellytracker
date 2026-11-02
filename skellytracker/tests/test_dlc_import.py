from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from skellytracker.core.config.dlc_conversion import DlcConversionRequest, prepare_dlc_model, converter_python
from skellytracker.core.config import dlc_import as imports


@pytest.fixture
def project(tmp_path):
    root=tmp_path/'project'; train=root/'train'; train.mkdir(parents=True)
    config=root/'config.yaml'
    config.write_text(yaml.safe_dump(dict(engine='pytorch', bodyparts=['point_a','point_b'],video_sets={'video.mp4':{'crop':'0, 640, 0, 480'}})))
    (train/'pytorch_config.yaml').write_text(yaml.safe_dump(dict(model={}, metadata={'bodyparts':['point_a','point_b'],'project_path':str(root)})))
    checkpoint=train/'snapshot.pt'; checkpoint.write_bytes(b'checkpoint')
    return DlcConversionRequest(project_config_path=config,checkpoint_path=checkpoint)


def test_project_and_checkpoint_resolve_companion_and_declared_size(project):
    result=imports.resolve_dlc_request(project)
    assert result.pose_config_path == project.checkpoint_path.with_name('pytorch_config.yaml')
    assert (result.image_width,result.image_height)==(640,480)
    legacy=project.model_copy(update={'pose_config_path':project.project_config_path,'project_config_path':None})
    assert imports.resolve_dlc_request(legacy)==result


def test_missing_companion_requests_only_training_config(project):
    project.checkpoint_path.with_name('pytorch_config.yaml').unlink()
    with pytest.raises(imports.DlcInputRequired) as error:
        imports.resolve_dlc_request(project)
    assert error.value.fields==('pose_config_path',)


def test_ambiguous_size_requests_dimensions_and_accepts_answer(project):
    data=yaml.safe_load(project.project_config_path.read_text())
    data['video_sets']['other.mp4']={'crop':'0, 800, 0, 600'}
    project.project_config_path.write_text(yaml.safe_dump(data))
    events=[]
    with pytest.raises(imports.DlcInputRequired):
        prepare_dlc_model(project,project.checkpoint_path.parent/'cache',on_progress=events.append)
    assert events[-1]['stage']=='needs_input'
    assert events[-1]['required_fields']==['image_width','image_height']
    result=imports.resolve_dlc_request(project.model_copy(update={'image_width':800,'image_height':600}))
    assert result.image_width==800


@pytest.fixture
def fake_converter(monkeypatch):
    calls=[]
    def convert(request, output, **kwargs):
        calls.append(output); output.mkdir()
        (output/'pose_model.onnx').write_bytes(b'onnx')
        path=output/'dlc_resnet50_gn.yaml'; path.write_text('fixture: true')
        return path
    def metadata(*,sidecar_path):
        artifact=sidecar_path.with_name('pose_model.onnx')
        if artifact.read_bytes()!=b'onnx': raise ValueError('Artifact hash mismatch')
        return SimpleNamespace(model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),input_size=(480,640))
    monkeypatch.setattr(imports,'convert_dlc_model',convert)
    monkeypatch.setattr(imports,'dlc_tracker_metadata',metadata)
    monkeypatch.setattr(imports,'dlc_conversion_readiness',lambda **kwargs:{'available':True})
    return calls


def test_first_conversion_then_cache_without_converter_environment(project,tmp_path,fake_converter,monkeypatch):
    first=prepare_dlc_model(project,tmp_path/'cache')
    monkeypatch.setattr(imports,'dlc_conversion_readiness',lambda **kwargs:pytest.fail('cache requires no converter'))
    events=[]
    assert prepare_dlc_model(project,tmp_path/'cache',on_progress=events.append)==first
    assert len(fake_converter)==1 and events[-1]['cache_hit'] is True


def test_changed_checkpoint_uses_distinct_bundle_and_corruption_rebuilds(project,tmp_path,fake_converter):
    first=prepare_dlc_model(project,tmp_path/'cache')
    project.checkpoint_path.write_bytes(b'new checkpoint')
    second=prepare_dlc_model(project,tmp_path/'cache')
    assert second!=first and first.is_file()
    second.with_name('pose_model.onnx').write_bytes(b'corrupt')
    third=prepare_dlc_model(project,tmp_path/'cache')
    assert third!=second and second.is_file() and len(fake_converter)==3


def test_concurrent_same_import_converts_once(project,tmp_path,fake_converter):
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths=list(pool.map(lambda _:prepare_dlc_model(project,tmp_path/'cache'),range(2)))
    assert paths[0]==paths[1] and len(fake_converter)==1


def test_persistent_converter_setting_and_stale_path(tmp_path,monkeypatch):
    monkeypatch.delenv('SKELLYTRACKER_DLC_CONVERTER_PYTHON',raising=False)
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    config=tmp_path/'.config/skellytracker/dlc-converter.json'; config.parent.mkdir(parents=True)
    executable=tmp_path/'python.exe'; executable.write_bytes(b'fixture')
    config.write_text(json.dumps({'python_executable':str(executable)}))
    assert converter_python()==str(executable)
    executable.unlink()
    with pytest.raises(ValueError,match='no longer exists'):
        converter_python()


def test_latest_epoch_not_best_score_or_modification_time(project):
    data=yaml.safe_load(project.project_config_path.read_text())
    data.update(Task='object',date='Nov02',iteration=2,TrainingFraction=[0.91])
    project.project_config_path.write_text(yaml.safe_dump(data))
    train=project.project_config_path.parent/'dlc-models-pytorch/iteration-2/objectNov02-trainset91shuffle1/train'
    train.mkdir(parents=True)
    (train/'pytorch_config.yaml').write_bytes(project.checkpoint_path.with_name('pytorch_config.yaml').read_bytes())
    for name in ['snapshot-010.pt','snapshot-029.pt','snapshot-best-020.pt','snapshot-best-029.pt']:
        (train/name).write_bytes(b'checkpoint')
    request=project.model_copy(update={'checkpoint_path':None})
    assert imports.resolve_dlc_request(request).checkpoint_path==train/'snapshot-029.pt'
    explicit=request.model_copy(update={'checkpoint_path':train/'snapshot-best-020.pt'})
    assert imports.resolve_dlc_request(explicit).checkpoint_path==train/'snapshot-best-020.pt'
    # An unrelated iteration is not considered, even with a later saved epoch.
    unrelated=train.parent.parent.parent/'iteration-3/objectNov02-trainset91shuffle1/train'
    unrelated.mkdir(parents=True);(unrelated/'snapshot-999.pt').write_bytes(b'other')
    assert imports.resolve_dlc_request(request).checkpoint_path==train/'snapshot-029.pt'
    (train/'snapshot-best-030.pt').write_bytes(b'new best')
    assert imports.resolve_dlc_request(request).checkpoint_path==train/'snapshot-best-030.pt'
    other=train.parent.parent/'objectNov02-trainset91shuffle2/train'
    other.mkdir(parents=True);(other/'pytorch_config.yaml').write_bytes((train/'pytorch_config.yaml').read_bytes())
    with pytest.raises(imports.DlcInputRequired) as error:
        imports.resolve_dlc_request(request)
    assert error.value.fields==('checkpoint_path',)
    assert imports.resolve_dlc_request(explicit).checkpoint_path==train/'snapshot-best-020.pt'
