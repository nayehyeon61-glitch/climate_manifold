"""Runtime write boundaries, GPU allocation membership and admission failures."""
import fcntl
import os

import pytest
import torch

from climate_manifold import gpu_guard as guard
from climate_manifold.workspace import validate_workspace

GPU0='GPU-11111111-1111-1111-1111-111111111111'
GPU1='GPU-22222222-2222-2222-2222-222222222222'


def test_workspace_requires_repo_environment_and_outputs_inside_root(tmp_path):
    allowed=tmp_path/'allowed';allowed.mkdir()
    raw=tmp_path/'raw';raw.mkdir()
    work=allowed/'experiment';repo=work/'code';prefix=work/'venv'
    assert validate_workspace(work,repo,prefix,raw,work_root=allowed)==(allowed,work)
    for paths in [(tmp_path/'outside',repo,prefix),(work,tmp_path/'repo',prefix),
                  (work,repo,tmp_path/'venv'),(allowed,repo,prefix)]:
        with pytest.raises(ValueError,match='subdirectory'):
            validate_workspace(*paths,raw,work_root=allowed)
    (work/'cache').mkdir(parents=True)
    (work/'cache'/'escaped').symlink_to(raw,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        validate_workspace(work,repo,prefix,raw,work_root=allowed)


def test_workspace_rejects_top_level_symlink_before_writing(tmp_path):
    allowed=tmp_path/'allowed';allowed.mkdir()
    raw=tmp_path/'raw';raw.mkdir()
    (allowed/'work').symlink_to(raw,target_is_directory=True)
    with pytest.raises(ValueError):
        validate_workspace(allowed/'work',allowed/'repo',allowed/'venv',raw,work_root=allowed)
    assert not list(raw.iterdir())


def test_gpu_metrics_fail_closed_and_require_all_thresholds():
    rows=guard.parse_gpus(f'0, {GPU0}, 100, 24000, 5\n1, {GPU1}, 500, 48000, 0\n'
                          f'2, {GPU0}, 0, 24000, N/A\n')
    assert len(rows)==2
    assert [g.uuid for g in guard.candidates(rows,{GPU0,GPU1})]==[GPU1,GPU0]
    assert [g.uuid for g in guard.candidates(rows,{GPU0})]==[GPU0]
    assert not guard.candidates([guard.GPU(0,GPU0,0,24000,11)],{GPU0})
    assert not guard.candidates([guard.GPU(0,GPU0,3000,24000,0)],{GPU0})
    assert not guard.candidates([guard.GPU(0,GPU0,0,4096,0)],{GPU0})
    assert not guard.candidates(rows,set())
    with pytest.raises(ValueError):
        guard.candidates(rows,{GPU0},max_utilization=float('nan'))


def test_visible_devices_use_cuda_uuid_not_global_numeric_indices(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    class Properties:
        uuid=GPU1[4:]
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:Properties())
    assert guard.visible_gpu_uuids()=={GPU1}
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setenv('SLURM_JOB_ID','123')
    with pytest.raises(RuntimeError,match='allocation'):
        guard.visible_gpu_uuids()


def _mock_gpu(monkeypatch, visible=None, metrics=None):
    monkeypatch.setattr(guard,'visible_gpu_uuids',lambda:{GPU1} if visible is None else visible)
    monkeypatch.setattr(guard,'query_gpus',lambda:metrics if metrics is not None else [
        guard.GPU(0,GPU0,0,24000,0),guard.GPU(1,GPU1,100,24000,5)])


def test_guard_launches_only_allocated_uuid_and_preserves_parent_visibility(tmp_path,monkeypatch):
    _mock_gpu(monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    calls=[]
    def call(command,**kwargs):
        calls.append((command,kwargs))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==GPU1
        assert kwargs['env']['DEVICE']=='cuda:0'
        assert kwargs['pass_fds']
        return 17
    monkeypatch.setattr(guard.subprocess,'call',call)
    result=guard.run_guarded(['python','train.py'],tmp_path/'locks',work_root=tmp_path)
    assert result==17 and len(calls)==1
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'


@pytest.mark.parametrize('reason',['busy','no_visible','locked','became_busy'])
def test_guard_refuses_without_launching_or_using_other_gpus(tmp_path,monkeypatch,reason):
    _mock_gpu(monkeypatch,visible=set() if reason=='no_visible' else None,
              metrics=[guard.GPU(1,GPU1,100,24000,80)] if reason=='busy' else None)
    monkeypatch.setattr(guard.subprocess,'call',lambda *a,**k:pytest.fail('Must not launch'))
    if reason=='became_busy':
        records=iter([[guard.GPU(1,GPU1,100,24000,0)],[guard.GPU(1,GPU1,100,24000,80)]])
        monkeypatch.setattr(guard,'query_gpus',lambda:next(records))
    locks=tmp_path/'locks';locks.mkdir()
    with (locks/(GPU1+'.lock')).open('a+') as held:
        if reason=='locked':
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='No allocated/visible GPU'):
            guard.run_guarded(['python','train.py'],locks,work_root=tmp_path)


def test_gpu_lock_write_cannot_escape_allowed_root(tmp_path,monkeypatch):
    _mock_gpu(monkeypatch)
    allowed=tmp_path/'allowed';allowed.mkdir()
    with pytest.raises(ValueError,match='below'):
        guard.run_guarded(['python'],tmp_path/'outside',work_root=allowed)
    lock_root=allowed/'locks';lock_root.mkdir()
    outside=tmp_path/'outside.txt';outside.write_text('unchanged')
    (lock_root/(GPU1+'.lock')).symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):
        guard.run_guarded(['python'],lock_root,work_root=allowed)
    assert outside.read_text()=='unchanged'
