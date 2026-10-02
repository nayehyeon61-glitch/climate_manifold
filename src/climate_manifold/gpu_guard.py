"""Run one command on a lightly loaded GPU within existing CUDA visibility.

Checks are admission checks, not a memory/utilization cap during training.
Local advisory locks coordinate this user's runners; they do not reserve GPUs
against other users. On managed clusters, request a scheduler allocation first.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import UUID

from .workspace import WORK_ROOT


class NoGPUAvailable(RuntimeError):
    """No visible GPU currently passes admission (busy or locked)."""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    used_mib: float
    total_mib: float
    utilization: float


def canonical_uuid(value):
    # Torch's UUID identifies the actual visible CUDA device. Never reinterpret
    # CUDA_VISIBLE_DEVICES numeric ordinals as nvidia-smi indices.
    text = str(value).strip()
    if text.startswith('GPU-'):
        text = text[4:]
    return 'GPU-'+str(UUID(text))


def parse_gpus(text):
    records = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError('Unexpected nvidia-smi GPU record')
        index, uuid, used, total, utilization = (v.strip() for v in row)
        try:
            gpu = GPU(int(index), canonical_uuid(uuid), float(used), float(total), float(utilization))
        except ValueError:
            # Unsupported utilization (e.g. N/A) cannot certify a quiet device.
            continue
        if (all(math.isfinite(v) for v in (gpu.used_mib, gpu.total_mib, gpu.utilization))
                and 0 <= gpu.used_mib <= gpu.total_mib and gpu.total_mib > 0
                and 0 <= gpu.utilization <= 100):
            records.append(gpu)
    return records


def query_gpus():
    result = subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    return parse_gpus(result.stdout)


def occupied_gpu_uuids():
    """UUIDs of GPUs running any compute process (anyone's). Such GPUs are never admitted."""
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=20)
    return {canonical_uuid(line) for line in result.stdout.splitlines() if line.strip()}


def visible_gpu_uuids():
    if (any(os.environ.get(key) for key in ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'))
            and 'CUDA_VISIBLE_DEVICES' not in os.environ):
        raise RuntimeError('Scheduler job has no CUDA_VISIBLE_DEVICES allocation; refuse global GPU selection')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA device; check the allocation and CUDA PyTorch build')
    try:
        return {canonical_uuid(torch.cuda.get_device_properties(i).uuid)
                for i in range(torch.cuda.device_count())}
    except (AttributeError, ValueError) as exc:
        raise RuntimeError('Cannot resolve visible CUDA UUIDs; require UUID-capable PyTorch and non-MIG GPUs') from exc


def candidates(records, visible, *, max_utilization=10, max_memory_percent=10, min_free_gib=8,
               occupied=frozenset()):
    thresholds = (max_utilization, max_memory_percent, min_free_gib)
    if (not all(math.isfinite(v) for v in thresholds) or not 0 <= max_utilization <= 100
            or not 0 <= max_memory_percent <= 100 or min_free_gib < 0):
        raise ValueError('Invalid GPU admission thresholds')
    return sorted((g for g in records if g.uuid in visible and g.uuid not in occupied
                   and g.utilization <= max_utilization
                   and 100*g.used_mib/g.total_mib <= max_memory_percent
                   and (g.total_mib-g.used_mib)/1024 >= min_free_gib),
                  key=lambda g: (g.utilization, g.used_mib/g.total_mib, -g.total_mib, g.index))


def run_guarded(command, lock_root, *, max_utilization=10, max_memory_percent=10,
                min_free_gib=8, work_root=WORK_ROOT):
    if not command:
        raise ValueError('A command is required')
    allowed = Path(work_root).resolve(strict=True)
    lock_root = Path(lock_root).resolve()
    if lock_root == allowed or not lock_root.is_relative_to(allowed):
        raise ValueError('GPU locks must stay below the allowed work root')
    visible = visible_gpu_uuids()
    thresholds = dict(max_utilization=max_utilization, max_memory_percent=max_memory_percent,
                      min_free_gib=min_free_gib)
    options = candidates(query_gpus(), visible, occupied=occupied_gpu_uuids(), **thresholds)
    lock_root.mkdir(parents=True, exist_ok=True)
    for gpu in options:
        path = lock_root/(gpu.uuid+'.lock')
        if path.is_symlink():
            raise ValueError('GPU lock files cannot be symlinks')
        with path.open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            # Recheck after locking, immediately before launch.
            latest = candidates(query_gpus(), {gpu.uuid}, occupied=occupied_gpu_uuids(), **thresholds)
            if not latest:
                continue
            selected = latest[0]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': selected.uuid,
                   'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'DEVICE': 'cuda:0'}
            record = {'time': datetime.now(timezone.utc).isoformat(), 'gpu': asdict(selected),
                      'thresholds': thresholds, 'inherited_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'command': command}
            print('GPU admission: '+json.dumps(record), flush=True)
            # Pass the lock to the child so it remains held if this wrapper is
            # terminated while training still runs. Never terminate other jobs.
            return subprocess.call(command, env=env, pass_fds=(lock.fileno(),))
    raise NoGPUAvailable('No allocated/visible GPU satisfies the low-load thresholds (or all are locked). '
                       'No training started; rerun in an available GPU allocation.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-root', required=True)
    parser.add_argument('--max-utilization', type=float, default=10)
    parser.add_argument('--max-memory-percent', type=float, default=10)
    parser.add_argument('--min-free-gib', type=float, default=8)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    # GPU_WAIT_SECONDS>0: poll until a GPU is idle instead of refusing. Never preempts.
    wait = float(os.environ.get('GPU_WAIT_SECONDS', '300'))
    try:
        while True:
            try:
                return run_guarded(command, args.lock_root, max_utilization=args.max_utilization,
                                   max_memory_percent=args.max_memory_percent, min_free_gib=args.min_free_gib)
            except NoGPUAvailable:
                if wait <= 0:
                    raise
                print(f'{datetime.now().isoformat(timespec="seconds")} GPU admission: all GPUs busy; '
                      f'retrying in {wait:.0f}s', file=sys.stderr, flush=True)
                time.sleep(wait)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'GPU admission refused: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
