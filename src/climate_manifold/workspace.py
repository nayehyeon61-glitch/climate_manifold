"""Validate managed write paths before the daily server experiment starts.

This is an application path policy, not an operating-system sandbox. System
executables/libraries and the ERA5 source are read outside the work root.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

WORK_ROOT = Path('/lustre/home/yehyeon')
CACHE_PATHS = {
    'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    'XDG_CACHE_HOME': 'cache', 'PIP_CACHE_DIR': 'cache/pip',
    'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
    'CUDA_CACHE_PATH': 'cache/cuda', 'TRITON_CACHE_DIR': 'cache/triton',
    'PYTHONPYCACHEPREFIX': 'cache/pycache', 'MPLCONFIGDIR': 'cache/matplotlib',
    'NUMBA_CACHE_DIR': 'cache/numba',
}


def validate_workspace(work, repository, prefix, raw_root, *, work_root=WORK_ROOT):
    allowed = Path(work_root).resolve(strict=True)
    raw = Path(raw_root).resolve(strict=True)
    work, repository, prefix = (Path(p).resolve() for p in (work, repository, prefix))
    for label, path in [('DAILY_WORK', work), ('repository', repository), ('virtualenv', prefix)]:
        if path == allowed or not path.is_relative_to(allowed) or path.is_relative_to(raw):
            raise ValueError(f'{label} must be a subdirectory of {allowed}, outside the raw source: {path}')
    # These are all writable directories used by the runner. Symlink escapes in
    # an existing preparation/cache tree are rejected before creating anything.
    directories = {work/p for p in (*CACHE_PATHS.values(), 'logs', 'prepared', 'runs')}
    directories.add(allowed/'climate_manifold_gpu_locks')
    for directory in directories:
        resolved = directory.resolve()
        if not resolved.is_relative_to(allowed) or resolved.is_relative_to(raw):
            raise ValueError(f'Writable path escapes work root or enters raw data: {directory}')
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_symlink() and (not path.resolve().is_relative_to(allowed)
                                         or path.resolve().is_relative_to(raw)):
                    raise ValueError(f'Writable symlink escapes work root or enters raw data: {path}')
    return allowed, work


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Use the virtual environment installed below /lustre/home/yehyeon')
    allowed, work = validate_workspace(os.environ['DAILY_WORK'], Path.cwd(),
                                      sys.prefix, os.environ['ERA5_ROOT'])
    work.mkdir(parents=True, exist_ok=True)
    print(f'Write root: {allowed}; experiment: {work}', flush=True)


if __name__ == '__main__':
    main()
