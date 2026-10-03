"""Bind a Fusion experiment directory to its exact inputs and configuration.

Only prepared inputs are read here, never the original ERA5 source tree. Resume
means skipping completed fits, not restoring an optimizer from a partial fit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def digest(path):
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Input changed while fingerprinting: {path}')
    return {'bytes': after.st_size, 'sha256': result.hexdigest()}


def identity(path):
    path = Path(path).resolve(strict=True)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'Empty prepared input: {path}')
    return {'path': str(path), 'files': {
        str(p.relative_to(path)) if path.is_dir() else path.name: digest(p) for p in files}}


def archive_step(archive, information):
    archive, information = Path(archive), Path(information)
    if not archive.is_file() or not information.exists():
        raise ValueError('Missing prepared ARCHIVE/INFO')
    schema = json.loads(archive.with_suffix('.schema.json').read_text())
    step = schema.get('forecast_step_hours')
    if step not in (6, 24):
        raise ValueError('Use a 6-hour or 24-hour prepared archive')
    return int(step)


def source_identity(root):
    root = Path(root).resolve(strict=True)
    paths = sorted((root/'src').rglob('*.py'))
    paths += [root/'scripts'/name for name in (
        'run_guided_fusion_comparison.sh', 'run_daily_manifold_fusion.sh',
        'run_daily_guided_transformer.sh')]
    paths += [root/'pyproject.toml']
    return {str(path.relative_to(root)): digest(path) for path in paths}


def verify_completed_fits(run, settings):
    for model in settings['models'].split():
        for seed in settings['seeds'].split():
            for arm in settings['arms'].split():
                prefix = run/f'{model}-{arm}-seed{seed}'
                checkpoint = prefix.with_suffix('.pt')
                manifest = prefix.with_suffix('.manifest.json')
                reports = [prefix.with_suffix(f'.{split}.json') for split in ('validation', 'test')]
                if not checkpoint.exists() and not manifest.exists():
                    if any(p.exists() for p in reports):
                        raise ValueError(f'Report without a completed checkpoint: {prefix}')
                    continue
                if not checkpoint.is_file() or not manifest.is_file():
                    raise ValueError(f'Incomplete fit artifacts: {prefix}. Preserve these files and use a new RUN; optimizer resume is unsupported.')
                checksum = digest(checkpoint)['sha256']
                if json.loads(manifest.read_text()).get('checkpoint_sha256') != checksum:
                    raise ValueError(f'Checkpoint manifest/hash mismatch: {checkpoint}')
                for report in reports:
                    if report.exists() and json.loads(report.read_text()).get('checkpoint_sha256') != checksum:
                        raise ValueError(f'Report/checkpoint mismatch: {report}')


def initialize_run(run, archive, information, source_root, settings, arguments, *, resume):
    run = Path(run).resolve()
    archive = Path(archive).resolve(strict=True)
    information = Path(information).resolve(strict=True)
    if run == archive.parent or run == information or run.is_relative_to(information):
        raise ValueError('RUN must be separate from prepared inputs')
    manifest = run/'run-manifest.json'
    if run.exists() and not resume:
        raise ValueError('Choose a new RUN (or RESUME=1 with the same experiment)')
    if resume and not manifest.is_file():
        raise ValueError('RESUME=1 requires an existing run-manifest.json; use a new RUN for older runs')
    versions = {}
    for package in ('torch', 'numpy', 'xarray', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    expected = {
        'format': 'climate_manifold.fusion_run.v1',
        'settings': settings, 'arguments': arguments,
        'inputs': {'archive': identity(archive),
                   'schema': identity(archive.with_suffix('.schema.json')),
                   'information': identity(information)},
        'source': source_identity(source_root),
        'runtime': {'python': sys.version, 'packages': versions},
    }
    if resume:
        actual = json.loads(manifest.read_text())
        differences = [key for key in expected if actual.get(key) != expected[key]]
        if differences:
            raise ValueError('Refusing RESUME: run manifest differs in '+', '.join(differences)+'. Use a new RUN; existing artifacts were not changed.')
        verify_completed_fits(run, settings)
    else:
        run.mkdir(parents=True, exist_ok=False)
        # Exclusive create: never replace the contract of an existing run.
        with manifest.open('x') as stream:
            json.dump(expected, stream, indent=2, sort_keys=True)
            stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--information', required=True)
    parser.add_argument('--step-only', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--source-root')
    parser.add_argument('--resume', type=int, choices=(0, 1), default=0)
    parser.add_argument('--setting', nargs=2, action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        step = archive_step(args.archive, args.information)
        if args.step_only:
            print(step)
            return
        if not args.run or not args.source_root:
            parser.error('--run and --source-root are required unless --step-only')
        settings = dict(args.setting)
        if len(settings) != len(args.setting):
            parser.error('Duplicate setting')
        manifest = initialize_run(args.run, args.archive, args.information, args.source_root,
                                  settings, args.arguments, resume=bool(args.resume))
        print(f'Experiment contract: {manifest}')
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
