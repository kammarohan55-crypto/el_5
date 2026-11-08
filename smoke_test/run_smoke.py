#!/usr/bin/env python3
"""End-to-end CPU smoke test of prepare_data.py + train_landslide.py on synthetic CAS-shaped data.
Run from anywhere:  python smoke_test/run_smoke.py [--skip-ddp] [--keep]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / '_work'
PY = sys.executable


def run(args, env=None, ok_codes=(0,)):
    args = [str(x) for x in args]
    print('\n$', ' '.join(args), flush=True)
    t0 = time.time()
    p = subprocess.run(args, env={**os.environ, **(env or {})}, cwd=str(ROOT))
    print(f'  -> exit {p.returncode} in {time.time() - t0:.1f} s', flush=True)
    if p.returncode not in ok_codes:
        raise SystemExit(f'SMOKE TEST FAILED: {" ".join(args)}')
    return p.returncode


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expect(cond, msg):
    if not cond:
        raise SystemExit(f'SMOKE TEST FAILED: {msg}')
    print(f'  ok  {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ddp', action='store_true')
    ap.add_argument('--keep', action='store_true', help='keep the synthetic data from a previous run')
    a = ap.parse_args()
    data = WORK / 'kaggle_input'
    if not a.keep and WORK.exists():
        shutil.rmtree(WORK)
    if not (data / 'cas-landslide').exists():
        run([PY, HERE / 'make_synthetic_cas.py', '--root', data, '--size', 64])
    splits = WORK / 'splits'
    runs = WORK / 'runs'
    common = ['--encoder', 'convnext_atto', '--no-pretrained', '--img-size', '64', '--batch-size', '4',
              '--workers', '0', '--save-vis', '4', '--log-every', '5']

    # 1. data preparation -------------------------------------------------------------
    run([PY, ROOT / 'prepare_data.py', '--subset', 'SAT', '--input-root', data, '--out', splits,
         '--csv-dir', WORK / 'scnet_csv'])
    s = load(splits / 'SAT' / 'summary.json')
    expect(s['n'] == {'train': 4683, 'val': 521, 'trainval': 5204, 'test': 2218}, f'SAT split sizes {s["n"]}')
    expect(len(s['region_dirs']) == 9, 'nine SAT regions found (UAV/decoy folders ignored)')
    expect(s['per_region']['test']['Tiburon Peninsula (Sentinel)'] == 183, 'Haiti (Sentinel) test tiles = 183')
    expect(s['per_region']['test']['Tiburon Peninsula (Planet)'] == 79, 'Haiti (Planet) test tiles = 79')
    run([PY, ROOT / 'prepare_data.py', '--subset', 'SAT', '--input-root', data, '--out', WORK / 'splits_dl'])
    expect(load(WORK / 'splits_dl' / 'SAT' / 'summary.json')['split_hash'] == s['split_hash'],
           'downloaded SC-Net CSVs give the identical split')
    run([PY, ROOT / 'prepare_data.py', '--subset', 'UAV', '--input-root', data, '--out', splits,
         '--allow-count-mismatch'])
    u = load(splits / 'UAV' / 'summary.json')
    expect(u['n'] == {'train': 133, 'val': 14, 'trainval': 147, 'test': 63}, f'UAV split sizes {u["n"]}')

    sat = ['--splits', splits / 'SAT']
    # 2. train 2 epochs, then resume to 3 -----------------------------------------------
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'r1', '--epochs', '2', '--debug-limit', '40', *common])
    r1 = load(runs / 'r1' / 'results.json')
    expect(len(load(runs / 'r1' / 'history.json')) == 2, 'r1: 2 epochs in history')
    expect((runs / 'r1' / 'best.pth').exists() and (runs / 'r1' / 'last.pth').exists(), 'r1: best.pth + last.pth')
    expect(r1['tta'] is True and 'test_TTA@t*' in r1 and 'test_noTTA@0.5' in r1, 'r1: no-TTA and TTA results')
    expect(r1['threshold_source'].startswith('best validation'), 'r1: threshold chosen on validation')
    expect(len(r1['per_region']) >= 1 and 'model_stats' in r1, 'r1: per-region metrics + model stats')
    expect(len(list((runs / 'r1' / 'vis_test').glob('*.png'))) == 4, 'r1: 4 qualitative panels')
    expect((runs / 'r1' / 'results.md').read_text(encoding='utf-8').count('SC-Net') >= 1, 'r1: results.md has benchmarks')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'r1b', '--epochs', '3', '--debug-limit', '40',
         '--resume', runs / 'r1' / 'last.pth', *common])
    h = load(runs / 'r1b' / 'history.json')
    expect([e['epoch'] for e in h] == [1, 2, 3], 'r1b: resumed history covers epochs 1-3')
    expect(load(runs / 'r1b' / 'results.json')['schedule_complete'] is True, 'r1b: schedule complete')

    # 3. ensemble evaluation -------------------------------------------------------------
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'ens', '--eval-only', '--debug-limit', '40',
         '--img-size', '64', '--workers', '0', '--eval-ckpts', runs / 'r1' / 'best.pth', runs / 'r1b' / 'best.pth'])
    e = load(runs / 'ens' / 'results.json')
    expect(e['weights'].startswith('ensemble of'), 'ensemble evaluated')

    # 4. full-train, ablation baseline, other encoder + workers + no-TTA + track-test ------
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'full', '--epochs', '1', '--debug-limit', '40',
         '--full-train', *common])
    f = load(runs / 'full' / 'results.json')
    expect((runs / 'full' / 'final.pth').exists() and f['threshold_source'].startswith('fixed'),
           'full-train: final.pth + fixed threshold')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'base', '--epochs', '1', '--debug-limit', '40',
         '--no-hires', '--no-prior', '--boundary-weight', '0', '--contrast-weight', '0', '--aux-weights', '0', '0', *common])
    expect(load(runs / 'base' / 'results.json')['args']['no_prior'] is True, 'baseline ablation ran')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'res18', '--epochs', '1', '--debug-limit', '40',
         '--encoder', 'resnet18', '--no-pretrained', '--img-size', '64', '--batch-size', '4', '--workers', '2',
         '--no-tta', '--track-test'])
    r = load(runs / 'res18' / 'results.json')
    expect(r['tta'] is False and r['main_key'] == 'test_noTTA@t*' and 'test_TTA@t*' not in r,
           'no-TTA run labelled correctly')
    expect(load(runs / 'res18' / 'history.json')[0]['test_tracked'] is not None, 'track-test logged')
    # 5. UAV + time guard ----------------------------------------------------------------
    run([PY, ROOT / 'train_landslide.py', '--splits', splits / 'UAV', '--out', runs / 'uav', '--epochs', '1',
         '--debug-limit', '40', *common])
    pr = load(runs / 'uav' / 'results.json')['per_region']
    expect(len(pr) == 5 and 'Moxi Town (UAV-1m)' in pr, f'UAV run: per-region metrics for {sorted(pr)}')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'guard', '--epochs', '3', '--debug-limit', '40',
         '--max-train-hours', '0', *common])
    g = load(runs / 'guard' / 'results.json')
    expect(g['schedule_complete'] is False and (runs / 'guard' / 'final.pth').exists(), 'time guard stops and still evaluates')

    # 6. two-process DDP (gloo, CPU) ---------------------------------------------------------
    if not a.skip_ddp:
        ddp_args = [ROOT / 'train_landslide.py', *sat, '--out', runs / 'ddp', '--epochs', '2', '--debug-limit', '41',
                    *common]
        if os.name == 'nt':
            # Windows PyTorch builds lack libuv and torchrun's launcher store ignores USE_LIBUV, so start the two
            # ranks with exactly the environment torchrun would set (Kaggle/Linux uses torchrun --standalone).
            procs = []
            for rank in range(2):
                env = {**os.environ, 'USE_LIBUV': '0', 'OMP_NUM_THREADS': '1', 'RANK': str(rank),
                       'LOCAL_RANK': str(rank), 'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2',
                       'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29533'}
                print('\n$ [rank', rank, ']', ' '.join(str(x) for x in [PY, *ddp_args]), flush=True)
                procs.append(subprocess.Popen([str(x) for x in [PY, *ddp_args]], env=env, cwd=str(ROOT)))
            codes = [p.wait() for p in procs]
            expect(codes == [0, 0], f'2-process DDP exit codes {codes}')
        else:
            run([PY, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2', *ddp_args],
                env={'OMP_NUM_THREADS': '1'})
        dd = load(runs / 'ddp' / 'results.json')
        tot = dd['test_noTTA@0.5']
        expect(abs(tot['tp'] + tot['fp'] + tot['fn'] + tot['tn'] - 41 * 64 * 64) < 0.5,
               'DDP evaluation counts every test pixel exactly once (41 tiles, odd split)')
    print('\nALL SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
