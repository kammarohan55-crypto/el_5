#!/usr/bin/env python3
"""End-to-end CPU smoke test of prepare_data.py + train_landslide.py + ablation.py on synthetic CAS-shaped data.
Run from anywhere:  python smoke_test/run_smoke.py [--skip-ddp] [--keep]"""
import argparse
import csv
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
TINY_FM = 'vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m'  # random 5.5 M-parameter stand-in for the ViT-L
FM_MAP = [f'vit_large_patch16_dinov3.sat493m={TINY_FM}', f'vit_large_patch16_dinov3.lvd1689m={TINY_FM}']


def run(args, env=None, ok_codes=(0,)):
    args = [str(x) for x in args]
    print('\n$', ' '.join(args), flush=True)
    t0 = time.time()
    p = subprocess.run(args, env={**os.environ, 'PYTHONIOENCODING': 'utf-8', **(env or {})}, cwd=str(ROOT))
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


def tiles_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


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
    for d in ('runs', 'q1', 'q2', 'q3'):
        if (WORK / d).exists():
            shutil.rmtree(WORK / d)
    splits = WORK / 'splits'
    runs = WORK / 'runs'
    common = ['--encoder', 'convnext_atto', '--no-pretrained', '--img-size', '64', '--batch-size', '4',
              '--workers', '0', '--save-vis', '4', '--log-every', '5']
    spu = common + ['--fm', 'none']      # SP-UNet configuration
    fm = common + ['--fm', TINY_FM]      # DPF-Net configuration with a tiny random foundation model

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
    # 2. SP-UNet: train 2 epochs, then resume to 3 -------------------------------------
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'r1', '--epochs', '2', '--debug-limit', '40', *spu])
    r1 = load(runs / 'r1' / 'results.json')
    expect(len(load(runs / 'r1' / 'history.json')) == 2, 'r1: 2 epochs in history')
    expect((runs / 'r1' / 'best.pth').exists() and (runs / 'r1' / 'last.pth').exists(), 'r1: best.pth + last.pth')
    expect(r1['tta'] is True and 'test_TTA@t*' in r1 and 'test_noTTA@0.5' in r1, 'r1: no-TTA and TTA results')
    expect(r1['threshold_source'].startswith('best validation'), 'r1: threshold chosen on validation')
    expect(len(r1['per_region']) >= 1 and 'model_stats' in r1, 'r1: per-region metrics + model stats')
    expect(len(list((runs / 'r1' / 'vis_test').glob('*.png'))) == 4, 'r1: 4 qualitative panels')
    expect((runs / 'r1' / 'results.md').read_text(encoding='utf-8').count('SC-Net') >= 1, 'r1: results.md has benchmarks')
    pt = tiles_csv(runs / 'r1' / 'per_tile_test.csv')
    tot = r1['test_TTA@t*']
    expect(len(pt) == 40 and abs(sum(float(r['TTA_tp']) for r in pt) - tot['tp']) < 0.5
           and abs(sum(float(r['TTA_fn']) for r in pt) - tot['fn']) < 0.5, 'r1: per-tile counts sum to the test counts')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'r1b', '--epochs', '3', '--debug-limit', '40',
         '--resume', runs / 'r1' / 'last.pth', *spu])
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
         '--full-train', *spu])
    f = load(runs / 'full' / 'results.json')
    expect((runs / 'full' / 'final.pth').exists() and f['threshold_source'].startswith('fixed'),
           'full-train: final.pth + fixed threshold')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'base', '--epochs', '1', '--debug-limit', '40',
         '--no-hires', '--no-prior', '--boundary-weight', '0', '--contrast-weight', '0', '--aux-weights', '0', '0', *spu])
    expect(load(runs / 'base' / 'results.json')['args']['no_prior'] is True, 'baseline ablation ran')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'res18', '--epochs', '1', '--debug-limit', '40',
         '--encoder', 'resnet18', '--no-pretrained', '--img-size', '64', '--batch-size', '4', '--workers', '2',
         '--fm', 'none', '--no-tta', '--track-test'])
    r = load(runs / 'res18' / 'results.json')
    expect(r['tta'] is False and r['main_key'] == 'test_noTTA@t*' and 'test_TTA@t*' not in r,
           'no-TTA run labelled correctly')
    expect(load(runs / 'res18' / 'history.json')[0]['test_tracked'] is not None, 'track-test logged')
    # 5. UAV + time guard ----------------------------------------------------------------
    run([PY, ROOT / 'train_landslide.py', '--splits', splits / 'UAV', '--out', runs / 'uav', '--epochs', '1',
         '--debug-limit', '40', *spu])
    pr = load(runs / 'uav' / 'results.json')['per_region']
    expect(len(pr) == 5 and 'Moxi Town (UAV-1m)' in pr, f'UAV run: per-region metrics for {sorted(pr)}')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'guard', '--epochs', '3', '--debug-limit', '40',
         '--max-train-hours', '0', *spu])
    g = load(runs / 'guard' / 'results.json')
    expect(g['schedule_complete'] is False and (runs / 'guard' / 'final.pth').exists(), 'time guard stops and still evaluates')

    # 6. DPF-Net: foundation-model fusion, multi-scale TTA, DPF-Lite, pause / resume, eval-only ----
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'fm1', '--epochs', '2', '--debug-limit', '40',
         '--ms-tta', *fm])
    f1 = load(runs / 'fm1' / 'results.json')
    expect(all(k in f1 for k in ('test_noTTA@0.5', 'test_TTA@t*', 'test_MSTTA@t*', 'val_MSTTA@t*')),
           'fm1: no-TTA, D4-TTA and D4+MS-TTA evaluated')
    expect(f1['tta_mode'] in ('TTA', 'MSTTA') and f1['main_key'] == f'test_{f1["tta_mode"]}@t*'
           and f1['ablation_key'] == 'test_TTA@t*', f'fm1: main setting chosen on validation ({f1["tta_mode"]})')
    expect(f1['model_stats']['fm_frozen_params_M'] > 5 and f1['args']['hard_mining'] == 'error+fm',
           'fm1: frozen foundation model counted, error+fm mining')
    pt = tiles_csv(runs / 'fm1' / 'per_tile_test.csv')
    expect(len(pt) == 40 and {'noTTA_tp', 'TTA_tp', 'MSTTA_tp'} <= set(pt[0])
           and abs(sum(float(x['MSTTA_tp']) for x in pt) - f1['test_MSTTA@t*']['tp']) < 0.5,
           'fm1: per-tile counts for every TTA mode')
    md = (runs / 'fm1' / 'results.md').read_text(encoding='utf-8')
    expect('semantic prior' in md and '(main)' in md and 'D4+MS-TTA' in md, 'fm1: results.md describes DPF-Net')
    expect(load(runs / 'fm1' / 'status.json')['evaluated'] is True, 'fm1: status.json written')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'lite', '--epochs', '1', '--debug-limit', '40',
         '--fm-mode', 'distill', *fm])
    li = load(runs / 'lite' / 'results.json')['model_stats']
    expect(li['fm_frozen_params_M'] == 0 and li['fm_training_only_params_M'] > 5,
           'DPF-Lite: foundation model used only in training')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'fm3', '--epochs', '2', '--debug-limit', '40',
         '--stop-after-epochs', '1', '--no-eval-on-timeout', *fm])
    st = load(runs / 'fm3' / 'status.json')
    expect(st['schedule_complete'] is False and not (runs / 'fm3' / 'results.json').exists()
           and (runs / 'fm3' / 'last.pth').exists(), 'paused run: last.pth kept, evaluation skipped')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'fm3', '--epochs', '2', '--debug-limit', '40',
         '--resume', runs / 'fm3' / 'last.pth', *fm])
    expect([x['epoch'] for x in load(runs / 'fm3' / 'history.json')] == [1, 2]
           and load(runs / 'fm3' / 'results.json')['schedule_complete'] is True, 'paused run resumed in place')
    run([PY, ROOT / 'train_landslide.py', *sat, '--out', runs / 'fm1_eval', '--eval-only', '--debug-limit', '40',
         '--img-size', '64', '--workers', '0', '--eval-ckpts', runs / 'fm1' / 'best.pth'])
    ev = load(runs / 'fm1_eval' / 'results.json')
    expect(ev['model_args']['fm'] == TINY_FM and ev['eval_only'] is True, 'eval-only records the trained model args')
    expect(abs(ev['thresholds']['TTA'] - f1['thresholds']['TTA']) < 1e-9
           and abs(ev['test_TTA@t*']['iou'] - f1['test_TTA@t*']['iou']) < 1e-9,
           'eval-only reproduces the D4-TTA threshold and test IoU of the training run')

    # 7. ablation queue: run, pause, carry forward + resume in a "new session", import --------------
    q1, q2 = WORK / 'q1', WORK / 'q2'
    for d in (q1, q2):
        (d / 'splits').mkdir(parents=True, exist_ok=True)
    prep = f'--prep-args=--input-root {data} --csv-dir {WORK / "scnet_csv"}'
    extra = ['--encoder', 'convnext_atto', '--no-pretrained', '--img-size', '64', '--workers', '0', '--save-vis', '2',
             '--log-every', '5', '--debug-limit', '40']
    qrun = [PY, ROOT / 'ablation.py', 'run', '--epochs', '2', '--batch-size', '4', '--eval-reserve-hours', '0',
            '--min-start-hours', '0', '--fm-map', *FM_MAP, prep]
    run(qrun + ['--work', q1, '--prev', '--queue', 'A0', '--extra=' + ' '.join(extra)])
    expect(load(q1 / 'runs' / 'A0' / 'results.json')['tag'] == 'A0' and not (q1 / 'runs' / 'A0' / 'last.pth').exists(),
           'queue: A0 finished, last.pth of a finished run removed')
    run(qrun + ['--work', q1, '--prev', '--queue', 'A0', 'A6', 'A5',
                '--extra=' + ' '.join(extra + ['--stop-after-epochs', '1'])])
    expect(load(q1 / 'runs' / 'A6' / 'status.json')['schedule_complete'] is False
           and not (q1 / 'runs' / 'A5').exists(), 'queue: A6 paused by the session budget, A5 not started')
    run(qrun + ['--work', q2, '--prev', q1, '--queue', 'A0', 'A6', 'A3', '--import', f'A3={runs / "r1" / "best.pth"}',
                '--extra=' + ' '.join(extra)])
    a6 = load(q2 / 'runs' / 'A6' / 'results.json')
    expect([x['epoch'] for x in load(q2 / 'runs' / 'A6' / 'history.json')] == [1, 2] and a6['schedule_complete'] is True
           and a6['args']['fm'] == TINY_FM and 'test_MSTTA@t*' in a6, 'new session: A6 carried forward and resumed')
    expect((q2 / 'runs' / 'A0' / 'results.json').exists(), 'new session: finished A0 carried forward')
    a3 = load(q2 / 'runs' / 'A3' / 'results.json')
    expect(a3['tag'] == 'A3' and a3['model_args']['fm'] == 'none' and (q2 / 'runs' / 'A3' / 'per_tile_test.csv').exists(),
           'import: SP-UNet checkpoint evaluated as A3')
    run(qrun + ['--work', q2, '--prev', '--queue', 'A5', '--session-hours', '0', '--extra=' + ' '.join(extra)])
    expect(not (q2 / 'runs' / 'A5').exists(), 'queue: no time left -> nothing started')
    expect(len(load(q2 / 'queue_log.json')) == 2, 'queue log kept')

    # 8. collect: tables, significance, figures ------------------------------------------------------
    run([PY, ROOT / 'ablation.py', 'collect', '--roots', q2 / 'runs', '--out', q2 / 'ablation', '--n-boot', '300'])
    md = (q2 / 'ablation' / 'ablation.md').read_text(encoding='utf-8')
    sm = load(q2 / 'ablation' / 'summary.json')
    expect(all(f'| {t} |' in md for t in ('A0', 'A3', 'A6')) and 'Paired bootstrap' in md, 'collect: ablation table')
    expect({(p['reference'], p['candidate']) for p in sm['pairs']} >= {('A3', 'A6'), ('A0', 'A6')},
           'collect: paired bootstrap A3->A6 and A0->A6')
    expect(all((q2 / 'ablation' / f).exists() for f in ('ablation_table.csv', 'ablation_table.tex', 'per_region_SAT.csv',
                                                         'fig_ablation_SAT.png', 'fig_per_region_SAT.pdf',
                                                         'fig_val_curves_SAT.png')), 'collect: CSV / LaTeX / figures')

    # 9. two-process DDP with the foundation model (gloo, CPU) ---------------------------------------
    if not a.skip_ddp:
        ddp_args = [ROOT / 'train_landslide.py', *sat, '--out', runs / 'ddp', '--epochs', '2', '--debug-limit', '41',
                    '--ms-tta', *fm]
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
        pt = tiles_csv(runs / 'ddp' / 'per_tile_test.csv')
        expect(len(pt) == 41 and abs(sum(float(x['MSTTA_tp']) for x in pt) - dd['test_MSTTA@t*']['tp']) < 0.5,
               'DDP per-tile counts gathered from both ranks')
    print('\nALL SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
