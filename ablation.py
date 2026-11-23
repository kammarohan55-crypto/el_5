#!/usr/bin/env python3
"""
ablation.py - experiment registry, Kaggle session queue and results collector for DPF-Net (v2).

  python ablation.py list
  python ablation.py weights --queue B6 V6          # pre-download the encoder + foundation models of a queue
  python ablation.py run --work /kaggle/working --queue B3 B6 B4 [--prev /kaggle/input] [--session-hours 11.5]
  python ablation.py collect --roots /kaggle/working/runs /kaggle/input --out /kaggle/working/ablation

Registry v2 (B* = SAT, V* = UAV): every row is trained with the same precision ('--precision auto' = FP16 with an
FP32 recompute of any step whose FP16 forward overflows), so rows differ only in the component under test. The
v1 rows (A*, U*: mixed FP16 / imported FP32 runs) are no longer queued; their results are listed as earlier runs.

run      Runs the queued experiments one after another (torchrun on all GPUs). Runs found below --prev (outputs of
         earlier notebook versions attached as inputs) are carried forward into <work>/runs; an unfinished run is
         resumed from its last.pth. Each run gets the session time that is left minus a reserve for its final
         evaluation; when the session is nearly over the queue stops cleanly and the next session (with this
         output attached as input) continues it. --import ID=best.pth evaluates an existing checkpoint (e.g. the
         earlier SP-UNet runs) as experiment ID instead of training it again.
collect  Builds the ablation / comparison tables (Markdown, CSV, LaTeX), paired-bootstrap significance tests on
         per-tile counts, per-region tables and figures from every results.json found below --roots. Also works
         offline on downloaded run folders (only results.json, history.json and per_tile_test.csv are read).
"""
import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FM_SAT = 'vit_large_patch16_dinov3.sat493m'   # DINOv3 ViT-L/16 distilled from the satellite (SAT-493M) 7B model
FM_WEB = 'vit_large_patch16_dinov3.lvd1689m'   # same architecture, web-image (LVD-1689M) pretraining
TEST_TILES = {'SAT': 2218, 'UAV': 4034}
VAL_TILES = {'SAT': 521, 'UAV': 941}
TORCHRUN_ENV = {'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1', 'OMP_NUM_THREADS': '2', 'PYTHONIOENCODING': 'utf-8'}
SKIP_DIRS = {'img', 'imgs', 'image', 'images', 'mask', 'masks', 'label', 'labels', 'vis_test', '__pycache__',
             '.git', 'splits'}

# ----------------------------------------------------------------------------------------
# experiment registry v2 (every row: ConvNeXt-T DINOv3 encoder, 80 epochs, seed 42, '--precision auto', same
# evaluation; rows with prior substitution are also evaluated without the foundation model -> row '<id>:lite')
# ----------------------------------------------------------------------------------------
EXPERIMENTS = OrderedDict()
BASE = ['--no-hires', '--no-prior', '--boundary-weight', '0', '--contrast-weight', '0']


def fuse(fm=FM_SAT, sub='hal', mining='error', fusion='gated'):
    return ['--fm', fm, '--fm-mode', 'fuse', '--fm-fusion', fusion, '--fm-sub', sub, '--hard-mining', mining]


def _exp(eid, subset, title, args, group, priority):
    EXPERIMENTS[eid] = {'id': eid, 'subset': subset, 'title': title, 'args': list(args), 'group': group,
                        'priority': priority}


_exp('B0', 'SAT', 'UNet + PPM baseline', ['--fm', 'none'] + BASE, 'component', 1)
_exp('B1', 'SAT', '+ spectral-structural prior & high-res stem',
     ['--fm', 'none', '--boundary-weight', '0', '--contrast-weight', '0'], 'component', 2)
_exp('B2', 'SAT', '+ boundary head', ['--fm', 'none', '--contrast-weight', '0'], 'component', 2)
_exp('B3', 'SAT', '+ error-mined pixel contrast (= SP-UNet)', ['--fm', 'none'], 'component', 1)
_exp('B4', 'SAT', '+ frozen-FM prior, gated fusion (v1 DPF-Net)', fuse(sub='none'), 'component', 1)
_exp('B5', 'SAT', '+ prior dropout (p = 0.5)', fuse(sub='zero'), 'component', 2)
_exp('B6', 'SAT', '+ prior hallucination = DPF-Net v2 (ours)', fuse(sub='hal'), 'component', 1)
_exp('B7', 'SAT', 'DPF-Net v2 + semantic hard-negative mining', fuse(sub='hal', mining='error+fm'), 'mining', 2)
_exp('B8', 'SAT', 'FM feature distillation only (FM-free baseline)',
     ['--fm', FM_SAT, '--fm-mode', 'distill', '--hard-mining', 'error'], 'efficiency', 2)
_exp('B9', 'SAT', 'DPF-Net v2 with a web-image FM', fuse(fm=FM_WEB), 'pretraining', 3)
_exp('C6', 'SAT', 'DPF-Net v2 without the structural prior / stem', fuse() + ['--no-hires', '--no-prior'],
     'complementarity', 2)
_exp('G6', 'SAT', 'DPF-Net v2 with additive instead of gated fusion', fuse(fusion='add'), 'fusion', 3)
_exp('P0', 'SAT', 'SP-UNet, plain FP16 (precision control)', ['--fm', 'none', '--precision', 'fp16'], 'precision', 3)
_exp('R3', 'SAT', 'SP-UNet, full FP32 (reference)', ['--fm', 'none', '--precision', 'fp32'], 'precision', 3)
_exp('S1', 'SAT', 'DPF-Net v2, seed 1', fuse() + ['--seed', '1'], 'seeds', 3)
_exp('S2', 'SAT', 'DPF-Net v2, seed 2', fuse() + ['--seed', '2'], 'seeds', 3)
_exp('V0', 'UAV', 'UAV: UNet + PPM baseline', ['--fm', 'none'] + BASE, 'uav', 2)
_exp('V3', 'UAV', 'UAV: SP-UNet', ['--fm', 'none'], 'uav', 1)
_exp('V4', 'UAV', 'UAV: v1 DPF-Net (frozen-FM fusion)', fuse(sub='none'), 'uav', 3)
_exp('V6', 'UAV', 'UAV: DPF-Net v2 (ours)', fuse(sub='hal'), 'uav', 1)
_exp('W3', 'UAV', 'UAV: SP-UNet, full FP32 (reference)', ['--fm', 'none', '--precision', 'fp32'], 'precision', 3)

# comparisons tested with the paired bootstrap (reference, candidate): candidate - reference; 'X:lite' = run X
# evaluated without the foundation model
PAIRS = [('B0', 'B1'), ('B1', 'B2'), ('B2', 'B3'), ('B3', 'B4'), ('B4', 'B5'), ('B4', 'B6'), ('B5', 'B6'),
         ('B3', 'B6'), ('B3', 'B6:lite'), ('B6:lite', 'B6'), ('B8', 'B6:lite'), ('B3', 'B8'), ('B6', 'B7'),
         ('B9', 'B6'), ('C6', 'B6'), ('G6', 'B6'), ('B0', 'B6'), ('P0', 'B3'), ('R3', 'B3'),
         ('V0', 'V3'), ('V3', 'V6'), ('V3', 'V6:lite'), ('V4', 'V6'), ('V0', 'V6'), ('W3', 'V3')]
CHAIN = {'SAT': ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B6:lite', 'B7', 'B8', 'B9', 'C6', 'G6'],
         'UAV': ['V0', 'V3', 'V4', 'V6', 'V6:lite']}
BASELINE = {'SAT': 'B0', 'UAV': 'V0'}
SEEDS = ['B6', 'S1', 'S2']
LEGACY = {'fm': 'none', 'fm_mode': 'fuse', 'fm_fusion': 'gated', 'hard_mining': 'error', 'fm_sub': 'none'}


def fm_enabled(name):
    return name not in (None, '', 'none')


def garg(g, key):
    return g.get(key, LEGACY.get(key))


def short_fm(name):
    if name == FM_SAT:
        return 'DINOv3-SAT ViT-L'
    if name == FM_WEB:
        return 'DINOv3-LVD ViT-L'
    return name


def get_exp(eid):
    if eid not in EXPERIMENTS:
        raise SystemExit(f'unknown experiment "{eid}"; known: {", ".join(EXPERIMENTS)}')
    return EXPERIMENTS[eid]


def map_fm(args, fm_map):
    """Replaces foundation-model names (smoke tests use a tiny ViT instead of ViT-L)."""
    out = list(args)
    for i in range(len(out) - 1):
        if out[i] == '--fm' and out[i + 1] in fm_map:
            out[i + 1] = fm_map[out[i + 1]]
    return out


def parse_map(items, what):
    m = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'{what}: expected KEY=VALUE, got "{it}"')
        k, v = it.split('=', 1)
        m[k.strip()] = v.strip()
    return m


def split_args(s):
    """Command-line words of a string; keeps Windows backslashes (POSIX rules elsewhere, e.g. Kaggle)."""
    return shlex.split(s or '', posix=os.name != 'nt')


def load_json(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def stream(cmd, env=None):
    cmd = [str(c) for c in cmd]
    print('$', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding='utf-8', errors='replace', env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end='', flush=True)
    return p.wait()


# ----------------------------------------------------------------------------------------
# run folders
# ----------------------------------------------------------------------------------------
def find_run_dirs(roots, max_depth=5):
    """Folders containing results.json / last.pth / status.json below the roots (bounded, skips data folders)."""
    found, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            p, depth = stack.pop()
            try:
                entries = list(os.scandir(p))
            except OSError:
                continue
            names = {e.name for e in entries}
            if names & {'results.json', 'last.pth', 'status.json'}:
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(p)
                continue
            if depth >= max_depth:
                continue
            for e in entries:
                if e.is_dir(follow_symlinks=False) and e.name.lower() not in SKIP_DIRS:
                    stack.append((Path(e.path), depth + 1))
    return found


def run_tag(p):
    for f in ('results.json', 'status.json'):
        j = load_json(Path(p) / f)
        if j and j.get('tag'):
            return j['tag']
    return Path(p).name


def run_state(p):
    """('done' | 'partial' | 'evaluated-incomplete' | 'empty', epochs done)."""
    p = Path(p)
    if not p.is_dir():
        return 'empty', 0
    res = load_json(p / 'results.json')
    hist = load_json(p / 'history.json') or []
    ep = len(hist) if isinstance(hist, list) else 0
    if res is not None and res.get('schedule_complete') is not False:
        return 'done', ep
    if (p / 'last.pth').exists():
        return 'partial', ep
    if res is not None:
        return 'evaluated-incomplete', ep
    return 'empty', ep


def _rank(state):
    return {'done': 3, 'evaluated-incomplete': 1, 'partial': 2, 'empty': 0}[state[0]] * 10000 + state[1]


def carry_forward(prev_roots, runs_dir, keep_last=False):
    """Copies registry runs found in earlier outputs into runs_dir (the more advanced copy wins)."""
    runs_dir = Path(runs_dir).resolve()
    for p in find_run_dirs(prev_roots):
        rp = Path(p).resolve()
        if rp == runs_dir or runs_dir in rp.parents:
            continue
        tag = run_tag(p)
        if tag not in EXPERIMENTS:
            continue
        dst = runs_dir / tag
        st_src, st_dst = run_state(p), run_state(dst)
        if dst.exists() and _rank(st_dst) >= _rank(st_src):
            continue
        if dst.exists():
            shutil.rmtree(dst)
        skip = [] if (keep_last or st_src[0] != 'done') else ['last.pth']
        shutil.copytree(p, dst, ignore=shutil.ignore_patterns(*skip, '*.tmp'))
        print(f'[carry] {tag}: {st_src[0]} ({st_src[1]} epochs) from {p}', flush=True)


# ----------------------------------------------------------------------------------------
# list / weights
# ----------------------------------------------------------------------------------------
def cmd_list(a):
    print(f'{"id":4} {"subset":6} {"prio":4} {"group":15} title / extra arguments')
    for e in EXPERIMENTS.values():
        print(f'{e["id"]:4} {e["subset"]:6} {e["priority"]:<4} {e["group"]:15} {e["title"]}')
        print(f'{"":31} {" ".join(e["args"])}')


def resolved_args(eid, a, extra=()):
    sys.path.insert(0, str(HERE))
    import train_landslide as T  # noqa: E402  (heavy import only for run / weights)
    e = get_exp(eid)
    argv = ['--splits', 'x', '--out', 'y'] + map_fm(e['args'], parse_map(getattr(a, 'fm_map', None), '--fm-map'))
    return T.parse_args(argv + list(extra))


def cmd_weights(a):
    import timm
    extra = split_args(a.extra)
    names = []
    for eid in a.queue:
        ta = resolved_args(eid, a, extra)
        for n in [ta.encoder] + ([ta.fm] if fm_enabled(ta.fm) else []):
            if n not in names:
                names.append(n)
    for n in names:
        t0 = time.time()
        m = timm.create_model(n, pretrained=True)
        print(f'weights OK: {n} ({sum(p.numel() for p in m.parameters()) / 1e6:.1f} M params, '
              f'{time.time() - t0:.0f} s)', flush=True)
        del m


# ----------------------------------------------------------------------------------------
# run (queue)
# ----------------------------------------------------------------------------------------
def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def launcher(nproc):
    if nproc > 1:
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}']
    return [sys.executable]


def eval_reserve_hours(ta, subset, nproc):
    """Conservative estimate of the final evaluation time (T4, FP16): validation sweep + test, all TTA passes;
    models with prior substitution are evaluated twice (with and without the foundation model)."""
    passes = 1 if ta.no_tta else (12 if ta.ms_tta else 8)
    fused = fm_enabled(ta.fm) and ta.fm_mode == 'fuse'
    n = VAL_TILES.get(subset, 1000) + TEST_TILES.get(subset, 3000)
    sec = n * passes / ((9.0 if fused else 16.0) * max(1, nproc))  # single-pass images / s
    if fused and getattr(ta, 'fm_sub', 'none') != 'none':
        sec += n * passes / (16.0 * max(1, nproc))  # 'lite' evaluation, no foundation model
    return sec / 3600 * 1.15 + 0.12


def ensure_splits(work, subset, prep_args):
    sp = Path(work) / 'splits' / subset
    if (sp / 'train.csv').exists() and (sp / 'test.csv').exists():
        return sp
    rc = stream([sys.executable, HERE / 'prepare_data.py', '--subset', subset, '--out', Path(work) / 'splits']
                + split_args(prep_args))
    if rc != 0:
        raise SystemExit(f'prepare_data.py failed for {subset} (exit {rc})')
    return sp


def cmd_run(a):
    t0 = a.t0 if a.t0 else time.time()
    deadline = t0 + a.session_hours * 3600
    work = Path(a.work)
    runs = work / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    prev = [p for p in (a.prev or []) if Path(p).is_dir()]
    if prev:
        carry_forward(prev, runs, a.keep_last)
    imports = parse_map(a.imports, '--import')
    fm_map = parse_map(a.fm_map, '--fm-map')
    extra = split_args(a.extra)
    nproc = a.nproc or max(1, gpu_count())
    queue = list(a.queue)
    if a.quick:
        queue = queue[:1]
    summary = []
    for eid in queue:
        e = get_exp(eid)
        ta = resolved_args(eid, a, extra)
        tag = eid + ('_quick' if a.quick else '')
        rd = runs / tag
        state, ep_done = run_state(rd)
        if state == 'done':
            summary.append((tag, 'done (already finished)'))
            continue
        sp = ensure_splits(work, e['subset'], a.prep_args)
        if state == 'empty' and eid in imports:  # evaluate an existing checkpoint as this experiment
            ck = Path(imports[eid])
            if not ck.is_file():
                summary.append((tag, f'IMPORT FAILED: {ck} not found'))
                continue
            left = (deadline - time.time()) / 3600
            need = eval_reserve_hours(ta, e['subset'], nproc)
            if left < need:
                summary.append((tag, f'import not started: {left:.2f} h left, ~{need:.2f} h needed'))
                break
            rd.mkdir(parents=True, exist_ok=True)
            cmd = (launcher(nproc) + [HERE / 'train_landslide.py', '--eval-only', '--eval-ckpts', ck, '--splits', sp,
                                      '--out', rd, '--tag', tag, '--batch-size', a.batch_size]
                   + (['--ms-tta'] if ta.ms_tta else []) + extra)
            rc = stream(cmd, TORCHRUN_ENV)
            if (ck.parent / 'history.json').exists() and not (rd / 'history.json').exists():
                shutil.copy2(ck.parent / 'history.json', rd / 'history.json')
            summary.append((tag, 'imported (evaluated)' if rc == 0 else f'IMPORT FAILED (exit {rc})'))
            continue
        left = (deadline - time.time()) / 3600
        reserve = a.eval_reserve_hours if a.eval_reserve_hours is not None else eval_reserve_hours(ta, e['subset'], nproc)
        budget = left - reserve
        if budget < a.min_start_hours:
            summary.append((tag, f'not started: {left:.2f} h left in this session (eval reserve {reserve:.2f} h)'))
            break
        cmd = (launcher(nproc) + [HERE / 'train_landslide.py', '--splits', sp, '--out', rd, '--tag', tag,
                                  '--epochs', 1 if a.quick else a.epochs, '--batch-size', a.batch_size,
                                  '--max-train-hours', f'{budget:.3f}', '--no-eval-on-timeout']
               + map_fm(e['args'], fm_map) + extra)
        if state == 'partial':
            cmd += ['--resume', rd / 'last.pth']
            print(f'[queue] {tag}: resuming after epoch {ep_done}', flush=True)
        print(f'[queue] {tag}: {e["title"]} | {left:.2f} h left, training budget {budget:.2f} h, '
              f'evaluation reserve {reserve:.2f} h', flush=True)
        t_run = time.time()
        rc = stream(cmd, TORCHRUN_ENV)
        st2, ep2 = run_state(rd)
        took = (time.time() - t_run) / 3600
        if rc != 0:
            summary.append((tag, f'FAILED (exit {rc}) after {took:.2f} h - see the log above'))
            if a.stop_on_error:
                break
            continue
        if st2 == 'done':
            summary.append((tag, f'done ({ep2} epochs, {took:.2f} h this session)'))
            if not a.keep_last and (rd / 'last.pth').exists() and not a.quick:
                (rd / 'last.pth').unlink()  # finished: keep best.pth / results only (saves output space)
            continue
        summary.append((tag, f'paused after epoch {ep2} (session budget) - re-run with this output attached'))
        break
    done_ids = {t for t, s in summary}
    for eid in queue:
        if eid not in done_ids and eid + '_quick' not in done_ids:
            summary.append((eid, 'waiting (next session)'))
    print('\n[queue] summary')
    for tag, s in summary:
        print(f'  {tag:8s} {s}')
    log = load_json(work / 'queue_log.json') or []
    log.append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'queue': queue, 'summary': summary})
    write_json(log, work / 'queue_log.json')


# ----------------------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------------------
def discover(roots):
    runs = {}
    for p in find_run_dirs(roots):
        res = load_json(Path(p) / 'results.json')
        if not res or 'main' not in res:
            continue
        tag = res.get('tag') or Path(p).name
        cand = {'dir': Path(p), 'res': res, 'mtime': (Path(p) / 'results.json').stat().st_mtime}
        old = runs.get(tag)
        key = (res.get('schedule_complete') is not False, cand['mtime'])
        if old is None or key > (old['res'].get('schedule_complete') is not False, old['mtime']):
            runs[tag] = cand
    for r in runs.values():
        h = load_json(r['dir'] / 'history.json')
        r['hist'] = h if isinstance(h, list) else []
    return runs


def margs(r):
    return r['res'].get('model_args') or r['res']['args']


def abl_key(res):
    return res.get('ablation_key') or ('test_TTA@t*' if res.get('tta') else 'test_noTTA@t*')


SUB_SHORT = {'none': '', 'zero': ' + prior dropout', 'hal': ' + hallucination'}


def precision_of(g):
    if g.get('precision'):
        return g['precision']
    return 'fp32' if g.get('no_amp') else 'fp16'


def components(r, lite=False):
    g = margs(r)
    struct = 'yes' if not g.get('no_prior') else ('stem only' if not g.get('no_hires') else 'no')
    bnd = 'yes' if g.get('boundary_weight', 0) > 0 else 'no'
    con = garg(g, 'hard_mining') if g.get('contrast_weight', 0) > 0 else 'no'
    fm = garg(g, 'fm')
    if not fm_enabled(fm):
        sem = 'no'
    elif garg(g, 'fm_mode') == 'distill':
        sem = f'{short_fm(fm)}, training only'
    else:
        sem = f'{short_fm(fm)}, {garg(g, "fm_fusion")}{SUB_SHORT.get(garg(g, "fm_sub") or "none", "")}'
        if lite:
            sem += ' (FM not used at inference)'
    return {'struct': struct, 'boundary': bnd, 'contrast': con, 'semantic': sem, 'precision': precision_of(g)}


def title_of(tag, r, lite=False):
    if tag in EXPERIMENTS:
        t = EXPERIMENTS[tag]['title']
    else:  # earlier runs (e.g. the v1 'A6' found in an attached output)
        t = f'earlier run {r["res"].get("run", tag)}'
    return t + (' - without the FM at inference' if lite else '')


def entries_of(runs):
    """Table rows: every run, plus '<id>:lite' for runs also evaluated without the foundation model."""
    order = [t for t in EXPERIMENTS if t in runs] + sorted(t for t in runs if t not in EXPERIMENTS)
    ent = OrderedDict()
    for tag in order:
        r = runs[tag]
        res = r['res']
        ent[tag] = {'tag': tag, 'run': r, 'key': abl_key(res), 'main_key': res.get('main_key'), 'lite': False}
        lk = res.get('lite_ablation_key')
        if lk and res.get(lk):
            ent[tag + ':lite'] = {'tag': tag, 'run': r, 'key': lk, 'main_key': res.get('lite_key') or lk, 'lite': True}
    return ent


def key_mode(key):
    """'test_TTA-lite@t*' -> 'TTA-lite'."""
    return key.split('_', 1)[1].split('@')[0]


def read_tiles(r, mode):
    f = r['dir'] / (r['res'].get('per_tile_file') or 'per_tile_test.csv')
    if not f.exists():
        return None
    out = {}
    with open(f, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                out[row['name']] = [float(row[f'{mode}_{k}']) for k in ('tp', 'fp', 'fn', 'tn')]
            except KeyError:
                return None
    return out


def _iou(s):
    import numpy as np
    den = s[..., 0] + s[..., 1] + s[..., 2]
    return np.where(den > 0, s[..., 0] / np.maximum(den, 1e-12), 0.0)


def bootstrap(A, B=None, n_boot=10000, seed=0, chunk=250):
    """Tile-level bootstrap of the dataset-level IoU (A, B: n x 4 count arrays of the same tiles, paired)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    ia, ib, done = [], [], 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        ia.append(_iou(A[idx].sum(axis=1)))
        if B is not None:
            ib.append(_iou(B[idx].sum(axis=1)))
        done += k
    ia = np.concatenate(ia)
    out = {'iou': float(_iou(A.sum(0))), 'ci': [float(v) for v in np.percentile(ia, [2.5, 97.5])]}
    if B is not None:
        ib = np.concatenate(ib)
        dd = ib - ia
        out.update({'iou_b': float(_iou(B.sum(0))), 'delta': float(_iou(B.sum(0)) - _iou(A.sum(0))),
                    'delta_ci': [float(v) for v in np.percentile(dd, [2.5, 97.5])],
                    'p': float(min(1.0, 2.0 * min((dd <= 0).mean(), (dd >= 0).mean())))})
    return out


def holm(ps):
    """Holm-Bonferroni adjusted p-values (same order as given)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def pct(v, nd=2):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.{nd}f}'


def fmt_p(p, n_boot):
    return f'<{1.0 / n_boot:.0e}' if p == 0 else (f'{p:.4f}' if p < 0.01 else f'{p:.3f}')


def tex_escape(s):
    return (str(s).replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%')
            .replace('_', '\\_').replace('#', '\\#'))


def cmd_collect(a):
    import numpy as np
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = discover(a.roots)
    if not runs:
        print('collect: no results.json found below', a.roots)
        return
    ent = entries_of(runs)
    summary = {'runs': {}, 'pairs': [], 'seeds': {}}

    # ---- per-entry rows (a run, or a run evaluated without the foundation model: '<id>:lite') -------------------
    rows = []
    for eid, e in ent.items():
        r, res = e['run'], e['run']['res']
        g = margs(r)
        m = res[e['key']]
        main = res.get(e['main_key']) or m
        st = (res.get('model_stats_lite') if e['lite'] else res.get('model_stats')) or {}
        subset = (res.get('subset') or EXPERIMENTS.get(e['tag'], {}).get('subset') or 'SAT')
        base = ent.get(BASELINE.get(subset, ''))
        d0 = (m['iou'] - base['run']['res'][base['key']]['iou']) if base and base is not e else None
        comp = components(r, e['lite'])
        hist = r.get('hist') or []
        row = OrderedDict([
            ('id', eid), ('subset', subset), ('configuration', title_of(e['tag'], r, e['lite'])),
            ('struct_prior', comp['struct']), ('boundary', comp['boundary']), ('contrast_mining', comp['contrast']),
            ('semantic_prior', comp['semantic']), ('precision', comp['precision']),
            ('epochs', res.get('epochs_done') or g.get('epochs')),
            ('complete', res.get('schedule_complete')), ('eval_setting', key_mode(e['key'])),
            ('P', m['precision']), ('R', m['recall']), ('F1', m['f1']), ('IoU', m['iou']), ('mIoU', m['miou']),
            ('dIoU_vs_baseline', d0), ('main_setting', key_mode(e['main_key']) if e['main_key'] else ''),
            ('main_IoU', main['iou']), ('main_F1', main['f1']),
            ('params_M', st.get('params_M')),
            ('inference_params_M', st.get('inference_params_lite_M' if e['lite'] else 'inference_params_M')),
            ('gflops', st.get('gflops')), ('fps', st.get('fps_batch8')),
            ('train_hours', res.get('train_hours_total', res.get('train_hours'))),
            ('fp32_recomputes', sum(int(h.get('fp32_recomputes') or 0) for h in hist) if hist else None)])
        rows.append(row)
        summary['runs'][eid] = {k: v for k, v in row.items()}
        summary['runs'][eid]['dir'] = str(r['dir'])

    with open(out / 'ablation_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for row in rows:
            w.writerow(['' if v is None else (f'{v:.6f}' if isinstance(v, float) else v) for v in row.values()])

    def num(v, nd=1):
        return '-' if v is None else f'{v:.{nd}f}'

    md = ['# Ablation and comparison (test split)', '',
          'IoU / F1 / P / R in % for the landslide class from dataset-level pixel counts. "Eval" is the setting '
          'compared across rows (validation-selected threshold, D4-TTA unless stated); "Main" is the '
          'validation-selected main setting of each run (may add multi-scale TTA). dIoU = IoU - baseline IoU '
          '(B0 for SAT, V0 for UAV), percentage points. Rows "<id>:lite" are the same trained network evaluated '
          'WITHOUT the foundation model (hallucinated prior / no prior). "Prec." = training precision (auto = '
          'FP16 with FP32 recompute of overflowing steps).', '']
    for subset in ('SAT', 'UAV'):
        sub = [r for r in rows if r['subset'] == subset]
        if not sub:
            continue
        md += [f'## {subset}', '',
               '| ID | Configuration | Struct. prior | Boundary | Contrast mining | Semantic prior | Prec. | Ep. | P | R | '
               'F1 | IoU | dIoU | Main IoU | Train. params (M) | Infer. params (M) | GFLOPs | FPS | Train h |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in sub:
            d = '-' if r['dIoU_vs_baseline'] is None else f'{100 * r["dIoU_vs_baseline"]:+.2f}'
            inc = '' if r['complete'] is not False else ' (incomplete)'
            md.append(f'| {r["id"]} | {r["configuration"]}{inc} | {r["struct_prior"]} | {r["boundary"]} | '
                      f'{r["contrast_mining"]} | {r["semantic_prior"]} | {r["precision"]} | {r["epochs"]} | '
                      f'{pct(r["P"])} | {pct(r["R"])} | {pct(r["F1"])} | **{pct(r["IoU"])}** | {d} | '
                      f'{pct(r["main_IoU"])} ({r["main_setting"]}) | {num(r["params_M"], 2)} | '
                      f'{num(r["inference_params_M"], 1)} | {num(r["gflops"])} | {num(r["fps"])} | '
                      f'{num(r["train_hours"], 2)} |')
        md.append('')

    # ---- LaTeX (booktabs) ---------------------------------------------------------------------------
    tex = ['% requires \\usepackage{booktabs,amssymb}', '\\begin{tabular}{llccccrrr}', '\\toprule',
           'ID & Configuration & Struct. & Bnd. & Contrast & Semantic & IoU (\\%) & F1 (\\%) & $\\Delta$IoU \\\\',
           '\\midrule']
    yes = {'yes': '\\checkmark', 'no': '--', 'stem only': 'stem'}
    for r in rows:
        d = '--' if r['dIoU_vs_baseline'] is None else f'{100 * r["dIoU_vs_baseline"]:+.2f}'
        tex.append(f'{tex_escape(r["id"])} & {tex_escape(r["configuration"])} & {yes.get(r["struct_prior"], "")} & '
                   f'{yes.get(r["boundary"], "")} & {tex_escape(r["contrast_mining"]) if r["contrast_mining"] != "no" else "--"} & '
                   f'{tex_escape(r["semantic_prior"]) if r["semantic_prior"] != "no" else "--"} & {pct(r["IoU"])} & '
                   f'{pct(r["F1"])} & {d} \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    (out / 'ablation_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')

    # ---- paired bootstrap ---------------------------------------------------------------------------
    tiles = {eid: read_tiles(e['run'], key_mode(e['key'])) for eid, e in ent.items()}
    ci = {}
    for eid in ent:
        if tiles[eid]:
            names = sorted(tiles[eid])
            A = np.array([tiles[eid][n] for n in names])
            ci[eid] = bootstrap(A, n_boot=a.n_boot, seed=1)['ci']
            summary['runs'][eid]['IoU_ci95'] = ci[eid]
    sig = ['## Paired bootstrap tests (resampling test tiles; dIoU = candidate - reference)', '',
           f'{a.n_boot} bootstrap samples of the test tiles; 95% percentile interval; two-sided bootstrap p-value and '
           'its Holm adjustment over all comparisons listed. Both runs evaluated in the "Eval" setting of the '
           'ablation table.', '',
           '| Reference | Candidate | IoU ref | IoU cand | dIoU (pp) | 95% CI (pp) | p | p (Holm) | tiles |',
           '|---|---|---|---|---|---|---|---|---|']
    for ref, cand in PAIRS:
        if not (tiles.get(ref) and tiles.get(cand)):
            continue
        names = sorted(set(tiles[ref]) & set(tiles[cand]))
        if len(names) < 10:
            continue
        A = np.array([tiles[ref][n] for n in names])
        B = np.array([tiles[cand][n] for n in names])
        b = bootstrap(A, B, n_boot=a.n_boot, seed=7)
        summary['pairs'].append({'reference': ref, 'candidate': cand, **b, 'tiles': len(names)})
    for pr_, ph in zip(summary['pairs'], holm([pr_['p'] for pr_ in summary['pairs']])):
        pr_['p_holm'] = ph
        sig.append(f'| {pr_["reference"]} | {pr_["candidate"]} | {pct(pr_["iou"])} | {pct(pr_["iou_b"])} | '
                   f'{100 * pr_["delta"]:+.2f} | [{100 * pr_["delta_ci"][0]:+.2f}, {100 * pr_["delta_ci"][1]:+.2f}] | '
                   f'{fmt_p(pr_["p"], a.n_boot)} | {fmt_p(ph, a.n_boot)} | {pr_["tiles"]} |')
    if not summary['pairs']:
        sig.append('| (no pair with per-tile counts for both runs yet) | | | | | | | | |')
    md += sig + ['']

    # ---- seeds ------------------------------------------------------------------------------------
    for sfx in ('', ':lite'):
        seed_tags = [t + sfx for t in SEEDS if t + sfx in summary['runs']]
        if len(seed_tags) < 2:
            continue
        for key, label in (('IoU', 'Eval'), ('main_IoU', 'Main')):
            vals = [summary['runs'][t][key] for t in seed_tags]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
            summary['seeds'][key + sfx] = {'runs': seed_tags, 'mean': mu, 'std': sd}
            md.append(f'Seeds ({", ".join(seed_tags)}), {label} IoU: {100 * mu:.2f} +- {100 * sd:.2f} (mean +- s.d.)')
        md.append('')

    # ---- per-region ---------------------------------------------------------------------------------
    for subset in ('SAT', 'UAV'):
        tags = [eid for eid in ent if summary['runs'][eid]['subset'] == subset]
        if not tags:
            continue
        per = {}
        for t in tags:
            res = ent[t]['run']['res']
            mode = key_mode(ent[t]['key'])
            per[t] = (res.get('per_region_by_mode') or {}).get(mode) or ({} if ent[t]['lite'] else
                                                                        res.get('per_region') or {})
        regions = sorted({rg for t in tags for rg in per[t]})
        if not regions:
            continue
        hdr = '| Region | tiles | ' + ' | '.join(tags) + ' |'
        md += [f'## Per-region IoU, {subset} (Eval setting)', '', hdr, '|' + '---|' * (len(tags) + 2)]
        prc = []
        for rg in regions:
            n_t = next((per[t][rg].get('tiles') for t in tags if rg in per[t]), '')
            vals = [pct(per[t][rg]['iou']) if rg in per[t] else '-' for t in tags]
            md.append(f'| {rg} | {n_t} | ' + ' | '.join(vals) + ' |')
            prc.append([rg, n_t] + [per[t][rg]['iou'] if rg in per[t] else '' for t in tags])
        md.append('')
        with open(out / f'per_region_{subset}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['region', 'tiles'] + tags)
            w.writerows(prc)
        summary[f'per_region_{subset}'] = {t: {rg: per[t][rg]['iou'] for rg in per[t]} for t in tags}

    (out / 'ablation.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    write_json(summary, out / 'summary.json')
    print('\n'.join(md))
    try:
        make_figures(runs, summary, ci, out)
    except Exception as e:  # noqa: BLE001  (figures are optional; tables are the record)
        print(f'collect: figures skipped ({type(e).__name__}: {e})')
    print(f'\ncollect: wrote {out}/ablation.md, ablation_table.csv/.tex, summary.json, per_region_*.csv, fig_*.png/.pdf')


# ----------------------------------------------------------------------------------------
# figures (static, light surface; validated categorical palette, fixed slot order)
# ----------------------------------------------------------------------------------------
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']        # blue, orange, aqua, yellow (validated, light mode)
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
MARKERS = ['o', 's', 'D', '^']


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)


def _save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor=SURFACE)


def make_figures(runs, summary, ci, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.labelcolor': INK2,
                         'text.color': INK, 'axes.titlesize': 9})
    for subset, keys in (('SAT', ['B6', 'B4', 'B3', 'B0']), ('UAV', ['V6', 'V4', 'V3', 'V0'])):
        series = [k for k in keys if k in runs]
        names = {k: (EXPERIMENTS[k]['title'].split('= ')[-1] if k in EXPERIMENTS else k) for k in series}
        names.update({'B6': 'DPF-Net v2', 'B4': 'v1 fusion', 'B3': 'SP-UNet', 'B0': 'Baseline',
                      'V6': 'DPF-Net v2', 'V4': 'v1 fusion', 'V3': 'SP-UNet', 'V0': 'Baseline'})
        # 1. validation IoU per epoch (line chart, <= 4 series, legend)
        curves = [(k, [(h['epoch'], h['val']['iou']) for h in runs[k]['hist'] if h.get('val')]) for k in series]
        curves = [(k, c) for k, c in curves if len(c) >= 2]
        if curves:
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            _style(ax)
            for i, (k, c) in enumerate(curves):
                slot = series.index(k)
                ax.plot([e for e, _ in c], [100 * v for _, v in c], color=SLOTS[slot], linewidth=1.5,
                        solid_capstyle='round', label=f'{names[k]} ({k})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Validation IoU (%), no TTA, t = 0.5')
            ax.set_title(f'CAS-{subset}: validation IoU during training', loc='left', color=INK)
            lo = min(100 * v for _, c in curves for _, v in c[len(c) // 4:])
            ax.set_ylim(max(0.0, lo - 3), None)  # zoom on the converged part; the early ramp is cut
            longest = max((c for _, c in curves), key=len)
            ax.set_xlim(longest[len(longest) // 4][0], longest[-1][0] + 0.5)
            ax.legend(frameon=False, fontsize=8, loc='lower right')
            _save(fig, out, f'fig_val_curves_{subset}')
            plt.close(fig)
        # 2. per-region dot plot (<= 3 series: all-pairs palette cap; marker shape as second channel)
        per = summary.get(f'per_region_{subset}') or {}
        dots = [k for k in series[:3] if k in per]
        if dots:
            regions = sorted({rg for k in dots for rg in per[k]}, key=lambda rg: per[dots[0]].get(rg, 0.0))
            fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(regions) + 0.9))
            _style(ax)
            ax.grid(True, axis='x', color=GRID, linewidth=0.6)
            ax.grid(False, axis='y')
            for y, rg in enumerate(regions):
                vals = [100 * per[k][rg] for k in dots if rg in per[k]]
                if len(vals) > 1:
                    ax.plot([min(vals), max(vals)], [y, y], color=AXIS, linewidth=1.2, zorder=1)
            for i, k in enumerate(dots):
                ys = [y for y, rg in enumerate(regions) if rg in per[k]]
                xs = [100 * per[k][rg] for rg in regions if rg in per[k]]
                ax.scatter(xs, ys, s=30, color=SLOTS[series.index(k)], marker=MARKERS[i], edgecolors=SURFACE,
                           linewidths=1.0, zorder=3, label=f'{names[k]} ({k})')
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions, color=INK2)
            ax.set_xlabel('Test IoU (%)')
            ax.set_title(f'CAS-{subset}: per-region test IoU', loc='left', color=INK)
            ax.legend(frameon=False, fontsize=8, loc='lower right')
            _save(fig, out, f'fig_per_region_{subset}')
            plt.close(fig)
    # 3. ablation forest plot (single series: one hue, no legend; whiskers = 95% bootstrap CI)
    for subset in ('SAT', 'UAV'):
        chain = [t for t in CHAIN[subset] if t in summary['runs'] and t in ci]
        if len(chain) < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(chain) + 0.9))
        _style(ax)
        ax.grid(True, axis='x', color=GRID, linewidth=0.6)
        ax.grid(False, axis='y')
        ys = list(range(len(chain)))[::-1]
        for y, t in zip(ys, chain):
            v = 100 * summary['runs'][t]['IoU']
            lo, hi = 100 * ci[t][0], 100 * ci[t][1]
            ax.plot([lo, hi], [y, y], color=SLOTS[0], linewidth=1.5, solid_capstyle='round', zorder=2)
            ax.scatter([v], [y], s=34, color=SLOTS[0], edgecolors=SURFACE, linewidths=1.0, zorder=3)
            if t in ('B0', 'B3', 'B6', 'B6:lite', 'V0', 'V3', 'V6', 'V6:lite'):
                ax.annotate(f'{v:.2f}', (hi, y), xytext=(5, 0), textcoords='offset points', va='center',
                            fontsize=8, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([f'{t}  {summary["runs"][t]["configuration"]}' for t in chain], color=INK2)
        ax.set_xlabel('Test IoU (%) with 95% bootstrap interval')
        ax.set_title(f'CAS-{subset}: ablation', loc='left', color=INK)
        _save(fig, out, f'fig_ablation_{subset}')
        plt.close(fig)


# ----------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description='DPF-Net experiment queue and results collector')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='print the experiment registry')
    w = sub.add_parser('weights', help='download the encoder + foundation-model weights a queue needs')
    w.add_argument('--queue', nargs='+', required=True)
    w.add_argument('--extra', default='')
    w.add_argument('--fm-map', nargs='*', default=[])
    r = sub.add_parser('run', help='run a queue of experiments within the session budget')
    r.add_argument('--queue', nargs='+', required=True, help='experiment ids, in order (see "list")')
    r.add_argument('--work', default='/kaggle/working')
    r.add_argument('--prev', nargs='*', default=['/kaggle/input'], help='folders with earlier outputs (carried forward)')
    r.add_argument('--epochs', type=int, default=80)
    r.add_argument('--batch-size', type=int, default=8, help='per GPU')
    r.add_argument('--session-hours', type=float, default=11.5, help='hard session limit minus a safety margin')
    r.add_argument('--t0', type=float, default=0.0, help='session start (time.time()); default: now')
    r.add_argument('--min-start-hours', type=float, default=0.4, help='do not start / resume a run with less time')
    r.add_argument('--eval-reserve-hours', type=float, default=None, help='override the evaluation-time estimate')
    r.add_argument('--nproc', type=int, default=0, help='processes (default: number of GPUs)')
    r.add_argument('--extra', default='', help='extra train_landslide.py arguments for every run (use --extra=...)')
    r.add_argument('--import', dest='imports', nargs='*', default=[], metavar='ID=CKPT',
                   help='evaluate an existing checkpoint as experiment ID instead of training it')
    r.add_argument('--prep-args', default='', help='extra prepare_data.py arguments (use --prep-args=...)')
    r.add_argument('--fm-map', nargs='*', default=[], metavar='OLD=NEW', help='replace foundation models (tests)')
    r.add_argument('--keep-last', action='store_true', help='keep last.pth of finished runs')
    r.add_argument('--stop-on-error', action='store_true')
    r.add_argument('--quick', action='store_true', help='1 epoch of the first queued experiment (<id>_quick)')
    c = sub.add_parser('collect', help='build ablation tables, significance tests and figures')
    c.add_argument('--roots', nargs='+', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--n-boot', type=int, default=10000)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {'list': cmd_list, 'weights': cmd_weights, 'run': cmd_run, 'collect': cmd_collect}[a.cmd](a)


if __name__ == '__main__':
    main()
