#!/usr/bin/env python3
"""Builds CAS_Landslide_DPFNet_Kaggle.ipynb from prepare_data.py, train_landslide.py and ablation.py, so the
notebook always contains exactly the tested scripts."""
import json
import uuid
from pathlib import Path

import ablation

HERE = Path(__file__).resolve().parent
OUT = HERE / 'CAS_Landslide_DPFNet_Kaggle.ipynb'
SCRIPTS = ('prepare_data.py', 'train_landslide.py', 'ablation.py')


def md(text):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {},
            'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n').splitlines(keepends=True)}


def experiment_table():
    prio = {1: 'essential', 2: 'recommended', 3: 'optional'}
    lines = ['| ID | Subset | Experiment | Priority |', '|---|---|---|---|']
    for e in ablation.EXPERIMENTS.values():
        p = 'import your earlier FP32 run' if e['id'] in ('R3', 'W3') else prio[e['priority']]
        lines.append(f'| `{e["id"]}` | {e["subset"]} | {e["title"]} | {p} |')
    return '\n'.join(lines)


INTRO = r'''
# CAS Landslide segmentation — DPF-Net v2 experiments (Kaggle, GPU T4 x2)

**Why v2.** The v1 DPF-Net run (A6) reached 77.29 test IoU on CAS-SAT, below the earlier SP-UNet (79.41). The
diagnosis (see `DIAGNOSIS.md` in the repository):
1. **The comparison was confounded.** SP-UNet's 79.41 / 88.49 came from **FP32** reruns (log: `AMP False`), after the
   **FP16** SP-UNet run diverged to NaN at epoch 18. A6 trained in FP16. The DINOv3 ConvNeXt-T has outlier
   activations (up to 160x the mean at 1/16, layer-scale up to 23), and the v1 loop had no protection against an
   overflowing FP16 forward. Corrupted BatchNorm statistics were also copied into the EMA weights.
2. **The frozen-feature fusion is a shortcut.** Under the same FP16 precision, A6 was 1.5–3.8 IoU *ahead* of SP-UNet in
   every epoch from 2 to 17. It still finished lower, in all nine regions, which points to a global effect rather than a
   sensor-specific one. The decoder leans on the frozen 1/16-resolution ViT features and the trainable CNN path
   learns less.
3. **The ViT-L costs 10x the inference FLOPs** (812 vs 80 GFLOPs) and about 60 % of the training step.

**DPF-Net v2** keeps the two priors and fixes the failure modes:
* **spectral-structural prior** (10 differentiable descriptors → high-resolution stem) + boundary head + error-mined
  pixel contrast (= SP-UNet);
* **semantic prior** of a frozen satellite foundation model (DINOv3 ViT-L/16 SAT-493M), injected at 1/8, 1/16 and
  1/32 by **structure-aware gated fusion**;
* **stochastic prior substitution (new)**: in every batch the real prior of a random half of the tiles is replaced by
  a **hallucinated prior** that the network predicts from its own CNN features (trained to match the real prior). This
  removes the shortcut, halves the foundation-model cost in training, and lets the *same* network run **without the
  foundation model** at test time (`:lite` rows, about 80 GFLOPs like SP-UNet);
* **overflow-safe mixed precision (`--precision auto`)**: FP16 speed, but a step whose FP16 forward overflows is
  undone and recomputed in FP32. All rows below use it, so they differ only in the component under test.

`--fm none` reproduces SP-UNet and `--fm-sub none` reproduces the v1 DPF-Net, bit for bit.

**One notebook, all experiments.** Each *Save & Run All* trains the experiments in `QUEUE`, in order, until the 12 h
session is nearly used up. Every run checkpoints. To continue in a new session, attach this notebook's previous
output as an input and run again: finished runs are carried forward, an unfinished run resumes from its checkpoint,
and the tables and figures are rebuilt from everything found.

### Before running
1. *Settings → Accelerator → GPU T4 x2*, *Settings → Internet → On*.
2. *Add Input →* dataset **`ngxunninh/cas-landslide`**.
3. From the 2nd session on: *Add Input → Your Work →* this notebook (its latest output).
4. Recommended: attach the notebook versions with your earlier **FP32** SP-UNet runs and set `IMPORT` so they become
   the reference rows `R3` (SAT) / `W3` (UAV). Imported rows are re-evaluated (≈20 min each), not retrained.
5. Edit CONFIG → *Save Version → Save & Run All (Commit)*. Run once with `QUICK_TEST = True` first (≈25 min) to
   measure the epoch time.

### Experiments
EXPERIMENT_TABLE

All rows share the encoder, the 80-epoch schedule, seed 42, the precision mode (except `P0`, `R3`, `W3`) and the
evaluation protocol. The checkpoint, the decision threshold and the TTA mode are chosen on validation only. The
ablation table compares D4-TTA at the validation threshold. Rows with prior substitution are also reported as
`<id>:lite`, evaluated without the foundation model.

**Cost (estimated, T4 x2; the log prints the measured epoch time and projection after epoch 1).** These are
inferred from the earlier runs:

| Rows | Time per row |
|---|---|
| B0–B3, P0 | ≈ 3–3.5 h (2.4 min/epoch measured for FP16 SP-UNet) |
| B5, B6, B7, B9, C6, G6, S1, S2 | ≈ 6–6.5 h (ViT-L on half of each batch, plus Lite evaluation) |
| B4, B8 | ≈ 8–8.5 h (ViT-L on every image; A6 measured 5.8 min/epoch) |
| UAV rows | ≈ 1.8× the SAT rows |
| FP32 rows (R3, W3) | ≈ 3.5× FP16; prefer `IMPORT` |

A run that does not fit in a session pauses and resumes in the next one.

**Suggested order** (two Kaggle accounts, one `QUEUE` each):
* account 1: `['B3', 'B6']` → `['V6']` → `['B5', 'B7']` → `['C6', 'S1', 'S2']`;
* account 2: `['B4', 'B0']` → `['V3', 'B8']` → `['B1', 'B2', 'P0']` → `['V0', 'V4', 'B9', 'G6']`.

Then download every `runs/<ID>` folder (the small files are enough) into one folder and run the collect cell, or
`python ablation.py collect --roots <folder> --out ablation`.
'''

CONFIG = r'''
import time
NB_T0 = time.time()          # session start - the queue keeps every run inside the 12 h Kaggle limit
# ================================ CONFIG ================================
QUEUE = ['B3', 'B6']         # experiment ids, run in this order (table above); account 2: ['B4', 'B0']
EPOCHS = 80                  # identical for every row (the SP-UNet results used 80)
BATCH_PER_GPU = 8            # global batch = 16 on T4 x2; use 6 if CUDA runs out of memory
SESSION_HOURS = 11.5         # Kaggle stops sessions at 12 h; the queue stops cleanly before this
EXTRA_ARGS = ''              # appended to every run, e.g. '--grad-ckpt'
IMPORT = {}                  # earlier FP32 SP-UNet checkpoints as the precision reference rows, e.g.
# IMPORT = {'R3': '/kaggle/input/<sat-notebook>/runs/sat_spunet_convnext_t_dinov3_fp32_rerun/best.pth',
#           'W3': '/kaggle/input/<uav-notebook>/runs/sat_spunet_convnext_t_dinov3_fp32_rerun/best.pth'}
QUICK_TEST = False           # True: 1 epoch of QUEUE[0] (+ its evaluation) to measure time and memory
WORK = '/kaggle/working'
PREV = ['/kaggle/input']     # earlier outputs attached as inputs are found here automatically
'''

ENV = r'''
import os, sys, json, glob, shlex, subprocess
import torch

def run(args, env=None):
    """Run a command (argument list, no shell), stream its output into the notebook, raise on failure."""
    args = [str(x) for x in args]
    print('$', ' '.join(shlex.quote(x) for x in args), flush=True)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding='utf-8', errors='replace', env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end='', flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f'command failed with exit code {rc}')

print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.version.cuda)
N_GPU = torch.cuda.device_count()
print('GPUs:', N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
if N_GPU == 0:
    raise SystemExit('No GPU visible: Settings -> Accelerator -> GPU T4 x2')

# timm >= 1.0.20 is needed for the DINOv3 models; checked in a fresh interpreter so an upgrade takes effect
CHECK = ("import sys, timm; have = set(timm.list_pretrained('*dinov3*')); "
         "need = ['convnext_tiny.dinov3_lvd1689m', 'vit_large_patch16_dinov3.sat493m']; "
         "print('timm', timm.__version__); sys.exit(0 if all(n in have for n in need) else 3)")
if subprocess.run([sys.executable, '-c', CHECK]).returncode != 0:
    run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'timm>=1.0.20'])
    if subprocess.run([sys.executable, '-c', CHECK]).returncode != 0:
        raise SystemExit('timm does not provide the DINOv3 models even after upgrading (Internet must be On).')

# download the encoder and the foundation-model weights once (ViT-L: 1.2 GB), so every process finds them cached
run([sys.executable, f'{WORK}/ablation.py', 'weights', '--queue', *QUEUE, f'--extra={EXTRA_ARGS}'])
run([sys.executable, f'{WORK}/ablation.py', 'list'])
'''

RUN = r'''
# Train + evaluate the queued experiments (DDP on all GPUs). Safe to re-run: finished runs are skipped.
cmd = [sys.executable, f'{WORK}/ablation.py', 'run', '--work', WORK, '--queue', *QUEUE, '--epochs', EPOCHS,
       '--batch-size', BATCH_PER_GPU, '--session-hours', SESSION_HOURS, '--t0', NB_T0, f'--extra={EXTRA_ARGS}',
       '--prev', *PREV]
if IMPORT:
    cmd += ['--import'] + [f'{k}={v}' for k, v in IMPORT.items()]
if QUICK_TEST:
    cmd.append('--quick')
t0 = time.time()
run(cmd)
print(f'queue finished in {(time.time() - t0) / 3600:.2f} h')
'''

COLLECT = r'''
# Ablation / comparison tables, paired-bootstrap significance tests, per-region tables and figures
run([sys.executable, f'{WORK}/ablation.py', 'collect', '--roots', f'{WORK}/runs', *PREV, '--out', f'{WORK}/ablation'])
from IPython.display import Markdown, Image, display
for tag in QUEUE:
    f = f'{WORK}/runs/{tag}/results.md'
    if os.path.exists(f):
        display(Markdown(open(f, encoding='utf-8').read()))
for f in sorted(glob.glob(f'{WORK}/ablation/fig_*.png')):
    display(Image(filename=f))
print('download for the paper:', sorted(os.listdir(f'{WORK}/ablation')))
'''

NEXT = r'''
## After the run

* **Continue the queue**: open a new version, *Add Input → Your Work →* this notebook's latest output, keep or extend
  `QUEUE`, *Save & Run All*. Finished runs are not repeated.
* **What to keep for the paper**: `/kaggle/working/ablation/` holds `ablation.md`, `ablation_table.csv` / `.tex`,
  `summary.json`, `per_region_*.csv` and `fig_*.png` / `.pdf`. Each run keeps `results.md` / `results.json`,
  `history.json`, `per_tile_test.csv`, `best.pth` and `vis_test/`.
* **Two accounts**: run different queues, download each `runs/<ID>` folder (the small files are enough), put them in
  one folder and run locally: `python ablation.py collect --roots <folder> --out ablation`.
* **Reading the log**:
  * `no-FM val IoU` is the same network evaluated without the foundation model.
  * `FP32 recomputes N` counts FP16 steps that overflowed and were redone in FP32. With
    `--precision auto` this is harmless. If it happens often, the run switches to FP32 by itself.
* **Single custom run**: `python train_landslide.py --help` lists every option:
  * `--fm none`: SP-UNet;
  * `--fm-sub none`: v1 DPF-Net;
  * `--fm-sub zero`: prior dropout;
  * `--fm-mode distill`: feature distillation only;
  * `--precision fp32`: full FP32.
'''


def main():
    srcs = {s: (HERE / s).read_text(encoding='utf-8') for s in SCRIPTS}
    cells = [md(INTRO.replace('EXPERIMENT_TABLE', experiment_table())), code(CONFIG)]
    cells += [code(f'%%writefile /kaggle/working/{s}\n' + srcs[s]) for s in SCRIPTS]
    cells += [code(ENV), code(RUN), code(COLLECT), md(NEXT)]
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True, 'isGpuEnabled': True,
                       'language': 'python', 'sourceType': 'notebook', 'dataSources': []},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    # self-check: the embedded scripts are byte-identical to the source files
    back = json.loads(OUT.read_text(encoding='utf-8'))
    code_srcs = [''.join(c['source']) for c in back['cells'] if c['cell_type'] == 'code']
    for s in SCRIPTS:
        assert f'%%writefile /kaggle/working/{s}\n' + srcs[s].strip('\n') in code_srcs, s
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)')


if __name__ == '__main__':
    main()
