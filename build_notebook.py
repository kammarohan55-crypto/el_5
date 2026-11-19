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
        lines.append(f'| `{e["id"]}` | {e["subset"]} | {e["title"]} | {prio[e["priority"]]} |')
    return '\n'.join(lines)


INTRO = r'''
# CAS Landslide segmentation — DPF-Net experiments (Kaggle, GPU T4 x2)

**DPF-Net (Dual-Prior Fusion Network)**: a ConvNeXt-T (DINOv3) UNet that combines

* a **spectral-structural prior**: 10 differentiable descriptors that feed a high-resolution stem;
* a **semantic prior**: multi-layer features of a **frozen satellite foundation model** (DINOv3 ViT-L/16, SAT-493M),
  injected into the 1/8, 1/16 and 1/32 encoder features by **structure-aware gated fusion**;
* **semantic hard-negative mining**: anchors of the pixel-contrastive loss chosen both by the current prediction
  errors and by foundation-model similarity to landslides (bare soil, river beds, roads, ...);
* **DPF-Lite**: the same foundation model used only during training (feature distillation + mining), so it adds no
  inference cost.

`--fm none` reproduces the earlier SP-UNet bit for bit.

**One notebook, all experiments.** Each *Save & Run All* trains the experiments in `QUEUE`, in order, until the 12 h
session is nearly used up. Every run checkpoints. To continue in a new session, attach this notebook's previous
output as an input and run again: finished runs are carried forward, an unfinished run resumes from its checkpoint,
and the tables and figures are rebuilt from everything found.

### Before running
1. *Settings → Accelerator → GPU T4 x2*, *Settings → Internet → On*.
2. *Add Input →* dataset **`ngxunninh/cas-landslide`**.
3. From the 2nd session on: *Add Input → Your Work →* this notebook (its latest output).
4. Optional, to reuse your earlier SP-UNet runs as rows A3 / U3 without retraining: *Add Input →* the notebook
   versions that produced them, then set `IMPORT` in CONFIG. They are re-evaluated (≈20 min each), not retrained.
5. Edit CONFIG → *Save Version → Save & Run All (Commit)*.

### Experiments
EXPERIMENT_TABLE

All rows share the encoder, the 80-epoch schedule, seed 42 and the evaluation protocol. The protocol: the
checkpoint, the decision threshold and the TTA mode are chosen on validation only; the ablation table compares D4-TTA
at the validation threshold.

**Cost.** Runs without a foundation model (A0–A3, U0, U3) take as long as your earlier SP-UNet runs. Runs with the
ViT-L foundation model take about 1.3–1.6× longer, because they add one frozen forward pass per image. UAV runs take
about 1.8× longer than SAT runs. After epoch 1 the log prints the measured epoch time and the projected total. A run
that does not fit in a session pauses and resumes in the next one.

Suggested order: `A6` (main SAT result) → `A0`, `A5`, `A7` → `U6` (main UAV result) → `A4`, `A8`, `A9` → `A1`,
`A2` → `S1`, `S2`, `U0`, `U8`. Import `A3` / `U3` from your SP-UNet runs. With two Kaggle accounts, give each a
different `QUEUE`, then run the collect cell on all outputs.
'''

CONFIG = r'''
import time
NB_T0 = time.time()          # session start - the queue keeps every run inside the 12 h Kaggle limit
# ================================ CONFIG ================================
QUEUE = ['A6']               # experiment ids, run in this order (table above); e.g. ['A6', 'A0', 'A5']
EPOCHS = 80                  # identical for every row (the SP-UNet results used 80)
BATCH_PER_GPU = 8            # global batch = 16 on T4 x2; use 6 if CUDA runs out of memory
SESSION_HOURS = 11.5         # Kaggle stops sessions at 12 h; the queue stops cleanly before this
EXTRA_ARGS = ''              # appended to every run, e.g. '--grad-ckpt'
IMPORT = {}                  # reuse finished SP-UNet checkpoints instead of retraining, e.g.
# IMPORT = {'A3': '/kaggle/input/<sat-notebook>/runs/sat_spunet_convnext_t_dinov3_fp32_rerun/best.pth',
#           'U3': '/kaggle/input/<uav-notebook>/runs/sat_spunet_convnext_t_dinov3_fp32_rerun/best.pth'}
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
* **Single custom run**: `python train_landslide.py --help` lists every option. `--fm none` gives SP-UNet,
  `--fm-mode distill` gives DPF-Lite, and `--ms-tta` adds multi-scale TTA (chosen on validation).
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
