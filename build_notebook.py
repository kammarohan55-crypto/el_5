#!/usr/bin/env python3
"""Builds CAS_Landslide_StructuralPrior_Kaggle.ipynb from prepare_data.py and train_landslide.py,
so the notebook always contains exactly the tested scripts."""
import json
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'CAS_Landslide_StructuralPrior_Kaggle.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {},
            'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n').splitlines(keepends=True)}


INTRO = r'''
# CAS Landslide segmentation — Structural-Prior UNet (Kaggle, GPU T4 x2)

Trains and evaluates a ConvNeXt-UNet with a **GPU-native spectral-structural prior**, a **boundary head**,
**deep supervision** and **hard-negative pixel contrastive learning** on the CAS Landslide Dataset
(Xu et al., *Sci. Data* 2024; Zenodo 10294997).

* **SAT**: identical public test split as SC-Net (Song et al., *Remote Sens.* 2026; 5,204 train / 2,218 test).
  Benchmarks: Sci-Data DeepLabv3+ IoU 68.14; SC-Net IoU 79.76.
* **UAV**: seeded 7:3 split per region (written to `/kaggle/working/splits/UAV`). Benchmarks: MFFENet 84.38; SC-Net 89.89.

**Before running**
1. *Settings → Accelerator → GPU T4 x2*, *Settings → Internet → On* (to download the pretrained encoder and the SC-Net split CSVs).
2. *Add Input →* dataset **`ngxunninh/cas-landslide`** (or any copy of Zenodo 10294997 with the original region folders).
3. Edit the CONFIG cell, then *Save Version → Save & Run All (Commit)*. A full 80-epoch SAT run takes roughly 3–7 h.
4. Results: `/kaggle/working/runs/<RUN_NAME>/results.md` (also printed at the end of this notebook).
'''

CONFIG = r'''
# ================================ CONFIG ================================
SUBSET = 'SAT'              # 'SAT' (public SC-Net split) or 'UAV' (seeded 7:3 split per region)
RUN_NAME = 'sat_spunet_convnext_t_dinov3'
ENCODER = 'convnext_tiny.dinov3_lvd1689m'   # or 'convnext_tiny.fb_in22k_ft_in1k', 'convnext_small.dinov3_lvd1689m',
                                            #    'regnety_064.ra3_in1k' (SC-Net's backbone family)
FALLBACK_ENCODER = 'convnext_tiny.fb_in22k_ft_in1k'
EPOCHS = 80 if SUBSET == 'SAT' else 40
BATCH_PER_GPU = 8           # global batch = BATCH_PER_GPU x number of GPUs (16 on T4 x2)
SEED = 42
FULL_TRAIN = False          # True: train on train+val with the fixed schedule, evaluate the final EMA weights
EXTRA_ARGS = ''             # ablations, e.g. '--no-hires --no-prior --boundary-weight 0 --contrast-weight 0'
RESUME = ''                 # e.g. '/kaggle/input/<previous-version-output>/runs/<RUN_NAME>/last.pth'
MAX_TRAIN_HOURS = 10.5      # Kaggle stops sessions at 12 h; training checkpoints and stops before this budget
QUICK_TEST = False          # True: 1 epoch on the full data to measure epoch time / memory before the real run
WORK = '/kaggle/working'
'''

ENV = r'''
import os, sys, json, shlex, subprocess, time
import torch

def run(args, env=None):
    """Run a command (list of arguments, no shell) and stream its output into the notebook; raise on failure."""
    args = [str(x) for x in args]
    print('$', ' '.join(shlex.quote(x) for x in args), flush=True)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, **(env or {})})
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
try:
    import timm
except ImportError:
    run([sys.executable, '-m', 'pip', 'install', '-q', 'timm'])
    import timm
print('timm', timm.__version__)
if QUICK_TEST:
    EPOCHS, RUN_NAME = 1, RUN_NAME + '_quick'

# Download the pretrained encoder once (Internet ON) so that both training processes find it in the cache.
try:
    _m = timm.create_model(ENCODER, pretrained=True, features_only=True); del _m
    print('encoder weights OK:', ENCODER)
except Exception as e:
    print(f'!! could not load {ENCODER}: {type(e).__name__}: {e}')
    print(f'!! falling back to {FALLBACK_ENCODER} (recorded in results.json)')
    ENCODER = FALLBACK_ENCODER
    try:
        _m = timm.create_model(ENCODER, pretrained=True, features_only=True); del _m
    except Exception as e2:
        raise SystemExit(f'Cannot download pretrained weights ({e2}). Turn Settings -> Internet -> On.')
'''

PREP = r'''
# Build the split CSVs (finds the CAS region folders below /kaggle/input automatically)
run([sys.executable, f'{WORK}/prepare_data.py', '--subset', SUBSET, '--out', f'{WORK}/splits'])
'''

TRAIN = r'''
# Train on all GPUs with DDP, then evaluate (validation-selected EMA weights, D4-TTA, val-selected threshold)
args = ['--splits', f'{WORK}/splits/{SUBSET}', '--out', f'{WORK}/runs/{RUN_NAME}', '--encoder', ENCODER,
        '--epochs', EPOCHS, '--batch-size', BATCH_PER_GPU, '--seed', SEED, '--max-train-hours', MAX_TRAIN_HOURS]
if FULL_TRAIN:
    args.append('--full-train')
if RESUME:
    args += ['--resume', RESUME]
args += shlex.split(EXTRA_ARGS)
launcher = ([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={N_GPU}']
            if N_GPU > 1 else [sys.executable])
t0 = time.time()
run(launcher + [f'{WORK}/train_landslide.py'] + args,
    env={'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1', 'OMP_NUM_THREADS': '2'})
print(f'finished in {(time.time() - t0) / 3600:.2f} h')
'''

RESULTS = r'''
from IPython.display import Markdown, Image, display
res_dir = f'{WORK}/runs/{RUN_NAME}'
display(Markdown(open(f'{res_dir}/results.md', encoding='utf-8').read()))
vis = sorted(os.listdir(f'{res_dir}/vis_test')) if os.path.isdir(f'{res_dir}/vis_test') else []
for f in vis[:4]:   # RGB | ground truth | errors (TP green, FP red, FN blue)
    display(Image(filename=f'{res_dir}/vis_test/{f}'))
print('files:', sorted(os.listdir(res_dir)))
'''

NEXT = r'''
## Next steps

* **Resume** a run stopped by the time budget: open a new version of this notebook, *Add Input → Your Work →* the
  previous version's output, set `RESUME = '/kaggle/input/.../runs/<RUN_NAME>/last.pth'` (keep all other settings) and
  *Save & Run All*. The best checkpoint and the history are carried over.
* **Ablations for the paper** (one run each; set `EXTRA_ARGS` and a new `RUN_NAME`):

| Row | `EXTRA_ARGS` |
|---|---|
| UNet + PPM baseline | `--no-hires --no-prior --boundary-weight 0 --contrast-weight 0` |
| + high-resolution stem (RGB) | `--no-prior --boundary-weight 0 --contrast-weight 0` |
| + spectral-structural prior | `--boundary-weight 0 --contrast-weight 0` |
| + boundary head | `--contrast-weight 0` |
| + hard-negative contrast (full) | *(empty)* |

  TTA is reported separately in every `results.md` (no-TTA and D4-TTA rows).
* **Pretraining ablation**: `ENCODER = 'convnext_tiny.fb_in22k_ft_in1k'`; **SC-Net backbone**: `'regnety_064.ra3_in1k'`.
* **Seeds**: repeat the full model with `SEED = 1, 2, 3` and report mean ± std.
* **Ensemble** of finished runs (attach their outputs as inputs, run the CONFIG/setup/prepare cells, then in a new cell):
  `run([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={N_GPU}', f'{WORK}/train_landslide.py', '--eval-only', '--splits', f'{WORK}/splits/{SUBSET}', '--out', f'{WORK}/runs/ensemble', '--eval-ckpts', '/kaggle/input/.../best.pth', '/kaggle/input/.../best.pth'], env={'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1'})`
'''


def main():
    prep_src = (HERE / 'prepare_data.py').read_text(encoding='utf-8')
    train_src = (HERE / 'train_landslide.py').read_text(encoding='utf-8')
    cells = [
        md(INTRO),
        code(CONFIG),
        code(ENV),
        code('%%writefile /kaggle/working/prepare_data.py\n' + prep_src),
        code('%%writefile /kaggle/working/train_landslide.py\n' + train_src),
        code(PREP),
        code(TRAIN),
        code(RESULTS),
        md(NEXT),
    ]
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
    srcs = [''.join(c['source']) for c in back['cells'] if c['cell_type'] == 'code']
    assert srcs[2] == '%%writefile /kaggle/working/prepare_data.py\n' + prep_src.strip('\n')
    assert srcs[3] == '%%writefile /kaggle/working/train_landslide.py\n' + train_src.strip('\n')
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)')


if __name__ == '__main__':
    main()
