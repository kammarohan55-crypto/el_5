# CAS Landslide — Structural-Prior UNet (Kaggle T4 x2)

Code to beat the CAS Landslide benchmarks on the free Kaggle GPU (2 × T4).

| File | Purpose |
|---|---|
| `CAS_Landslide_StructuralPrior_Kaggle.ipynb` | **The notebook to upload to Kaggle** (contains both scripts below) |
| `prepare_data.py` | Finds the CAS region folders, writes train / val / trainval / test CSVs |
| `train_landslide.py` | Model, DDP training, evaluation, results tables |
| `build_notebook.py` | Rebuilds the notebook from the two scripts after any edit |
| `smoke_test/` | Local CPU tests (synthetic CAS-shaped data) |

## Targets

| Subset | Benchmark (paper, split) | IoU (landslide) |
|---|---|---|
| SAT | DeepLabv3+, Xu et al. 2024 *Sci. Data* (own random 7:3 split) | 68.14 |
| SAT | SegFormer 2025 (own split) | 79.30 |
| SAT | **SC-Net**, Song et al. 2026 *Remote Sens.* (**same public test split as ours**) | **79.76** |
| UAV | MFFENet, Xu et al. 2024 (own split) | 84.38 |
| UAV | **SC-Net** (own split) | **89.89** |

Xu et al.'s "F1 score" (89.68 for SAT DeepLabv3+) is the mean of the landslide and background F1; the landslide-class
F1 is 81.05. `results.md` reports both (`F1` and `mF1`) so every number can be compared like-for-like.

## Method (what is new compared with SC-Net)

* **Encoder**: ConvNeXt-T pretrained with DINOv3 (`convnext_tiny.dinov3_lvd1689m`, ~28 M parameters, similar to SC-Net's 30 M).
* **GPU-native spectral-structural prior**: 10 fixed descriptors computed on the GPU in FP32 (gray, Sobel magnitude,
  LoG, two DoG bands, local std 7×7 / 15×15, local contrast normalisation, ExG vegetation index, saturation) →
  BatchNorm → high-resolution stem (1/1, 1/2, 1/4) fused into the decoder. SC-Net computes its 10 descriptors on the CPU
  (28.6 % of its inference time, a limitation stated in its paper); here they are a few GPU convolutions inside the
  forward pass (no CPU preprocessing at all) and add a vegetation-loss cue. Measure the FPS reported in `results.md`
  with and without `--no-prior` for the paper's efficiency table.
* **Decoder**: UNet with pyramid pooling (PPM) context and scSE attention, full-resolution output.
* **Boundary head** supervised with a morphological-gradient boundary map.
* **Hard-negative pixel contrastive loss** (supervised contrast on 1/4-res embeddings, half of the anchors are
  currently-misclassified pixels) — SC-Net's stated future work (it used random negatives).
* **Recipe**: AdamW (enc 1e-4 / dec 4e-4, wd 0.05), 2-epoch warmup + cosine, EMA 0.9995, FP16 autocast with dynamic
  loss scaling (no manual scale hacks), gradient clipping, D4 + affine + photometric augmentation on the GPU, DDP on both T4s.
* **Evaluation**: EMA weights, D4 test-time augmentation, decision threshold chosen on the validation split only,
  per-region metrics, parameters / FLOPs / FPS.

## Run on Kaggle

1. kaggle.com → *Create → New Notebook → File → Import Notebook* → upload `CAS_Landslide_StructuralPrior_Kaggle.ipynb`.
2. *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**.
3. *Add Input*: dataset **`ngxunninh/cas-landslide`** (Zenodo 10294997 layout: `<Region>/img`, `<Region>/mask`).
4. Optional first check: set `QUICK_TEST = True`, run all, read the printed epoch time and the GPU memory
   (`nvidia-smi` in a cell), then set it back to `False`.
5. *Save Version → Save & Run All (Commit)*. The run trains, evaluates and writes
   `/kaggle/working/runs/<RUN_NAME>/` (`results.md`, `results.json`, `history.json`, `best.pth`, `last.pth`, `vis_test/`).

Expected cost (an estimate, not yet measured on Kaggle): ~2–5 min per SAT epoch on T4 x2. The script prints the measured
epoch time and the projected total after the first epoch → 80 epochs ≈ 3–7 h of the ~30 h weekly GPU quota.
Model size of the default configuration (measured): 35.0 M parameters incl. training-only heads, 40.2 GMACs at 512×512. If a run would exceed `MAX_TRAIN_HOURS` it stops,
saves `last.pth` and still evaluates the best checkpoint; continue it with `RESUME` in a new version (see notebook).

## Experiments for the paper

Each row is one run of the notebook (set `RUN_NAME`, `EXTRA_ARGS`, `ENCODER`, `SEED`).

| Experiment | Settings |
|---|---|
| A0 UNet + PPM baseline | `EXTRA_ARGS='--no-hires --no-prior --boundary-weight 0 --contrast-weight 0'` |
| A1 + high-res stem (RGB) | `EXTRA_ARGS='--no-prior --boundary-weight 0 --contrast-weight 0'` |
| A2 + spectral-structural prior | `EXTRA_ARGS='--boundary-weight 0 --contrast-weight 0'` |
| A3 + boundary head | `EXTRA_ARGS='--contrast-weight 0'` |
| A4 full model | `EXTRA_ARGS=''` |
| TTA effect | read the no-TTA vs D4-TTA rows of any `results.md` |
| Pretraining | `ENCODER='convnext_tiny.fb_in22k_ft_in1k'` (ImageNet-22k) vs DINOv3 |
| Same backbone family as SC-Net | `ENCODER='regnety_064.ra3_in1k'` |
| Larger model (if needed) | `ENCODER='convnext_small.dinov3_lvd1689m'` (+ `--grad-ckpt` or `BATCH_PER_GPU=6` if out of memory) |
| Seeds | full model with `SEED=1,2,3`, report mean ± std |
| Final numbers | `FULL_TRAIN=True` (train+val, fixed schedule, final EMA, threshold 0.5) |
| UAV | `SUBSET='UAV'`, `EPOCHS=40` (publish `splits/UAV/*.csv`) |
| Ensemble (optional, report separately) | `--eval-only --eval-ckpts a/best.pth b/best.pth ...` |

### Protocol notes to state in the paper

* SAT test set = the 2,218 tiles of the public SC-Net split (`datasets/test.csv` in github.com/LiaoChengCRCC/SC-Net),
  so the comparison with SC-Net is on identical test tiles. SC-Net's `train.py` selects the best epoch on that test set;
  this code never touches the test set for selection (validation = region-stratified 10 % of the training tiles,
  521 tiles; `--track-test` only logs test curves for reference).
* Metrics are computed from dataset-level pixel counts (TP/FP/FN/TN summed over all test tiles), landslide class,
  threshold t* selected on validation; mIoU / mF1 are 2-class means as in Xu et al. (2024).
* Xu et al. (2024) and SegFormer used different random 7:3 splits; label those comparisons as indicative.

## Local smoke test (CPU, no Kaggle needed)

```
python -m venv .venv_smoke --system-site-packages
.venv_smoke\Scripts\python -m pip install timm "torchvision<0.29"
.venv_smoke\Scripts\python smoke_test\unit_checks.py --download
.venv_smoke\Scripts\python smoke_test\run_smoke.py
```

(Pin `torchvision` to the release matching your installed torch — 0.28 for torch 2.13 — so pip does not replace torch.)
`run_smoke.py` builds a synthetic CAS-shaped dataset with all 7,422 SC-Net tile names and checks: split sizes
(4,683 / 521 / 2,218), UAV split, training, resume, ensemble, `--full-train`, ablations, no-TTA labelling,
`--track-test`, the time guard and 2-process DDP (on Windows the two ranks are started directly because Windows
PyTorch builds lack libuv for `torchrun`; Kaggle uses `torchrun --standalone`).

## Troubleshooting

* **CUDA out of memory**: `BATCH_PER_GPU = 6` (or 4), or `EXTRA_ARGS='--grad-ckpt'`.
* **Training hangs at start with 2 GPUs**: the notebook already sets `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`; as a fallback run
  single-GPU by replacing the launcher with `[sys.executable]`.
* **`Could not find the SAT region folders`**: the dataset is not attached, or its folder names differ; each region folder must
  contain `img/` and `mask/`.
* **Encoder download fails**: Internet must be On; the notebook falls back to `convnext_tiny.fb_in22k_ft_in1k` and records it.

## Licences and citations

Cite Xu et al. 2024 (CAS Landslide Dataset, CC BY-NC 4.0), Song et al. 2026 (SC-Net split), DINOv3 (Siméoni et al. 2025)
and ConvNeXt (Liu et al. 2022). DINOv3 weights are released under Meta's DINOv3 licence; models fine-tuned from them must follow
its terms. Use `ENCODER='convnext_tiny.fb_in22k_ft_in1k'` if that licence is a problem.
