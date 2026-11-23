# CAS Landslide — DPF-Net v2: Dual-Prior Fusion Network (Kaggle T4 x2)

Code for landslide segmentation on the CAS Landslide Dataset (Xu et al., *Sci. Data* 2024; Zenodo 10294997) and for
every experiment of the paper, runnable on the free Kaggle GPU (2 × T4).

| File | Purpose |
|---|---|
| `CAS_Landslide_DPFNet_Kaggle.ipynb` | **The notebook to upload to Kaggle.** It embeds the three scripts below and runs a queue of experiments |
| `train_landslide.py` | Model (DPF-Net v2 / v1 / SP-UNet / ablations), DDP training, evaluation, results tables |
| `ablation.py` | Experiment registry, session-aware queue (carry forward / resume / import) and the results collector (tables, bootstrap tests, figures) |
| `prepare_data.py` | Finds the CAS region folders and writes the train / val / trainval / test CSVs |
| `build_notebook.py` | Rebuilds the notebook from the scripts after any edit |
| `DIAGNOSIS.md` | **Why DPF-Net v1 (run A6) scored below SP-UNet**, with the evidence, and how v2 tests each explanation |
| `smoke_test/` | Local CPU tests (unit checks + end-to-end run on synthetic CAS-shaped data) |

## Where we are

Test-split results so far, landslide-class IoU (%):

| Subset | SP-UNet (FP32) | DPF-Net v1 = A6 (FP16) | Best published |
|---|---|---|---|
| SAT (public SC-Net test split, 2,218 tiles) | 79.41 | 77.29 | SC-Net 79.76 (checkpoint chosen on the test split) |
| UAV (our seeded 7:3 split, 4,034 tiles) | 88.49 | – | SC-Net 89.89 (own split) |

A6 is lower in all nine SAT regions, on validation as well. `DIAGNOSIS.md` explains why:
* **Precision confound.** SP-UNet's numbers come from FP32 reruns; the FP16 SP-UNet run diverged at epoch 18.
* **Shortcut.** Under the same FP16 precision, the fused model learned faster early (+1.5 to +3.8 IoU in epochs
  2–17) but finished lower: the decoder leans on the frozen ViT features and the CNN path learns less.
* **Cost.** The ViT-L costs 10× the inference FLOPs.

## Method: DPF-Net v2

* **Encoder**: ConvNeXt-T pretrained with DINOv3 (`convnext_tiny.dinov3_lvd1689m`), with a UNet decoder (PPM context,
  scSE attention, full-resolution output).
* **Structural prior**: 10 fixed, differentiable descriptors (gray, Sobel, LoG, two DoG bands, local std 7/15,
  local contrast normalisation, ExG vegetation index, saturation), computed in FP32. They feed a high-resolution
  stem at 1/1, 1/2 and 1/4. Boundary head and error-mined pixel contrast as in SP-UNet.
* **Semantic prior**: a frozen remote-sensing foundation model, **DINOv3 ViT-L/16 SAT-493M**
  (`vit_large_patch16_dinov3.sat493m`, 303 M parameters, never trained).
  * Features from blocks 6/12/18/24 pass through a light adapter, then **structure-aware gated fusion** injects them at
    1/8, 1/16 and 1/32: `c' = c + g ⊙ P(f)`, with gate `g = σ(MLP[c, P(f), S])` and a zero-initialised `P`.
* **Stochastic prior substitution with prior hallucination (new in v2, `--fm-sub hal`)**:
  * In every training batch, the real prior `f` of a random half of the tiles (`--fm-sub-p 0.5`) is replaced by a
    **hallucinated prior** `f̂ = H(c16, c32)`. A small head predicts it from the CNN's own 1/16 and 1/32 features.
  * `H` is trained by a normalised-MSE loss to the real prior (stop-gradient, weight 0.5), and by the segmentation loss
    through the substituted tiles.
  * This removes the shortcut through the frozen features, because the CNN must solve half of every batch on its own.
    It also halves the ViT cost in training.
  * The **same network runs without the foundation model** at test time (`lite` = always `f̂`, ≈ SP-UNet cost).
  * `--fm-sub zero` is the plain prior-dropout variant, and `--fm-sub none` is v1.
* **Overflow-safe mixed precision (`--precision auto`, new)**:
  * Training runs in FP16 autocast with dynamic loss scaling.
  * The BatchNorm statistics are saved before every step. If the FP16 forward overflows on any rank, they are restored
    and the step is recomputed in FP32.
  * After an epoch with more than 5 % such steps, the run continues in FP32. FP16 can therefore never corrupt the
    model or the EMA.
  * Every inference pass whose FP16 output is not finite is recomputed in FP32.
* **Recipe**: AdamW (encoder 1e-4 / decoder 4e-4, wd 0.05), 2-epoch warmup then cosine, EMA 0.9995, DDP + SyncBN,
  GPU augmentation, 80 epochs.
* **Evaluation**:
  * EMA weights of the best validation epoch;
  * dataset-level pixel counts, D4 TTA (optional multi-scale TTA);
  * the TTA mode and the decision threshold are chosen **on validation only**;
  * per-region metrics, and per-tile counts for paired significance tests;
  * networks trained with substitution are evaluated twice, with the foundation model and without it (`-lite` keys,
    `<id>:lite` rows).

`--fm none` gives SP-UNet exactly, and `--fm-sub none` gives the v1 DPF-Net exactly. Both were verified bit for bit
(initialisation, losses, gradients, outputs) against the previous version.

## Run on Kaggle

1. kaggle.com → *Create → New Notebook → File → Import Notebook* → upload `CAS_Landslide_DPFNet_Kaggle.ipynb`.
2. *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**.
3. *Add Input*: dataset **`ngxunninh/cas-landslide`**.
4. Run once with `QUICK_TEST = True` (1 epoch, ≈25 min) to see the real epoch time. Then set it back to `False`, set
   `QUEUE`, and *Save Version → Save & Run All (Commit)*.
5. **Next session**: *Add Input → Your Work →* this notebook (its latest output), then run again. Finished runs are
   carried forward and not repeated, and a paused run resumes from `last.pth`. Every run stops cleanly before the 12 h
   limit (`SESSION_HOURS = 11.5`).
6. Recommended: attach the notebook versions with your earlier **FP32** SP-UNet runs and set
   `IMPORT = {'R3': '/kaggle/input/<...>/runs/<run>/best.pth', 'W3': ...}`. They become the FP32 reference rows and
   are re-evaluated, not retrained (≈20 min each).

Outputs: `/kaggle/working/runs/<ID>/` (`results.md/json`, `history.json`, `per_tile_test.csv`, `best.pth`,
`vis_test/`) and `/kaggle/working/ablation/` (`ablation.md`, `ablation_table.csv/.tex`, `per_region_*.csv`,
`summary.json`, `fig_*.png/.pdf`).

## Experiments (`python ablation.py list`)

| ID | Subset | Experiment | Answers | Priority |
|---|---|---|---|---|
| B0 | SAT | UNet + PPM baseline | reference for ΔIoU | essential |
| B1 | SAT | + spectral-structural prior and high-res stem | does the structural prior help? | recommended |
| B2 | SAT | + boundary head | boundary supervision | recommended |
| B3 | SAT | + error-mined pixel contrast (= SP-UNet) | baseline of the semantic-prior rows; vs R3 = precision check | essential |
| B4 | SAT | + frozen-FM prior, gated fusion (v1 DPF-Net) | does naive fusion help at matched precision? (shortcut test) | essential |
| B5 | SAT | + prior dropout (`--fm-sub zero`) | does removing the shortcut help? | recommended |
| B6 | SAT | + prior hallucination = **DPF-Net v2** | **main SAT result**; `B6:lite` = without the FM | essential |
| B7 | SAT | B6 + semantic hard-negative mining | does FM-guided mining help? | recommended |
| B8 | SAT | FM feature distillation only | hallucination vs plain distillation (FM-free) | recommended |
| B9 | SAT | B6 with a web-image FM (LVD-1689M) | does satellite pretraining matter? | optional |
| C6 | SAT | B6 without the structural prior / stem | are the two priors complementary? | recommended |
| G6 | SAT | B6 with additive instead of gated fusion | does the structure-aware gate help? | optional |
| P0 | SAT | SP-UNet in plain FP16 | what the overflow guard fixes | optional |
| R3 | SAT | SP-UNet in FP32 | precision reference; **import your earlier run** | import |
| S1, S2 | SAT | B6 with seeds 1 and 2 | mean ± s.d. over 3 seeds | optional |
| V0 | UAV | UNet + PPM baseline | reference | recommended |
| V3 | UAV | SP-UNet | baseline | essential |
| V4 | UAV | v1 DPF-Net | shortcut test on UAV | optional |
| V6 | UAV | DPF-Net v2 | **main UAV result**; `V6:lite` | essential |
| W3 | UAV | SP-UNet in FP32 | precision reference; **import your earlier run** | import |

Every row shares the encoder, 80 epochs, seed 42, the precision mode (`auto`, except P0 / R3 / W3) and the
evaluation. The ablation table compares D4-TTA at the validation-selected threshold.

The collector runs a **paired bootstrap over test tiles** on these pairs, plus the UAV pairs, and reports ΔIoU, its
95 % CI and Holm-adjusted p-values:
- the component chain: B0→B1→B2→B3;
- the semantic-prior rows: B3→B4, B4→B5, B4→B6, B5→B6, B3→B6, B3→B6:lite, B6:lite→B6, B8→B6:lite, B3→B8, B6→B7,
  B9→B6;
- the precision checks: P0→B3 and R3→B3.

**Cost (estimates; the log prints the measured epoch time after epoch 1).**

| Rows | Estimated time |
|---|---|
| B0–B3, P0 | ≈ 3–3.5 h each (FP16 SP-UNet measured 2.4 min/epoch) |
| B5–B7, B9, C6, G6, S1, S2 | ≈ 6–6.5 h (the ViT-L on half of each batch, plus Lite evaluation) |
| B4, B8 | ≈ 8–8.5 h (the ViT-L on every image; A6 measured 5.8 min/epoch) |
| UAV rows | ≈ 1.8× the SAT rows |

`--fm-size 384` cuts the ViT cost by about 45 % if needed, but use it for every FM row or none.

**Suggested order** (two accounts, one `QUEUE` each):
* account 1: `['B3', 'B6']` → `['V6']` → `['B5', 'B7']` → `['C6', 'S1', 'S2']`;
* account 2: `['B4', 'B0']` → `['V3', 'B8']` → `['B1', 'B2', 'P0']` → `['V0', 'V4', 'B9', 'G6']`.

Import R3 / W3 in any session. Then run `python ablation.py collect --roots <folder with all runs/ID folders> --out
ablation` on the combined outputs (only the small files are needed).

### Protocol notes to state in the paper

* **Test split.** The SAT test set is the 2,218 tiles of the public SC-Net split
  (`datasets/test.csv` of github.com/LiaoChengCRCC/SC-Net), so the comparison with SC-Net uses identical test tiles.
  SC-Net's training script selects its best epoch on that test set. This code never uses the test set for any
  choice: the checkpoint, threshold and TTA mode are chosen on a region-stratified 10 % validation split of the
  training tiles (521 tiles).
* **Metrics.** Metrics come from dataset-level pixel counts (TP/FP/FN/TN summed over all test tiles), landslide
  class. mIoU / mF1 are 2-class means, as in Xu et al. (2024).
* **Other splits.** Xu et al. (2024) and SegFormer used other random splits. UAV has no public split, so
  `prepare_data.py` writes ours (seed 2024); publish `splits/UAV/*.csv`.
* **Cost reporting.** Report parameters, FLOPs and throughput for both inference variants (`results.md` gives them).
  The `:lite` rows are the like-for-like efficiency comparison with SP-UNet and SC-Net.
* **Precision.** State the training precision of every row. All ablation rows use the same mode; the FP32 reference
  (R3) quantifies its effect.

## Local smoke test (CPU, no Kaggle needed)

```
python -m venv .venv_smoke --system-site-packages
.venv_smoke\Scripts\python -m pip install timm "torchvision<0.29" matplotlib
.venv_smoke\Scripts\python smoke_test\unit_checks.py
.venv_smoke\Scripts\python smoke_test\run_smoke.py
.venv_smoke\Scripts\python smoke_test\check_py311_fstrings.py
```

Pin `torchvision` to the release that matches your installed torch (0.28 for torch 2.13), so pip does not replace
torch. The smoke test uses a tiny random DINOv3 ViT (5.5 M parameters) in place of the ViT-L, so nothing large is
downloaded. It checks:

* split sizes and SP-UNet training, resume, ensemble, full-train, ablations, no-TTA, time guard;
* the precision guard, which recomputes overflowing steps, switches to FP32 and saves its state for resume;
* DPF-Net v2:
  * evaluated with and without the foundation model;
  * prior dropout;
  * v1 fusion;
  * distillation;
  * pause / resume and eval-only reproducibility;
* the queue (carry forward, resume in a new session, import, no-time-left);
* the collector (`:lite` rows, bootstrap, figures);
* 2-process DDP with the foundation model and forced FP32 recomputes.

## Troubleshooting

* **CUDA out of memory**: `BATCH_PER_GPU = 6`, or `EXTRA_ARGS = '--fm-size 384'` (smaller ViT input) or
  `'--grad-ckpt'` (encoder checkpointing).
* **`FP32 recomputes N` in the log**: N training steps overflowed FP16 and were redone in FP32. This is harmless. If
  more than 5 % of an epoch's steps overflow, the run continues in FP32 (slower) and says so.
* **`FM FP32 fallbacks` in the log**: the frozen ViT overflowed FP16 on some batches and was recomputed in FP32. This
  is harmless.
* **Training hangs at start with 2 GPUs**: the notebook already sets `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.
* **`timm does not provide the DINOv3 models`**: Internet must be On (the notebook upgrades timm to ≥ 1.0.20).
* **`Could not find the SAT region folders`**: the dataset is not attached, or its folders differ. Each region
  folder must contain `img/` and `mask/`.

## Licences and citations

Cite:
* Xu et al. 2024, the CAS Landslide Dataset (CC BY-NC 4.0, non-commercial);
* Song et al. 2026, the SC-Net split;
* DINOv3 (Siméoni et al. 2025), ConvNeXt (Liu et al. 2022) and ViT (Dosovitskiy et al. 2021).

Both DINOv3 checkpoints (ConvNeXt-T LVD-1689M and ViT-L SAT-493M) are released under Meta's DINOv3 License. Models
trained from them, and any redistribution of their weights, must follow its terms.
