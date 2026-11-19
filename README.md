# CAS Landslide — DPF-Net: Dual-Prior Fusion Network (Kaggle T4 x2)

Code for landslide segmentation on the CAS Landslide Dataset (Xu et al., *Sci. Data* 2024; Zenodo 10294997) and for
every experiment of the paper, runnable on the free Kaggle GPU (2 × T4).

| File | Purpose |
|---|---|
| `CAS_Landslide_DPFNet_Kaggle.ipynb` | **The notebook to upload to Kaggle.** It embeds the three scripts below and runs a queue of experiments |
| `train_landslide.py` | Model (DPF-Net / SP-UNet / ablations), DDP training, evaluation, results tables |
| `ablation.py` | Experiment registry, session-aware queue (carry forward / resume / import) and the results collector (tables, bootstrap tests, figures) |
| `prepare_data.py` | Finds the CAS region folders and writes the train / val / trainval / test CSVs |
| `build_notebook.py` | Rebuilds the notebook from the scripts after any edit |
| `smoke_test/` | Local CPU tests (unit checks + end-to-end run on synthetic CAS-shaped data) |

## Where we are

Results of the earlier model, **SP-UNet** (structural prior only), on the test split, landslide-class IoU (%):

| Subset | SP-UNet, D4-TTA | Best published | Notes |
|---|---|---|---|
| SAT (public SC-Net test split, 2,218 tiles) | 79.41 | SC-Net 79.76 | SC-Net selected its checkpoint on this test split; we select on validation |
| UAV (our seeded 7:3 split, 4,034 tiles) | 88.49 | SC-Net 89.89 | SC-Net used its own, unpublished split |

Both runs kept improving until the last epoch. Errors are concentrated in a few regions with low recall: SAT Wenchuan
52.0, Lombok 64.6, Palu 66.2; UAV Moxitaidi-1m 77.5, Moxi Town-1m 79.8. The model misses landslides whose appearance
differs from the training majority. That is a recognition problem, not a boundary problem, and it motivates the
semantic prior below.

## Method: DPF-Net

* **Encoder**: ConvNeXt-T pretrained with DINOv3 (`convnext_tiny.dinov3_lvd1689m`), with a UNet decoder (PPM context,
  scSE attention, full-resolution output).
* **Structural prior**: 10 fixed, differentiable descriptors (gray, Sobel, LoG, two DoG bands, local std 7/15,
  local contrast normalisation, ExG vegetation index, saturation), computed in FP32. They feed a high-resolution
  stem at 1/1, 1/2 and 1/4.
* **Semantic prior (new)**: a frozen remote-sensing foundation model, **DINOv3 ViT-L/16 SAT-493M**
  (`vit_large_patch16_dinov3.sat493m`, 303 M parameters, never trained).
  * Its LayerNorm-ed features from blocks 6/12/18/24 are concatenated and passed through a light adapter.
  * They are injected into the encoder features at 1/8, 1/16 and 1/32 by **structure-aware gated fusion**:
    `c' = c + g ⊙ P(f)`, where the gate `g = σ(MLP[c, P(f), S])` sees the CNN features, the semantic features and the
    pooled structural-prior stem.
  * `P` ends in a zero-initialised BatchNorm, so training starts exactly from the CNN-only network.
* **Semantic hard-negative mining (new)**: anchors of the pixel contrastive loss are drawn from three groups, one
  third each:
  * pixels the current prediction gets wrong;
  * *semantic confusers*: background cells most similar (in foundation-feature space, relative to batch prototypes)
    to landslides, and landslide cells most similar to background;
  * random pixels.
* **DPF-Lite (new)**: the foundation model is used **only during training**, as a feature-distillation target for the
  CNN's 1/16 features and for the mining above. Inference cost is identical to SP-UNet's.
* **Other heads**: boundary head (morphological-gradient target), deep supervision at 1/4 and 1/8.
* **Recipe**: AdamW (encoder 1e-4 / decoder 4e-4, wd 0.05), 2-epoch warmup then cosine, EMA 0.9995, FP16 autocast
  with dynamic loss scaling (the frozen ViT also runs in FP16, with an automatic FP32 fallback on overflow), DDP +
  SyncBN, GPU augmentation, 80 epochs.
* **Evaluation**:
  * EMA weights of the best validation epoch;
  * dataset-level pixel counts;
  * D4 TTA, plus optional multi-scale TTA (`--ms-tta`: scales 0.75 / 1 / 1.25), sharing forward passes;
  * the TTA mode and the decision threshold are chosen **on validation only**;
  * per-region metrics, and per-tile counts for paired significance tests.

`--fm none` gives SP-UNet exactly; this was verified bit for bit against the previous version.

## Run on Kaggle

1. kaggle.com → *Create → New Notebook → File → Import Notebook* → upload `CAS_Landslide_DPFNet_Kaggle.ipynb`.
2. *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**.
3. *Add Input*: dataset **`ngxunninh/cas-landslide`**.
4. Set `QUEUE` in the CONFIG cell, then *Save Version → Save & Run All (Commit)*.
5. **Next session**: *Add Input → Your Work →* this notebook (its latest output), then run again. Finished runs are
   carried forward and not repeated; a paused run resumes from `last.pth`. Every run stops cleanly before the 12 h
   limit (`SESSION_HOURS = 11.5`).
6. Reuse the earlier SP-UNet runs as rows **A3 / U3** without retraining: attach their notebook outputs and set
   `IMPORT = {'A3': '/kaggle/input/<...>/runs/<run>/best.pth', 'U3': ...}`. They are re-evaluated with the new
   evaluation code (≈20 min each).

Outputs: `/kaggle/working/runs/<ID>/` (`results.md/json`, `history.json`, `per_tile_test.csv`, `best.pth`,
`vis_test/`) and `/kaggle/working/ablation/` (`ablation.md`, `ablation_table.csv/.tex`, `per_region_*.csv`,
`summary.json`, `fig_*.png/.pdf`).

## Experiments (`python ablation.py list`)

| ID | Subset | Experiment | Answers | Priority |
|---|---|---|---|---|
| A0 | SAT | UNet + PPM baseline | reference | essential |
| A1 | SAT | + spectral-structural prior and high-res stem | does the structural prior help? | recommended |
| A2 | SAT | + boundary head | boundary supervision | recommended |
| A3 | SAT | + error-mined pixel contrast (= SP-UNet) | earlier model; **import, don't retrain** | essential |
| A4 | SAT | + foundation semantic prior, additive fusion | does the semantic prior help? | recommended |
| A5 | SAT | + structure-aware gated fusion | gate vs plain addition | essential |
| A6 | SAT | + semantic hard-negative mining = **DPF-Net** (also MS-TTA) | **main SAT result** | essential |
| A7 | SAT | DPF-Net without structural prior / stem | are the two priors complementary? | essential |
| A8 | SAT | DPF-Lite (foundation model only in training) | accuracy without inference cost | recommended |
| A9 | SAT | DPF-Net with a web-image foundation model (LVD-1689M) | does satellite pretraining matter? (the DINOv3 authors found the web model best on semantic geospatial tasks) | recommended |
| S1, S2 | SAT | DPF-Net with seeds 1 and 2 | mean ± s.d. over 3 seeds | optional |
| U0 | UAV | UNet + PPM baseline | reference | recommended |
| U3 | UAV | SP-UNet | **import, don't retrain** | essential |
| U6 | UAV | DPF-Net (also MS-TTA) | **main UAV result** | essential |
| U8 | UAV | DPF-Lite | | optional |

Every row shares the encoder, 80 epochs, seed 42 and the evaluation; the ablation table compares D4-TTA at the
validation-selected threshold. The collector tests these pairs with a **paired bootstrap over test tiles**: A0→A1,
A1→A2, A2→A3, A3→A4, A4→A5, A5→A6, A3→A5, A3→A6, A0→A6, A7→A6, A8→A6, A3→A8, A9→A6, plus the UAV pairs. It reports
ΔIoU, its 95 % CI and a p-value.

**Cost (estimates; the log prints the measured epoch time after epoch 1).**
* Rows without the foundation model cost the same as your SP-UNet runs.
* With the ViT-L, expect roughly 1.3–1.6× (one frozen forward pass per image, ~623 GFLOPs at 512 px in FP16).
* UAV costs about 1.8× SAT.
* `--fm-size 384` cuts the ViT cost by ~45 % if needed.

**Suggested order**: A6 → A0, A5, A7 → U6 → A4, A8, A9 → A1, A2 → S1, S2, U0, U8; import A3/U3. With two accounts,
split the queue and run `python ablation.py collect --roots <folder with all runs/ID folders> --out ablation` on the
combined outputs (only the small files are needed).

### Protocol notes to state in the paper

* SAT test set = the 2,218 tiles of the public SC-Net split (`datasets/test.csv` of github.com/LiaoChengCRCC/SC-Net),
  so the comparison with SC-Net uses identical test tiles. SC-Net's training script selects its best epoch on that
  test set. This code never uses the test set for any choice: the checkpoint, threshold and TTA mode are chosen on a
  region-stratified 10 % validation split of the training tiles (521 tiles). `--track-test` only logs test curves.
* Metrics come from dataset-level pixel counts (TP/FP/FN/TN summed over all test tiles), landslide class. mIoU / mF1
  are 2-class means, as in Xu et al. (2024).
* Xu et al. (2024) and SegFormer used other random splits, so label those comparisons as indicative. UAV has no public
  split; `prepare_data.py` writes ours (seed 2024), so publish `splits/UAV/*.csv`.
* Report trainable, frozen and inference parameters, FLOPs and throughput (`results.md` gives all of them). The frozen
  ViT-L adds parameters and FLOPs, not trainable capacity. DPF-Lite is the like-for-like efficiency comparison.

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
* DPF-Net fusion with multi-scale TTA, and DPF-Lite;
* pause / resume and eval-only;
* the queue (carry forward, resume in a new session, import, no-time-left);
* the collector (tables, bootstrap, figures);
* 2-process DDP with the foundation model.

## Troubleshooting

* **CUDA out of memory**: `BATCH_PER_GPU = 6`, or `EXTRA_ARGS = '--fm-size 384'` (smaller ViT input) or
  `'--grad-ckpt'` (encoder checkpointing).
* **`FM FP32 fallbacks` in the log**: the frozen ViT overflowed FP16 on some batches and was recomputed in FP32. This
  is harmless but slower; a few per epoch is fine.
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
