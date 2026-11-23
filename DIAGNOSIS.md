# Why DPF-Net v1 (run A6) scored below SP-UNet, and what v2 changes

Sources:
- the full log of the Kaggle run **v3-cas** (A6, 80 epochs, 8.3 h);
- the SP-UNet logs **v1-landslide** (FP16, SAT) and **uav-v1** (UAV);
- the result files `sat-results.md` / `uav-results.md` / `results.md`;
- a local measurement of the pretrained encoder.

All IoU values are landslide-class IoU in %.

## 1. The numbers

| | SP-UNet (FP32 rerun) | DPF-Net v1 = A6 (FP16) | Δ |
|---|---|---|---|
| Validation, no TTA, t = 0.5 (selection metric) | 78.62 | 76.87 | −1.75 |
| Validation, D4-TTA, t = 0.5 | 79.85 | 77.85 | −2.00 |
| Test, D4-TTA, t = t* (main) | 79.41 | 77.29 | −2.12 |
| Best epoch | 80 (last) | 80 (last) | |

Per-region test IoU (main setting):

| Region | SP-UNet | A6 | Δ |
|---|---|---|---|
| Hokkaido Iburi-Tobu | 73.03 | 71.73 | −1.30 |
| Lombok | 64.56 | 59.22 | −5.34 |
| Longxi River (SAT) | 83.67 | 80.46 | −3.21 |
| Mengdong | 84.85 | 82.99 | −1.86 |
| Moxitaidi (SAT) | 86.83 | 84.90 | −1.93 |
| Palu | 66.20 | 65.15 | −1.05 |
| Tiburon Peninsula (Planet) | 78.01 | 76.77 | −1.24 |
| Tiburon Peninsula (Sentinel) | 73.00 | 71.00 | −2.00 |
| Wenchuan | 51.96 | 51.83 | −0.13 |

A6 is lower in **all nine regions**, on validation as well as on test. That rules out these explanations:
- **A test-set fluke.** The drop also shows on validation.
- **A sensor-specific mismatch of the satellite foundation model** (for example, only on the 10 m Sentinel tiles). The
  drop shows in every region.

What remains is a global training effect. The A6 run itself was clean:
- no NaN;
- no foundation-model FP32 fallbacks;
- the normal one AMP-skipped step about every 7 epochs (GradScaler growth tests);
- a smooth, monotonic validation curve.

## 2. Finding 1 (established): the comparison was confounded by training precision

- **SP-UNet's 79.41 (SAT) and 88.49 (UAV) are FP32 runs.**
  - Both are named `..._fp32_rerun`.
  - The UAV log prints `AMP False`.
  - The SAT rerun measured 18.5 img/s against 54.9 img/s for the FP16 run with the same code and GPU. That 3× drop
    is FP32 on a T4.
- **The same SP-UNet in FP16 diverged** (v1-landslide log):
  - epoch 18: 67 skipped steps and validation IoU 0.00;
  - epoch 19: 90 skipped steps;
  - epochs 20–80: every step non-finite.
  - The run fell back to epoch 17: test IoU 67.15.
- **A6 trained in FP16.** Every other planned ablation row would also have run in FP16, while A3 / U3 were meant to
  be the imported FP32 runs. So the v1 ablation compared FP16 rows against an FP32 baseline.

**Why this architecture is FP16-fragile.**
- **Outlier activations.** The DINOv3-distilled ConvNeXt-T has very large activations. Measured on a natural
  512 × 512 image with the pretrained weights (`tools/activation_range.py`):
  - the output of the 1/16 stage reaches |a| = 111 against a mean |a| of 0.69, a 160× outlier;
  - its layer-scale γ reaches 23.4.
  - DINOv3 models are widely reported to overflow FP16 (BF16 / FP32 is recommended), and a T4 has no BF16 tensor
    cores.
- **Weight growth before BatchNorm.** Conv→BatchNorm layers trained with AdamW let their weight norm grow freely,
  because the output is scale-invariant. In FP16 the pre-BatchNorm sums can then exceed 65 504.
- **No protection in the v1 loop.** GradScaler only skips steps with overflowing *gradients*. An overflowing
  *forward* had already written inf / NaN into the BatchNorm running statistics, and the EMA copied them. That
  matches the observed pattern exactly: validation IoU 0.00 first, then every step failing.

**Effect on A6.** A6 itself showed no overflow, so FP16 alone probably does not explain its −2.1. The conclusion is
narrower: the v1 numbers cannot show what the foundation-model prior does, because the reference differs in
precision.

## 3. Finding 2 (most likely): the frozen-feature fusion is a shortcut

Under the **same** FP16 precision, A6 was ahead of the SP-UNet FP16 run (v1-landslide) at every epoch before the
latter diverged:

| Epoch | 1 | 2 | 5 | 10 | 15 | 17 |
|---|---|---|---|---|---|---|
| SP-UNet, FP16 | 52.68 | 55.35 | 60.79 | 64.17 | 65.94 | 66.53 |
| A6, FP16 | 52.99 | 59.11 | 63.09 | 65.73 | 67.87 | 68.35 |
| Δ | +0.31 | +3.76 | +2.30 | +1.56 | +1.93 | +1.82 |

Training segmentation loss at epoch 17: 0.352 for A6 against 0.385 for SP-UNet. So the semantic prior is informative,
and the fused model fits and generalises faster early. Yet it finished 1.75 lower on validation than the FP32
SP-UNet.

**The mechanism.** "Fast early, lower final" is the signature of shortcut learning through a strong, frozen input:
- **Shortcut.** The frozen ViT-L features (493 M satellite images) are almost linearly separable for
  landslide / background at 1/16 resolution. The decoder therefore reduces the loss through that path first.
- **Less signal for the CNN.** The residual error, and with it the gradient reaching the trainable encoder and the
  high-resolution stem, shrinks. The CNN path learns fewer of the fine-scale features that boundaries and small
  landslides need.
- **A capped prediction.** The output is then bounded by coarse (16 px) frozen semantics.

This is the multi-modal training failure described by:
- Wang et al. (CVPR 2020), "What makes training multi-modal classification networks hard?": the multi-modal network
  underperforms the best uni-modal one.
- Peng et al. (CVPR 2022, OGM-GE) and Huang et al. (ICML 2022), on modality competition.

Its standard remedies are:
- dropping the dominant modality during training (ModDrop; Neverova et al., TPAMI 2016);
- learning to hallucinate it from the other modality (Hoffman et al., CVPR 2016).

This finding is an inference from the curves, not yet proven at matched precision. Rows `B3` vs `B4` of the new
ablation test it directly.

## 4. Smaller findings

- **Multi-scale TTA hurts** on CAS-SAT: −0.98 on test, −0.94 on validation. Downscaling to 0.75 erases small
  landslides. The validation protocol correctly rejected it, so it is not used by default in v2.
- **Semantic hard-negative mining** (error+fm) was untested in isolation. A6 changed fusion and mining at once. In v2
  it is an optional row (`B7`), not part of the default method.
- **Cost.** The ViT-L forward was about 60 % of A6's training step (5.8 against 2.4 min/epoch) and 10× the inference
  FLOPs (812 against 80 GFLOPs; 10.4 against 54.9 img/s on a T4). Reviewers will ask about this even if the
  accuracy improves.
- **Recall-limited regions.** All runs are weakest on Wenchuan, Lombok and Palu, with recall well below precision.
  This is a target for future work.

## 5. What v2 changes, and how each change is tested

| Change | Addresses | Tested by |
|---|---|---|
| `--precision auto`: an overflowing FP16 forward is undone (BatchNorm statistics restored) and recomputed in FP32; frequent overflows switch the run to FP32 | Finding 1 | `R3` (FP32 reference) vs `B3`; `P0` (plain FP16) vs `B3` |
| All rows retrained in the same precision; no FP32 imports in the ablation chain | Finding 1 | the whole table |
| **Stochastic prior substitution**: the real prior of a random half of each batch is replaced by a **hallucinated prior** that a small head predicts from the CNN's own 1/16 and 1/32 features (normalised-MSE loss to the real prior, plus the segmentation loss through the substituted samples) | Finding 2, cost | `B4` (v1 fusion) vs `B5` (prior dropout) vs `B6` (hallucination) |
| The same network evaluated without the foundation model (`:lite`), ≈ SP-UNet cost | cost | `B6:lite` vs `B3` (SP-UNet) and vs `B8` (plain feature distillation) |
| The ViT-L runs on half of each training batch only | cost | train hours column |
| Semantic mining optional | §4 | `B6` vs `B7` |

**How to read the results.**
- If **R3 ≈ B3** (within about 0.3), `--precision auto` matches FP32 and every comparison is clean.
- If **B4 < B3**, the shortcut (Finding 2) is confirmed at matched precision.
- If **B4 ≥ B3**, A6's drop came mainly from precision and seed, and the prior helps as it is.
- In both cases **B6** is the method row:
  - **B6 vs B4** shows what substitution adds;
  - **B6:lite** shows how much of the gain survives without the foundation model at inference.
