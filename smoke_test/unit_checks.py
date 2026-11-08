#!/usr/bin/env python3
"""Unit checks for train_landslide.py (CPU). Run: python smoke_test/unit_checks.py [--download]"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_landslide as T  # noqa: E402

PASSED = []


def check(name, cond, detail=''):
    if not cond:
        raise AssertionError(f'FAILED: {name} {detail}')
    PASSED.append(name)
    print(f'  ok  {name} {detail}')


def test_metrics():
    m = T.metrics_from_counts(50, 10, 20, 920)
    check('precision', abs(m['precision'] - 50 / 60) < 1e-12)
    check('recall', abs(m['recall'] - 50 / 70) < 1e-12)
    check('f1', abs(m['f1'] - 100 / 130) < 1e-12)
    check('iou', abs(m['iou'] - 50 / 80) < 1e-12)
    check('iou_bg', abs(m['iou_bg'] - 920 / 950) < 1e-12)
    check('oa', abs(m['oa'] - 970 / 1000) < 1e-12)
    pe = (60 * 70 + 940 * 930) / 1e6
    check('kappa', abs(m['kappa'] - (0.97 - pe) / (1 - pe)) < 1e-12)
    check('mf1', abs(m['mf1'] - 0.5 * (100 / 130 + 1840 / 1870)) < 1e-12)
    # Xu et al. 2024 SAT DeepLabv3+: P 74.275, R 89.187 -> IoU 68.137, landslide F1 81.05
    P, R = 0.74275, 0.89187
    check('nature IoU consistency', abs(1 / (1 / P + 1 / R - 1) - 0.68137) < 5e-5)
    check('nature landslide F1', abs(2 * P * R / (P + R) - 0.8105) < 5e-4)
    z = T.metrics_from_counts(0, 0, 0, 100)
    check('empty foreground safe', z['iou'] == 0.0 and z['oa'] == 1.0)


def test_tta_identity():
    class PixelModel(nn.Module):
        def forward(self, x):
            return {'seg': torch.logit(x[:, :1].clamp(1e-4, 1 - 1e-4))}

    x = torch.rand(3, 3, 32, 32)
    p = T.predict_probs(PixelModel(), x, tta=True, amp=False, channels_last=False)
    check('D4 TTA inverse = identity', torch.allclose(p, x[:, :1].clamp(1e-4, 1 - 1e-4), atol=1e-5),
          f'max err {(p - x[:, :1].clamp(1e-4, 1 - 1e-4)).abs().max():.2e}')

    class PositionModel(nn.Module):  # output depends on absolute position -> TTA must average 8 transforms
        def forward(self, x):
            h = x.shape[-1]
            ramp = torch.linspace(-2, 2, h).view(1, 1, 1, h).expand(x.shape[0], 1, h, h)
            return {'seg': ramp.clone()}

    p = T.predict_probs(PositionModel(), torch.rand(1, 3, 16, 16), tta=True, amp=False, channels_last=False)
    check('TTA output symmetric', torch.allclose(p, torch.rot90(p, 1, dims=[2, 3]), atol=1e-6))


def test_augment_alignment():
    torch.manual_seed(0)
    a = SimpleNamespace(aug_affine_p=1.0, aug_rotate=30.0, aug_scale_min=0.75, aug_scale_max=1.33,
                        aug_translate=0.1, aug_color_p=0.0, aug_brightness=0.2, aug_contrast=0.2,
                        aug_saturation=0.2, aug_channel_gain=0.05, aug_gamma_p=0.0, aug_noise_p=0.0,
                        aug_noise_std=0.02)
    y = torch.zeros(4, 1, 128, 128)
    y[:, :, 30:80, 40:100] = 1
    y[:, :, 90:120, 10:30] = 1
    x = torch.cat([y, torch.rand(4, 2, 128, 128)], 1)
    xa, ya = T.gpu_augment(x.clone(), y.clone(), a)
    agree = ((xa[:, :1] > 0.5) == (ya > 0.5)).float().mean().item()
    check('affine keeps image/mask aligned', agree > 0.98, f'agreement {agree:.4f}')
    check('mask stays binary', set(torch.unique(ya).tolist()) <= {0.0, 1.0})
    check('augmentation changed geometry', not torch.equal(ya, y))
    a.aug_affine_p, a.aug_color_p, a.aug_gamma_p, a.aug_noise_p = 0.0, 1.0, 1.0, 1.0
    xa, ya = T.gpu_augment(x.clone(), y.clone(), a)
    check('photometric keeps mask', torch.equal(ya, y))
    check('photometric output in [0,1]', float(xa.min()) >= 0.0 and float(xa.max()) <= 1.0)


def test_prior_and_losses():
    prior = T.SpectralStructuralPrior()
    out = prior(torch.rand(2, 3, 64, 64))
    check('prior shape', tuple(out.shape) == (2, 10, 64, 64))
    check('prior finite', bool(torch.isfinite(out).all()))
    m = torch.zeros(1, 1, 16, 16)
    m[:, :, 4:12, 4:12] = 1
    b = T.boundary_target(m)
    check('boundary ring', b[0, 0, 8, 8] == 0 and b[0, 0, 4, 8] == 1 and b[0, 0, 3, 8] == 1 and b[0, 0, 0, 0] == 0)
    check('boundary of empty mask', float(T.boundary_target(torch.zeros(1, 1, 8, 8)).sum()) == 0.0)
    emb = torch.randn(2, 16, 8, 8, requires_grad=True)
    logits = torch.randn(2, 1, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.6).float()
    l = T.pixel_contrast_loss(emb, logits, mask, n_per_class=32, temperature=0.1, pos_thr=0.5, neg_thr=0.5)
    l.backward()
    check('contrast loss finite & positive', bool(torch.isfinite(l)) and float(l) > 0, f'{float(l):.4f}')
    check('contrast gradient flows', emb.grad is not None and float(emb.grad.abs().sum()) > 0)
    emb2 = torch.randn(2, 16, 8, 8, requires_grad=True)
    l0 = T.pixel_contrast_loss(emb2, logits, torch.zeros(2, 1, 32, 32))
    l0.backward()
    check('contrast on empty mask = 0, graph-connected', float(l0) == 0.0 and emb2.grad is not None)
    d = T.bce_dice(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    check('bce_dice finite on empty mask', bool(torch.isfinite(d)))


def test_lr():
    f = [T.lr_factor(s, 1000, 100, 0.01) for s in (0, 50, 100, 550, 1000, 2000)]
    check('warmup starts at 1%', abs(f[0] - 0.01) < 1e-12)
    check('warmup mid', abs(f[1] - (0.01 + 0.99 * 0.5)) < 1e-12)
    check('peak after warmup', abs(f[2] - 1.0) < 1e-12)
    check('cosine middle', abs(f[3] - (0.01 + 0.99 * 0.5)) < 1e-9)
    check('end at min ratio', abs(f[4] - 0.01) < 1e-12 and abs(f[5] - 0.01) < 1e-12)


def test_encoders_and_model(download):
    for name in ('convnext_atto', 'resnet18', 'regnety_002', 'efficientnet_b0', 'convnext_tiny'):
        enc = T.Encoder(name, pretrained=False)
        x = torch.rand(1, 3, 64, 64)
        feats = enc(x)
        shapes = [tuple(f.shape) for f in feats]
        ok = all(f.shape[-1] == 64 // r and f.shape[1] == c for f, r, c in zip(feats, (4, 8, 16, 32), enc.channels))
        check(f'encoder {name} strides 4/8/16/32', ok, str(shapes))
    a = T.parse_args(['--splits', 'x', '--out', 'y', '--encoder', 'convnext_tiny', '--no-pretrained'])
    model = T.build_model(a, pretrained=False)
    model.train()
    out = model(torch.rand(2, 3, 64, 64))
    check('train outputs', set(out) == {'seg', 'aux4', 'aux8', 'bnd', 'emb'} and tuple(out['seg'].shape) == (2, 1, 64, 64),
          str({k: tuple(v.shape) for k, v in out.items()}))
    loss, _ = T.compute_loss(out, (torch.rand(2, 1, 64, 64) > 0.7).float(), a)
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    check('every parameter receives a gradient (DDP-safe)', not unused, str(unused[:5]))
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 512, 512))
    check('eval output 512', set(out) == {'seg'} and tuple(out['seg'].shape) == (1, 1, 512, 512))
    st = T.model_stats(model, 512, torch.device('cpu'), amp=False)
    check('model stats', 'gflops' in st, f"{st['params_M']:.2f} M params, {st.get('gmacs', float('nan')):.1f} GMACs")
    for flags in (['--no-hires', '--no-prior', '--boundary-weight', '0', '--contrast-weight', '0', '--aux-weights', '0', '0'],
                  ['--no-prior'], ['--no-attention']):
        a2 = T.parse_args(['--splits', 'x', '--out', 'y', '--encoder', 'convnext_atto', '--no-pretrained'] + flags)
        m2 = T.build_model(a2, pretrained=False).train()
        o2 = m2(torch.rand(2, 3, 64, 64))
        l2, _ = T.compute_loss(o2, (torch.rand(2, 1, 64, 64) > 0.7).float(), a2)
        l2.backward()
        unused = [n for n, p in m2.named_parameters() if p.requires_grad and p.grad is None]
        check(f'ablation {" ".join(flags)}', not unused and bool(torch.isfinite(l2)), str(unused[:3]))
    ema = T.ModelEMA(model, 0.9)
    pnames = {n for n, _ in model.named_parameters()}
    keys = [k for k, v in ema.module.state_dict().items() if v.dtype.is_floating_point]
    before = {k: ema.module.state_dict()[k].clone() for k in keys}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    dcy = min(0.9, 2.0 / 11.0)
    after = ema.module.state_dict()
    errs = [k for k in keys if not torch.allclose(
        after[k], before[k] * dcy + (before[k] + (1.0 if k in pnames else 0.0)) * (1 - dcy), atol=1e-5)]
    check('EMA update (params and buffers)', not errs, f'{len(keys)} tensors' + (f', wrong: {errs[:3]}' if errs else ''))
    if download:
        enc = T.Encoder('convnext_tiny.dinov3_lvd1689m', pretrained=True)
        check('DINOv3 ConvNeXt-T pretrained weights load', enc.channels == [96, 192, 384, 768], str(enc.channels))
        enc = T.Encoder('convnext_tiny.fb_in22k_ft_in1k', pretrained=True)
        check('IN22k ConvNeXt-T pretrained weights load', enc.channels == [96, 192, 384, 768], str(enc.channels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true', help='also download and load the pretrained encoders')
    a = p.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    for fn in (test_metrics, test_tta_identity, test_augment_alignment, test_prior_and_losses, test_lr):
        print(fn.__name__)
        fn()
    print('test_encoders_and_model')
    test_encoders_and_model(a.download)
    print(f'\nALL {len(PASSED)} UNIT CHECKS PASSED')


if __name__ == '__main__':
    main()
