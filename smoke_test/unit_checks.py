#!/usr/bin/env python3
"""Unit checks for train_landslide.py and ablation.py (CPU). Run: python smoke_test/unit_checks.py [--download]"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablation as A  # noqa: E402
import train_landslide as T  # noqa: E402

PASSED = []
TINY_FM = 'vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m'  # 5.5 M parameters, random weights in tests


def check(name, cond, detail=''):
    if not cond:
        raise AssertionError(f'FAILED: {name} {detail}')
    PASSED.append(name)
    print(f'  ok  {name} {detail}')


def args(*extra):
    return T.parse_args(['--splits', 'x', '--out', 'y', '--encoder', 'convnext_atto', '--no-pretrained', *extra])


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


class PixelModel(nn.Module):
    def forward(self, x):
        return {'seg': torch.logit(x[:, :1].clamp(1e-4, 1 - 1e-4))}


def test_tta():
    x = torch.rand(3, 3, 32, 32)
    ref = x[:, :1].clamp(1e-4, 1 - 1e-4)
    p = T.predict_probs(PixelModel(), x, tta=True, amp=False, channels_last=False)
    check('D4 TTA inverse = identity', torch.allclose(p, ref, atol=1e-5), f'max err {(p - ref).abs().max():.2e}')

    class PositionModel(nn.Module):  # output depends on absolute position -> TTA must average 8 transforms
        def forward(self, x):
            h = x.shape[-1]
            ramp = torch.linspace(-2, 2, h).view(1, 1, 1, h).expand(x.shape[0], 1, h, h)
            return {'seg': ramp.clone()}

    p = T.predict_probs(PositionModel(), torch.rand(1, 3, 16, 16), tta=True, amp=False, channels_last=False)
    check('TTA output symmetric', torch.allclose(p, torch.rot90(p, 1, dims=[2, 3]), atol=1e-6))
    out = T.predict_modes(PixelModel(), x, ('noTTA', 'TTA', 'MSTTA'), False, False, (0.75, 1.0, 1.25))
    check('shared passes: noTTA = identity pass', torch.allclose(out['noTTA'], ref, atol=1e-6))
    check('shared passes: TTA = predict_probs',
          torch.equal(out['TTA'], T.predict_probs(PixelModel(), x, True, False, False)))
    c = torch.full((2, 3, 64, 64), 0.3)
    ms = T.predict_modes(PixelModel(), c, ('MSTTA',), False, False, (0.75, 1.0, 1.25))['MSTTA']
    check('MS-TTA on a constant image is exact', torch.allclose(ms, torch.full_like(ms, 0.3), atol=1e-5)
          and tuple(ms.shape) == (2, 1, 64, 64))
    class PriorModel(nn.Module):
        def forward(self, x, prior=None):
            v = 0.8 if prior == 'lite' else 0.2
            return {'seg': torch.logit(torch.full((x.shape[0], 1, *x.shape[-2:]), v))}

    pl = T.predict_modes(PriorModel(), x, ('noTTA', 'TTA'), False, False, prior='lite')
    pd = T.predict_modes(PriorModel(), x, ('TTA',), False, False)
    check('predict_modes passes the prior to the model', torch.allclose(pl['TTA'], torch.full_like(pl['TTA'], 0.8))
          and torch.allclose(pd['TTA'], torch.full_like(pd['TTA'], 0.2)))
    calls = []

    class SizeSpy(nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape[-2:]))
            return {'seg': torch.zeros(x.shape[0], 1, *x.shape[-2:])}

    T.predict_modes(SizeSpy(), torch.rand(1, 3, 128, 128), ('noTTA', 'TTA', 'MSTTA'), False, False, (0.75, 1.0, 1.25))
    check('MS-TTA uses 8 + 2 + 2 passes, sizes multiple of 32', len(calls) == 12 and
          sorted(set(calls)) == [(96, 96), (128, 128), (160, 160)], str(sorted(set(calls))))
    calls.clear()
    T.predict_modes(SizeSpy(), torch.rand(1, 3, 64, 64), ('MSTTA',), False, False, (0.75, 1.0, 1.25))
    check('MS-TTA skips scales that round back to the input size', len(calls) == 8, str(len(calls)))


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
    logits = torch.randn(2, 1, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.6).float()
    sem = torch.randn(2, 1, 4, 4)
    for mining in ('error', 'fm', 'error+fm', 'random'):
        emb = torch.randn(2, 16, 8, 8, requires_grad=True)
        torch.manual_seed(3)
        l = T.pixel_contrast_loss(emb, logits, mask, n_per_class=32, temperature=0.1, pos_thr=0.5, neg_thr=0.5,
                                  mining=mining, sem_score=sem)
        l.backward()
        check(f'contrast loss ({mining}) finite, positive, gradient flows',
              bool(torch.isfinite(l)) and float(l) > 0 and float(emb.grad.abs().sum()) > 0, f'{float(l):.4f}')
    emb = torch.randn(2, 16, 8, 8)
    torch.manual_seed(5)
    l1 = T.pixel_contrast_loss(emb, logits, mask, 32, 0.1, 0.5, 0.5)
    torch.manual_seed(5)
    l2 = T.pixel_contrast_loss(emb, logits, mask, 32, 0.1, 0.5, 0.5, mining='fm', sem_score=None)
    check("'fm' mining without foundation features falls back to 'error'", torch.equal(l1, l2))
    emb2 = torch.randn(2, 16, 8, 8, requires_grad=True)
    l0 = T.pixel_contrast_loss(emb2, logits, torch.zeros(2, 1, 32, 32), mining='error+fm', sem_score=sem)
    l0.backward()
    check('contrast on empty mask = 0, graph-connected', float(l0) == 0.0 and emb2.grad is not None)
    d = T.bce_dice(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    check('bce_dice finite on empty mask', bool(torch.isfinite(d)))


def test_mining_helpers():
    # foundation features: landslide cells ~ e0, background ~ e1, one background cell looks like a landslide
    f = torch.zeros(1, 4, 4, 4)
    f[:, 1] = 1.0
    f[:, :, :2, :2] = 0.0
    f[:, 0, :2, :2] = 1.0
    f[:, :, 3, 3] = 0.0
    f[:, 0, 3, 3] = 1.0  # confuser
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :8, :8] = 1.0
    s = T.fm_semantic_score(f, mask)[0, 0]
    bg = s.clone()
    bg[:2, :2] = -9
    check('semantic score: confuser is the most landslide-like background cell',
          tuple((bg == bg.max()).nonzero()[0].tolist()) == (3, 3) and float(s[0, 0]) > 0 > float(s[2, 2]))
    cls = torch.tensor([True, True, False, True, True, True])
    score = torch.tensor([0.1, 0.9, 5.0, 0.3, 0.8, 0.2])
    top = T._top_frac(cls, score, 0.4)
    check('top-fraction mask', top.tolist() == [False, True, False, False, True, False])
    torch.manual_seed(0)
    cls = torch.rand(400) > 0.3
    g1, g2 = cls & (torch.rand(400) > 0.8), cls & (torch.rand(400) > 0.9)
    idx = T._sample_groups(cls, [g1, g2], 60)
    check('grouped sampling: distinct, in class, n', idx.numel() == 60 and idx.unique().numel() == 60 and
          bool(cls[idx].all()))
    check('grouped sampling: quota from each hard group', int(g1[idx].sum()) >= 20 and int(g2[idx].sum()) >= 1)
    check('fm block indices', T.fm_indices(24, 4) == [5, 11, 17, 23] and T.fm_indices(12, 4) == [2, 5, 8, 11]
          and T.fm_indices(12, 1) == [11])


def test_lr():
    f = [T.lr_factor(s, 1000, 100, 0.01) for s in (0, 50, 100, 550, 1000, 2000)]
    check('warmup starts at 1%', abs(f[0] - 0.01) < 1e-12)
    check('warmup mid', abs(f[1] - (0.01 + 0.99 * 0.5)) < 1e-12)
    check('peak after warmup', abs(f[2] - 1.0) < 1e-12)
    check('cosine middle', abs(f[3] - (0.01 + 0.99 * 0.5)) < 1e-9)
    check('end at min ratio', abs(f[4] - 0.01) < 1e-12 and abs(f[5] - 0.01) < 1e-12)


def grads_ok(m):
    return [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]


def test_encoders_and_model(download):
    for name in ('convnext_atto', 'resnet18', 'regnety_002', 'efficientnet_b0', 'convnext_tiny'):
        enc = T.Encoder(name, pretrained=False)
        feats = enc(torch.rand(1, 3, 64, 64))
        shapes = [tuple(f.shape) for f in feats]
        ok = all(f.shape[-1] == 64 // r and f.shape[1] == c for f, r, c in zip(feats, (4, 8, 16, 32), enc.channels))
        check(f'encoder {name} strides 4/8/16/32', ok, str(shapes))
    a0 = T.parse_args(['--splits', 'x', '--out', 'y'])
    check('defaults: DINOv3-SAT ViT-L prior, gated fusion, prior hallucination, error mining, auto precision',
          a0.fm == 'vit_large_patch16_dinov3.sat493m' and a0.fm_mode == 'fuse' and a0.fm_fusion == 'gated'
          and a0.fm_sub == 'hal' and a0.fm_sub_p == 0.5 and a0.hal_weight == 0.5 and a0.hard_mining == 'error'
          and a0.precision == 'auto')
    an = T.parse_args(['--splits', 'x', '--out', 'y', '--fm', 'none'])
    check("'--fm none': no substitution, no hallucination loss, 'error' mining",
          an.fm_sub == 'none' and an.hal_weight == 0.0 and an.hard_mining == 'error')
    check("'--no-amp' = '--precision fp32'; '--hard-mining auto' keeps the v1 meaning; distill has no substitution",
          T.parse_args(['--splits', 'x', '--out', 'y', '--no-amp']).precision == 'fp32' and
          T.parse_args(['--splits', 'x', '--out', 'y', '--hard-mining', 'auto']).hard_mining == 'error+fm' and
          T.parse_args(['--splits', 'x', '--out', 'y', '--fm-mode', 'distill']).fm_sub == 'none')
    a = T.parse_args(['--splits', 'x', '--out', 'y', '--encoder', 'convnext_tiny', '--no-pretrained', '--fm', 'none'])
    model = T.build_model(a, pretrained=False)
    model.train()
    out = model(torch.rand(2, 3, 64, 64))
    check('train outputs', set(out) == {'seg', 'aux4', 'aux8', 'bnd', 'emb'} and tuple(out['seg'].shape) == (2, 1, 64, 64),
          str({k: tuple(v.shape) for k, v in out.items()}))
    loss, parts = T.compute_loss(out, (torch.rand(2, 1, 64, 64) > 0.7).float(), a)
    loss.backward()
    check('loss parts [seg, aux4, aux8, bnd, con, dis]', parts.numel() == 6)
    check('every parameter receives a gradient (DDP-safe)', not grads_ok(model), str(grads_ok(model)[:5]))
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 512, 512))
    check('eval output 512', set(out) == {'seg'} and tuple(out['seg'].shape) == (1, 1, 512, 512))
    st = T.model_stats(model, 512, torch.device('cpu'), amp=False)
    check('model stats', 'gflops' in st and st['fm_frozen_params_M'] == 0.0,
          f"{st['params_M']:.2f} M params, {st.get('gmacs', float('nan')):.1f} GMACs")
    for flags in (['--no-hires', '--no-prior', '--boundary-weight', '0', '--contrast-weight', '0', '--aux-weights', '0', '0'],
                  ['--no-prior'], ['--no-attention']):
        a2 = args('--fm', 'none', *flags)
        m2 = T.build_model(a2, pretrained=False).train()
        o2 = m2(torch.rand(2, 3, 64, 64))
        l2, _ = T.compute_loss(o2, (torch.rand(2, 1, 64, 64) > 0.7).float(), a2)
        l2.backward()
        check(f'ablation {" ".join(flags)}', not grads_ok(m2) and bool(torch.isfinite(l2)), str(grads_ok(m2)[:3]))
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
    # legacy checkpoint arguments (SP-UNet version: no foundation-model keys)
    legacy = {k: v for k, v in vars(args('--fm', 'none')).items() if not k.startswith(('fm', 'hard_mining',
                                                                                         'distill', 'ms_', 'tag'))}
    ml = T.build_model(argparse.Namespace(**legacy), pretrained=False)
    check('legacy checkpoint args build SP-UNet', ml.fm_name is None and T.arg(legacy, 'hard_mining') == 'error')
    v1 = {k: v for k, v in vars(args('--img-size', '64', '--fm', TINY_FM, '--fm-sub', 'none')).items()
          if k not in ('fm_sub', 'fm_sub_p', 'hal_weight', 'precision', 'select_prior', 'fp32_switch_frac')}
    m1 = T.build_model(argparse.Namespace(**v1), pretrained=False)
    check('v1 DPF-Net checkpoint args build the v1 network (no hallucination head)',
          m1.fm_sub == 'none' and m1.hal is None and not m1.lite_capable and m1.needs_fm)
    check('training precision of earlier runs', T.run_precision({'no_amp': True}) == 'fp32' and
          T.run_precision({'no_amp': False}) == 'fp16' and T.run_precision({'precision': 'auto'}) == 'auto')
    if download:
        enc = T.Encoder('convnext_tiny.dinov3_lvd1689m', pretrained=True)
        check('DINOv3 ConvNeXt-T pretrained weights load', enc.channels == [96, 192, 384, 768], str(enc.channels))


def test_foundation_prior():
    torch.manual_seed(0)
    inj = T.SemanticInjection(32, 16, 8, gated=True).eval()
    c, f, s = torch.randn(2, 32, 8, 8), torch.randn(2, 16, 4, 4), torch.randn(2, 8, 16, 16)
    check('gated injection is the identity at initialisation (zero-init BN)', torch.equal(inj(c, f, s), c))
    inj2 = T.SemanticInjection(32, 16, 0, gated=False).eval()
    check('additive injection is the identity at initialisation', torch.equal(inj2(c, f), c))
    variants = {'v1 fuse/gated': ['--fm', TINY_FM, '--fm-sub', 'none'],
                'v1 fuse/add': ['--fm', TINY_FM, '--fm-sub', 'none', '--fm-fusion', 'add'],
                'prior dropout': ['--fm', TINY_FM, '--fm-sub', 'zero'],
                'prior hallucination': ['--fm', TINY_FM, '--fm-sub', 'hal'],
                'hallucination + error+fm mining': ['--fm', TINY_FM, '--hard-mining', 'error+fm'],
                'distill': ['--fm', TINY_FM, '--fm-mode', 'distill'],
                'no structural prior': ['--fm', TINY_FM, '--no-hires', '--no-prior'],
                'fm-size 48': ['--fm', TINY_FM, '--fm-size', '48']}
    for name, flags in variants.items():
        a = args('--img-size', '64', *flags)
        m = T.build_model(a, pretrained=False).train()
        out = m(torch.rand(4, 3, 64, 64))
        y = torch.zeros(4, 1, 64, 64)
        y[:, :, 8:40, 8:40] = 1.0
        loss, parts = T.compute_loss(out, y, a)
        loss.backward()
        sub = a.fm_sub
        want = ({'seg', 'aux4', 'aux8', 'bnd', 'emb', 'fm_sem'} | ({'distill'} if 'distill' in name else set())
                | ({'fm_rows'} if sub != 'none' else set()) | ({'hal', 'hal_target'} if sub == 'hal' else set()))
        if sub != 'none':
            check(f'FM {name}: the real prior is used for half of the batch only',
                  out['fm_rows'].numel() == 2 and out['fm_sem'].shape[0] == 2)
        check(f'FM {name}: outputs, finite loss, every parameter gets a gradient',
              set(out) == want and bool(torch.isfinite(loss)) and not grads_ok(m),
              f'parts {[round(v, 3) for v in parts.tolist()]}' + (f' unused {grads_ok(m)[:3]}' if grads_ok(m) else ''))
    # distill mode must not run the foundation model at inference
    a = args('--img-size', '64', '--fm', TINY_FM, '--fm-mode', 'distill')
    m = T.build_model(a, pretrained=False).eval()
    orig = T.get_foundation
    T.get_foundation = lambda *k, **kw: (_ for _ in ()).throw(RuntimeError('FM used at inference'))
    try:
        with torch.no_grad():
            o = m(torch.rand(1, 3, 64, 64))
        ok = set(o) == {'seg'}
    finally:
        T.get_foundation = orig
    check('DPF-Lite inference does not touch the foundation model', ok)
    # prior substitution: 'lite' inference without the foundation model; at initialisation full == lite
    for sub in ('zero', 'hal'):
        a = args('--img-size', '64', '--fm', TINY_FM, '--fm-sub', sub)
        m = T.build_model(a, pretrained=False).eval()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            full = m(x, prior='full')['seg']
        T.get_foundation = lambda *k, **kw: (_ for _ in ()).throw(RuntimeError('FM used at inference'))
        try:
            with torch.no_grad():
                lite = m(x, prior='lite')['seg']
            ok = True
        except RuntimeError:
            ok = False
        finally:
            T.get_foundation = orig
        check(f'--fm-sub {sub}: lite inference without the foundation model; identical to full at init (zero-init)',
              ok and torch.equal(full, lite) and m.lite_capable
              and T.inference_variants(m) == [('', 'full'), ('-lite', 'lite')])
    m = T.build_model(args('--img-size', '64', '--fm', TINY_FM, '--fm-sub', 'none'), pretrained=False).eval()
    try:
        m(torch.rand(1, 3, 64, 64), prior='lite')
        refused = False
    except ValueError:
        refused = True
    check('a v1 (no substitution) model refuses lite inference', refused and T.inference_variants(m) == [('', None)])
    t = torch.randn(2, 8, 4, 4)
    check('hallucination loss: 0 for a perfect prediction, 1 for a zero prediction',
          float(T.hallucination_loss(t, t)) == 0.0
          and abs(float(T.hallucination_loss(torch.zeros_like(t), t)) - 1) < 1e-5)
    f = torch.nn.functional.normalize(torch.randn(2, 8, 4, 4), dim=1)
    mask = torch.zeros(3, 1, 16, 16)
    mask[:, :, :8, :8] = 1.0
    s = T.semantic_score({'fm_sem': f, 'fm_rows': torch.tensor([0, 2])}, mask)
    check('semantic score: NaN exactly for the substituted samples',
          tuple(s.shape) == (3, 1, 4, 4) and bool(torch.isnan(s[1]).all()) and bool(torch.isfinite(s[[0, 2]]).all()))
    emb = torch.randn(3, 16, 8, 8, requires_grad=True)
    l = T.pixel_contrast_loss(emb, torch.randn(3, 1, 16, 16), mask, n_per_class=16, pos_thr=0.5, neg_thr=0.5,
                              mining='error+fm', sem_score=s)
    l.backward()
    check('semantic mining ignores samples without foundation features', bool(torch.isfinite(l)) and
          float(emb.grad.abs().sum()) > 0)
    # the frozen ViT is shared, excluded from state_dict / EMA / optimizer
    a = args('--img-size', '64', '--fm', TINY_FM)
    m = T.build_model(a, pretrained=False)
    vit = T.get_foundation(TINY_FM, False)
    check('foundation model frozen, outside state_dict and optimizer',
          not any(p.requires_grad for p in vit.parameters()) and not any(k.startswith('fm.') for k in m.state_dict())
          and all(p.requires_grad for g in T.build_optimizer(m, a).param_groups for p in g['params']))
    ema = T.ModelEMA(m, 0.99)
    check('EMA copy shares the frozen foundation model', T.get_foundation(TINY_FM, False) is vit and
          ema.module.fm_name == TINY_FM)
    m2 = T.build_model(a, pretrained=False)
    with torch.no_grad():
        f1 = m.foundation_features(torch.rand(1, 3, 64, 64))
    check('random foundation weights are deterministic across rebuilds (DDP ranks agree)',
          all(torch.equal(p1, p2) for p1, p2 in zip(T.get_foundation(TINY_FM, False).parameters(),
                                                    T.get_foundation(TINY_FM, False).parameters()))
          and len(f1) == 4 and tuple(f1[0].shape) == (1, 192, 4, 4) and m2.fm_idx == [2, 5, 8, 11])
    st = T.param_report(m)
    hal = sum(p.numel() for p in m.hal.parameters()) / 1e6
    adapter = sum(p.numel() for p in m.fm_adapter.parameters()) / 1e6
    check('parameter report: frozen FM counted for full inference, hallucination head only for lite',
          st['fm_frozen_params_M'] > 5.0 and abs(st['inference_params_M'] - (
              st['params_M'] - st['train_only_heads_M'] - hal + st['fm_frozen_params_M'])) < 1e-9
          and abs(st['inference_params_lite_M'] - (st['params_M'] - st['train_only_heads_M'] - adapter)) < 1e-9)


def test_evaluate_counts():
    class Loader:  # minimal stand-in for a DataLoader
        def __init__(self, n):
            g = torch.Generator().manual_seed(0)
            self.dataset = list(range(n))
            self.batches = []
            for s in range(0, n, 3):
                idx = torch.arange(s, min(n, s + 3))
                img = (torch.rand(len(idx), 3, 32, 32, generator=g) * 255).to(torch.uint8)
                mask = (torch.rand(len(idx), 1, 32, 32, generator=g) > 0.7).to(torch.uint8)
                self.batches.append((img, mask, idx))

        def __iter__(self):
            return iter(self.batches)

    d = SimpleNamespace(device=torch.device('cpu'), sum_=lambda t: t)
    a = SimpleNamespace(amp=False, channels_last=False, ms_scales=[0.75, 1.0, 1.25])
    ld = Loader(8)
    rid = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
    ev = T.evaluate(PixelModel(), ld, d, a, modes=('noTTA', 'TTA', 'MSTTA'),
                    thresholds={'noTTA': [0.5, 0.4], 'TTA': [0.5, 0.4], 'MSTTA': [0.5]}, region_ids=rid, n_regions=3,
                    per_tile=True)
    c, r, t = ev['counts'], ev['regions'], ev['tiles']
    check('per-tile counts sum to the dataset counts (last threshold)',
          all(torch.allclose(t[m].sum(0), c[m][-1]) for m in c))
    check('per-region counts sum to the dataset counts', all(torch.allclose(r[m].sum(0), c[m]) for m in c))
    check('every pixel counted once', all(float(c[m][0].sum()) == 8 * 32 * 32 for m in c))


def test_results_and_bootstrap(tmp):
    m = T.metrics_from_counts(80, 10, 10, 900)
    res = {'run': 'x', 'tag': 'B6', 'args': vars(args('--fm', TINY_FM, '--ms-tta')), 'n_test': 5, 'n_val': 2,
           'weights': 'w', 'threshold': 0.45, 'threshold_source': 'best validation IoU with TTA', 'tta': True,
           'tta_mode': 'MSTTA', 'main_key': 'test_MSTTA@t*', 'thresholds': {'TTA': 0.5, 'MSTTA': 0.45},
           'test_noTTA@0.5': m, 'test_TTA@0.5': m, 'test_TTA@t*': m, 'test_MSTTA@0.5': m, 'test_MSTTA@t*': m,
           'val_TTA@0.5': m, 'val_TTA@t*': m, 'val_MSTTA@0.5': m, 'val_MSTTA@t*': m, 'per_region': {}}
    md = T.results_markdown(res, 'SAT')
    check('results.md marks the validation-selected MS-TTA row as main',
          'D4+MS-TTA, t=t* (main)' in md and 'D4-TTA, t=t* (main)' not in md and 'semantic prior' in md)
    lite = {**res, 'args': vars(args('--fm', TINY_FM)), 'tta_mode': 'TTA', 'main_key': 'test_TTA@t*',
            'lite_key': 'test_TTA-lite@t*', 'tta_mode_lite': 'TTA-lite', 'test_noTTA-lite@0.5': m,
            'test_TTA-lite@0.5': m, 'test_TTA-lite@t*': m, 'val_TTA-lite@0.5': m, 'val_TTA-lite@t*': m,
            'thresholds': {'TTA': 0.5, 'TTA-lite': 0.45},
            'per_region_by_mode': {'TTA-lite': {'R1': {**m, 'tiles': 3}}}}
    md = T.results_markdown(lite, 'SAT')
    check('results.md lists the evaluation without the foundation model',
          'D4-TTA, t=t*, no FM at inference (main, no FM)' in md and 'D4-TTA, no FM at inference)' in md
          and 'prior hallucination' in md and 'training precision: auto' in md and 'without the foundation model' in md)
    old = {'run': 'old', 'args': vars(args('--fm', 'none')), 'n_test': 5, 'n_val': 2, 'weights': 'w',
           'threshold': 0.5, 'threshold_source': 's', 'tta': True, 'main_key': 'test_TTA@t*', 'test_TTA@t*': m,
           'val_at_0.5': m, 'val_at_t*': m}
    for k in [k for k in old['args'] if k.startswith(('fm', 'hard_mining'))]:
        del old['args'][k]
    md = T.results_markdown(old, 'SAT')
    check('results.md renders SP-UNet-era results', 'Ours, D4-TTA, t=t* (main)' in md and 'Validation (2 tiles' in md)
    rng = np.random.default_rng(0)
    tp = rng.integers(50, 200, 300).astype(float)
    Acnt = np.stack([tp, rng.integers(10, 60, 300), rng.integers(10, 60, 300), np.full(300, 5000.0)], 1).astype(float)
    b = A.bootstrap(Acnt, Acnt.copy(), n_boot=500)
    check('paired bootstrap of identical runs: delta 0, CI [0, 0]', b['delta'] == 0 and b['delta_ci'] == [0.0, 0.0])
    Bcnt = Acnt.copy()
    Bcnt[:, 1] = np.maximum(Bcnt[:, 1] - 8, 0)  # fewer false positives
    b = A.bootstrap(Acnt, Bcnt, n_boot=500)
    check('paired bootstrap detects an improvement', b['delta'] > 0 and b['delta_ci'][0] > 0 and b['p'] < 0.01,
          f'delta {100 * b["delta"]:.2f} pp, CI [{100 * b["delta_ci"][0]:.2f}, {100 * b["delta_ci"][1]:.2f}]')
    check('bootstrap CI contains the point estimate', b['ci'][0] <= b['iou'] <= b['ci'][1])
    hp = A.holm([0.01, 0.04, 0.03])
    check('Holm adjustment', np.allclose(hp, [0.03, 0.06, 0.06]) and A.holm([]) == [], str(hp))
    check('fm map replaces foundation models only', A.map_fm(['--fm', A.FM_SAT, '--fm-mode', 'fuse', '--fm', 'none'],
                                                             {A.FM_SAT: 'tiny'}) == ['--fm', 'tiny', '--fm-mode', 'fuse',
                                                                                      '--fm', 'none'])
    ids = list(A.EXPERIMENTS)

    def known(e):
        return e.split(':')[0] in A.EXPERIMENTS

    check('registry: unique ids, every pair / chain / seed entry references known ids',
          len(ids) == len(set(ids)) and all(known(x) and known(y) for x, y in A.PAIRS)
          and all(known(t) for c in A.CHAIN.values() for t in c) and all(known(t) for t in A.SEEDS))
    bad = []
    for eid, e in A.EXPERIMENTS.items():
        try:
            T.parse_args(['--splits', 'x', '--out', 'y'] + e['args'])
        except SystemExit:
            bad.append(eid)
    check('every registry entry parses', not bad, str(bad))
    def P(eid):
        return T.parse_args(['--splits', 'x', '--out', 'y'] + A.EXPERIMENTS[eid]['args'])

    t = P('B3')
    check('B3 = SP-UNet configuration', t.fm == 'none' and t.hard_mining == 'error' and not t.no_prior
          and t.boundary_weight > 0 and t.contrast_weight > 0 and t.precision == 'auto')
    t4, t5, t6, t7 = P('B4'), P('B5'), P('B6'), P('B7')
    check('B4 / B5 / B6 / B7 differ only in prior substitution and mining',
          (t4.fm_sub, t5.fm_sub, t6.fm_sub, t7.fm_sub) == ('none', 'zero', 'hal', 'hal')
          and (t4.hard_mining, t6.hard_mining, t7.hard_mining) == ('error', 'error', 'error+fm')
          and t6.fm == A.FM_SAT and t6.fm_fusion == 'gated' and t6.hal_weight == 0.5)
    prec = {eid: P(eid).precision for eid in ids}
    check('every row trains with --precision auto except the precision controls',
          all(v == 'auto' for k, v in prec.items() if k not in ('P0', 'R3', 'W3'))
          and (prec['P0'], prec['R3'], prec['W3']) == ('fp16', 'fp32', 'fp32'))
    runs = {'B6': {'res': {'ablation_key': 'test_TTA@t*', 'main_key': 'test_TTA@t*',
                           'lite_ablation_key': 'test_TTA-lite@t*', 'lite_key': 'test_TTA-lite@t*',
                           'test_TTA@t*': m, 'test_TTA-lite@t*': m}},
            'B3': {'res': {'ablation_key': 'test_TTA@t*', 'test_TTA@t*': m}}}
    ent = A.entries_of(runs)
    check('collector rows: run + "<id>:lite" for runs evaluated without the FM',
          list(ent) == ['B3', 'B6', 'B6:lite'] and A.key_mode(ent['B6:lite']['key']) == 'TTA-lite'
          and ent['B6:lite']['lite'] and not ent['B6']['lite'])
    # the exact command lines the Kaggle notebook builds (RUN / ENV / COLLECT cells)
    p = A.build_parser()
    r = p.parse_args(['run', '--work', '/kaggle/working', '--queue', 'B3', 'B6', '--epochs', '80', '--batch-size', '8',
                      '--session-hours', '11.5', '--t0', '1795000000.5', '--extra=', '--prev', '/kaggle/input',
                      '--import', 'R3=/kaggle/input/x/runs/r/best.pth', 'W3=/kaggle/input/y/best.pth', '--quick'])
    check('notebook RUN command parses', r.queue == ['B3', 'B6'] and r.prev == ['/kaggle/input'] and r.extra == ''
          and A.parse_map(r.imports, 'i') == {'R3': '/kaggle/input/x/runs/r/best.pth', 'W3': '/kaggle/input/y/best.pth'}
          and r.quick and r.t0 == 1795000000.5)
    r = p.parse_args(['run', '--work', 'w', '--queue', 'B6', '--extra=--grad-ckpt --fm-size 384', '--prev'])
    check('extra arguments and an empty --prev parse', A.split_args(r.extra) == ['--grad-ckpt', '--fm-size', '384']
          and r.prev == [])
    w = p.parse_args(['weights', '--queue', 'B6', 'V6', '--extra='])
    c = p.parse_args(['collect', '--roots', '/kaggle/working/runs', '/kaggle/input', '--out', '/kaggle/working/ablation'])
    check('notebook ENV and COLLECT commands parse', w.queue == ['B6', 'V6'] and len(c.roots) == 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true', help='also download and load the pretrained encoder')
    a = p.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    for fn in (test_metrics, test_tta, test_augment_alignment, test_prior_and_losses, test_mining_helpers, test_lr,
               test_foundation_prior, test_evaluate_counts):
        print(fn.__name__)
        fn()
    print('test_encoders_and_model')
    test_encoders_and_model(a.download)
    print('test_results_and_bootstrap')
    test_results_and_bootstrap(None)
    print(f'\nALL {len(PASSED)} UNIT CHECKS PASSED')


if __name__ == '__main__':
    main()
