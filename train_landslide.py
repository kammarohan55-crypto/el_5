#!/usr/bin/env python3
"""
train_landslide.py - DPF-Net (Dual-Prior Fusion Network) for CAS Landslide segmentation (SAT / UAV).

Model  : timm CNN encoder (default ConvNeXt-T, DINOv3) + PPM context + scSE UNet decoder, with two priors:
         * structural prior - 10 fixed spectral-structural descriptors (FP32) -> high-resolution stem;
         * semantic prior   - multi-layer features of a frozen remote-sensing foundation ViT
           (default DINOv3 ViT-L/16 SAT-493M) injected into the 1/8, 1/16, 1/32 encoder features by
           structure-aware gated fusion ('fuse'), or used only during training ('distill', no inference cost);
         + boundary head + deep supervision + pixel contrastive head with error- and foundation-guided
           hard-anchor mining.  '--fm none' gives the earlier SP-UNet exactly.
Train  : DDP (torchrun), FP16 autocast + dynamic GradScaler, AdamW, warmup + cosine,
         EMA weights, GPU augmentation, time budget with resume.
Eval   : EMA weights, dataset-level confusion counts, D4 (+ optional multi-scale) test-time augmentation,
         TTA mode and threshold selected on the validation split only, per-region and per-tile counts.

Kaggle (GPU T4 x2):
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m torch.distributed.run --standalone \
      --nproc_per_node=2 train_landslide.py --splits /kaggle/working/splits/SAT \
      --out /kaggle/working/runs/sat_full
Single GPU / CPU:  python train_landslide.py --splits ... --out ...
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise SystemExit('timm is required:  pip install timm') from exc

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:  # noqa: BLE001
    pass

SWEEP_THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(17)]  # 0.300 ... 0.700 (contains 0.5)
ARCH_KEYS = ('encoder', 'img_size', 'no_hires', 'no_prior', 'no_attention', 'fm', 'fm_mode', 'fm_fusion',
             'fm_channels', 'fm_layers', 'fm_size')
# values assumed for arguments that checkpoints / results of the earlier SP-UNet version do not contain
LEGACY_DEFAULTS = {'fm': 'none', 'fm_mode': 'fuse', 'fm_fusion': 'gated', 'fm_channels': 256, 'fm_layers': 4,
                   'fm_size': 0, 'hard_mining': 'error', 'distill_weight': 0.0, 'ms_tta': False,
                   'ms_scales': [0.75, 1.0, 1.25], 'tag': ''}
TRAIN_ONLY_HEADS = ('aux4', 'aux8', 'bnd_head', 'proj', 'distill_head')
MODE_LABELS = {'noTTA': 'no TTA', 'TTA': 'D4-TTA', 'MSTTA': 'D4+MS-TTA'}


def arg(ns, key):
    """Argument value of a Namespace or dict, falling back to LEGACY_DEFAULTS for older checkpoints."""
    if isinstance(ns, dict):
        return ns.get(key, LEGACY_DEFAULTS.get(key))
    return getattr(ns, key, LEGACY_DEFAULTS.get(key))


def fm_enabled(name):
    return name not in (None, '', 'none')


# ----------------------------------------------------------------------------------------
# distributed helpers
# ----------------------------------------------------------------------------------------
class Dist:
    def __init__(self):
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.enabled = self.world > 1
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cpu')
        if self.enabled:
            backend = 'nccl' if self.device.type == 'cuda' else 'gloo'
            dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=60))

    @property
    def main(self):
        return self.rank == 0

    def barrier(self):
        if self.enabled:
            if self.device.type == 'cuda':
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def sum_(self, t):
        if self.enabled:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def any_flag(self, flag):
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        if self.enabled:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0.5)

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def log(d, *args):
    if d.main:
        print(*args, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------
def read_split(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not {'name', 'region', 'img', 'mask'} <= set(r):
            raise ValueError(f'{path}: expected columns name,region,img,mask')
    return rows


def rows_hash(*splits):
    h = hashlib.sha256()
    for i, rows in enumerate(splits):
        for r in sorted(rows, key=lambda r: r['name']):
            h.update(f'{i}:{r["region"]}:{r["name"]}\n'.encode('utf-8'))
    return h.hexdigest()


def imread_any(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f'Empty file: {path}')
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Could not decode: {path}')
    return img


def to_uint8_rgb(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:
        img = np.stack([img[:, :, 0]] * 3, axis=-1)
    elif img.shape[2] >= 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
        x = (x - lo) / max(float(hi - lo), 1e-6)
        img = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, ::-1])  # OpenCV BGR -> RGB


def to_binary_mask(m):
    if m.ndim == 3:
        m = m[:, :, 0]
    thr = 0 if m.max() <= 1 else 127
    return (m > thr).astype(np.uint8)


class SegDataset(Dataset):
    """Returns (uint8 3xHxW RGB, uint8 1xHxW mask, index). D4 flips/rotations for training;
    all other augmentation runs on the GPU (gpu_augment)."""

    def __init__(self, rows, img_size, train):
        self.rows, self.size, self.train = rows, img_size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = to_uint8_rgb(imread_any(r['img']))
        m = to_binary_mask(imread_any(r['mask']))
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        s = self.size
        if img.shape[0] != s or img.shape[1] != s:
            interp = cv2.INTER_AREA if img.shape[0] > s else cv2.INTER_LINEAR
            img = cv2.resize(img, (s, s), interpolation=interp)
            m = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
        if self.train:
            k = random.randint(0, 3)
            if k:
                img, m = np.rot90(img, k), np.rot90(m, k)
            if random.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
        m = torch.from_numpy(np.ascontiguousarray(m))[None]
        return img, m, i


class ShardSampler(Sampler):
    """Exact, non-overlapping sharding for distributed evaluation (no padded duplicates)."""

    def __init__(self, n, rank, world):
        self.idx = list(range(rank, n, world))

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


@torch.no_grad()
def gpu_augment(x, y, a):
    """x: Bx3xHxW float in [0,1]; y: Bx1xHxW float {0,1}. Geometric + photometric, per sample."""
    B, _, H, W = x.shape
    dev = x.device
    if a.aug_affine_p > 0:
        on = (torch.rand(B, device=dev) < a.aug_affine_p).float()
        ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(a.aug_rotate) * on
        lo, hi = math.log(a.aug_scale_min), math.log(a.aug_scale_max)
        sc = torch.exp(torch.empty(B, device=dev).uniform_(lo, hi) * on)
        tx = (torch.rand(B, device=dev) * 2 - 1) * a.aug_translate * on
        ty = (torch.rand(B, device=dev) * 2 - 1) * a.aug_translate * on
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, [B, 3, H, W], align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        y = F.grid_sample(y, grid, mode='nearest', padding_mode='reflection', align_corners=False)
    if a.aug_color_p > 0:
        on = (torch.rand(B, 1, 1, 1, device=dev) < a.aug_color_p).float()

        def jitter(mag, shape=(B, 1, 1, 1)):
            return 1.0 + (torch.rand(shape, device=dev) * 2 - 1) * mag * on

        x = x * jitter(a.aug_brightness)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * jitter(a.aug_contrast) + mean
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = (x - gray) * jitter(a.aug_saturation) + gray
        x = x * jitter(a.aug_channel_gain, (B, 3, 1, 1))
        x = x.clamp(0.0, 1.0)
    if a.aug_gamma_p > 0:
        on = (torch.rand(B, 1, 1, 1, device=dev) < a.aug_gamma_p).float()
        g = torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(0.8), math.log(1.25)) * on)
        x = x.clamp(min=1e-6).pow(g)
    if a.aug_noise_p > 0:
        on = (torch.rand(B, 1, 1, 1, device=dev) < a.aug_noise_p).float()
        sigma = torch.rand(B, 1, 1, 1, device=dev) * a.aug_noise_std * on
        x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
    return x.contiguous(), y.contiguous()


# ----------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------
def _gauss1d(sigma):
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


class SpectralStructuralPrior(nn.Module):
    """Ten fixed, differentiable descriptors computed on the GPU in FP32 from RGB in [0,1]:
    gray, Sobel magnitude, LoG(s=1), DoG(1,2), DoG(2,4), local std (7x7, 15x15),
    local contrast normalisation, ExG vegetation index, saturation."""
    names = ('gray', 'sobel_mag', 'log_s1', 'dog_1_2', 'dog_2_4', 'local_std_7',
             'local_std_15', 'local_contrast', 'exg', 'saturation')
    out_channels = 10

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer('k_sobel', torch.stack([sx, sx.t()])[:, None].contiguous(), persistent=False)
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer('k_lap', lap[None, None].contiguous(), persistent=False)
        for s in (1, 2, 4):
            self.register_buffer(f'g{s}', _gauss1d(float(s)), persistent=False)

    @staticmethod
    def _blur(x, k):
        r = (k.numel() - 1) // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))

    @staticmethod
    def _box(x, w):
        r = w // 2
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='reflect'), w, stride=1)

    def forward(self, rgb):
        rgb = rgb.float().contiguous()
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gxy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), self.k_sobel)
        sobel = torch.sqrt(gxy.pow(2).sum(1, keepdim=True) + 1e-12)
        b1, b2, b4 = self._blur(gray, self.g1), self._blur(gray, self.g2), self._blur(gray, self.g4)
        log1 = F.conv2d(F.pad(b1, (1, 1, 1, 1), mode='reflect'), self.k_lap)

        def local_std(w):
            m = self._box(gray, w)
            m2 = self._box(gray * gray, w)
            return torch.sqrt((m2 - m * m).clamp(min=0.0) + 1e-8)

        std7, std15 = local_std(7), local_std(15)
        lcn = ((gray - b4) / (std15 + 1e-2)).clamp(-6.0, 6.0)
        exg = (2.0 * G - R - B) / (R + G + B + 1e-6)
        mx = rgb.amax(dim=1, keepdim=True)
        mn = rgb.amin(dim=1, keepdim=True)
        sat = (mx - mn) / (mx + 1e-6)
        return torch.cat([gray, sobel, log1, b1 - b2, b2 - b4, std7, std15, lcn, exg, sat], 1)


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze & excitation."""

    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, ch, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, attention=True):
        super().__init__()
        self.conv1 = ConvBNReLU(cin + cskip, cout)
        self.conv2 = ConvBNReLU(cout, cout)
        self.att = SCSE(cout) if attention else nn.Identity()

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], 1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.att(self.conv2(self.conv1(x)))


class PPM(nn.Module):
    """Pyramid pooling context module (no BatchNorm on the pooled branches)."""

    def __init__(self, cin, cout, bins=(1, 2, 3, 6)):
        super().__init__()
        red = max(cin // len(bins), 64)
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(cin, red, 1),
                                                   nn.ReLU(inplace=True)) for b in bins])
        self.bottleneck = ConvBNReLU(cin + red * len(bins), cout)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False)
                       for s in self.stages]
        return self.bottleneck(torch.cat(feats, 1))


class Encoder(nn.Module):
    """timm features_only encoder returning the stride 4/8/16/32 feature maps."""

    def __init__(self, name, pretrained, grad_ckpt=False):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True)
        reds = list(self.body.feature_info.reduction())
        chs = list(self.body.feature_info.channels())
        self.idx = []
        for r in (4, 8, 16, 32):
            cands = [i for i, rr in enumerate(reds) if rr == r]
            if not cands:
                raise ValueError(f'Encoder {name} has no stride-{r} feature map (reductions {reds}).')
            self.idx.append(cands[-1])
        self.channels = [chs[i] for i in self.idx]
        cfg = getattr(self.body, 'pretrained_cfg', None) or {}
        mean = tuple(cfg.get('mean', (0.485, 0.456, 0.406)))
        std = tuple(cfg.get('std', (0.229, 0.224, 0.225)))
        if len(mean) != 3 or len(std) != 3:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if grad_ckpt and hasattr(self.body, 'set_grad_checkpointing'):
            self.body.set_grad_checkpointing(True)
        # sanity check: NCHW outputs with the advertised channels
        was_training = self.body.training
        self.body.eval()
        with torch.no_grad():
            feats = self.body(torch.zeros(1, 3, 64, 64))
        self.body.train(was_training)
        for i, c in zip(self.idx, self.channels):
            if feats[i].ndim != 4 or feats[i].shape[1] != c:
                raise ValueError(f'Encoder {name}: feature {i} has shape {tuple(feats[i].shape)}, expected NCHW '
                                 f'with {c} channels. Use a CNN encoder (ConvNeXt / RegNet / ResNet / EfficientNet).')

    def forward(self, x):
        feats = self.body(x)
        return [feats[i] for i in self.idx]


# ----------------------------------------------------------------------------------------
# foundation-model semantic prior
# ----------------------------------------------------------------------------------------
_FM_CACHE = {}


def fm_indices(depth, n):
    """n evenly spaced block indices ending at the last block (depth 24, n 4 -> [5, 11, 17, 23])."""
    n = max(1, min(int(n), depth))
    return sorted({max(0, int(round(depth * (i + 1) / n)) - 1) for i in range(n)})


def get_foundation(name, pretrained):
    """Frozen timm ViT shared by every model instance of this process. It lives outside the trainable module
    tree, so it is never trained, never copied into the EMA model, never wrapped by DDP and never written to
    checkpoints; loading a checkpoint re-creates it from timm. Creation does not consume the global RNG, so
    runs with and without a foundation model share the same random streams."""
    key = (name, bool(pretrained))
    vit = _FM_CACHE.get(key)
    if vit is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)  # random weights (tests) are identical on every rank and every rebuild
            try:
                vit = timm.create_model(name, pretrained=pretrained, dynamic_img_size=True)
            except TypeError:
                vit = timm.create_model(name, pretrained=pretrained)
        if not hasattr(vit, 'forward_intermediates') or not hasattr(vit, 'blocks'):
            raise ValueError(f'--fm {name}: expected a timm ViT with forward_intermediates() '
                             '(e.g. vit_large_patch16_dinov3.sat493m).')
        vit.eval().requires_grad_(False)
        _FM_CACHE[key] = vit
    return vit


def _resize(f, size):
    """Resizes a feature map: average pooling when shrinking, bilinear interpolation otherwise."""
    size = (int(size[0]), int(size[1]))
    if tuple(f.shape[-2:]) == size:
        return f
    if f.shape[-2] >= size[0] and f.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(f, size)
    return F.interpolate(f, size=size, mode='bilinear', align_corners=False)


class SemanticInjection(nn.Module):
    """Injects the foundation-model semantic map f into the CNN feature map c of one pyramid level.
      gated (structure-aware):  c' = c + g * P(f),  g = sigmoid(MLP([c, P(f), S])), where S is the
                                structural-prior stem (1/4) pooled to this level, when the stem exists;
      add:                      c' = c + P(f).
    P ends with a zero-initialised BatchNorm, so training starts exactly from the CNN-only network."""

    def __init__(self, c, cf, cs=0, gated=True):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(cf, c, 1, bias=False), nn.BatchNorm2d(c))
        nn.init.zeros_(self.proj[1].weight)
        self.cs = cs
        self.gate = None
        if gated:
            mid = max(c // 4, 16)
            self.gate = nn.Sequential(nn.Conv2d(2 * c + cs, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, c, 1))

    def forward(self, c, f, s=None):
        fp = self.proj(_resize(f, c.shape[-2:])).to(c.dtype)
        if self.gate is None:
            return c + fp
        parts = [c, fp]
        if self.cs:
            parts.append(_resize(s, c.shape[-2:]).to(c.dtype))
        g = torch.sigmoid(self.gate(torch.cat(parts, 1))).to(c.dtype)
        return c + g * fp


class DualPriorUNet(nn.Module):
    """UNet decoder over a timm CNN encoder with two optional priors (see the module docstring).
    fm=None reproduces the earlier Structural-Prior UNet (SP-UNet) exactly."""

    def __init__(self, encoder='convnext_tiny.dinov3_lvd1689m', pretrained=True, hires=True, prior=True,
                 boundary=True, contrast=True, deep_sup=True, attention=True, proj_dim=64, grad_ckpt=False,
                 fm=None, fm_pretrained=True, fm_mode='fuse', fm_fusion='gated', fm_channels=256, fm_layers=4,
                 fm_scale=1.0):
        super().__init__()
        if prior and not hires:
            raise ValueError('The structural prior needs the high-resolution stem (use --no-prior with --no-hires).')
        if fm_mode not in ('fuse', 'distill') or fm_fusion not in ('gated', 'add'):
            raise ValueError(f'unknown fm_mode / fm_fusion: {fm_mode} / {fm_fusion}')
        self.hires, self.use_prior = hires, prior
        self.encoder = Encoder(encoder, pretrained, grad_ckpt)
        c4, c8, c16, c32 = self.encoder.channels
        s1 = s2 = s4 = 0
        if hires:
            cin = 3
            if prior:
                self.prior = SpectralStructuralPrior()
                self.prior_bn = nn.BatchNorm2d(SpectralStructuralPrior.out_channels)
                cin += SpectralStructuralPrior.out_channels
            self.stem1 = nn.Sequential(ConvBNReLU(cin, 16), ConvBNReLU(16, 16))
            self.stem2 = nn.Sequential(ConvBNReLU(16, 32, stride=2), ConvBNReLU(32, 32))
            self.stem4 = nn.Sequential(ConvBNReLU(32, 48, stride=2), ConvBNReLU(48, 48))
            s1, s2, s4 = 16, 32, 48
        self.ppm = PPM(c32, 256)
        self.dec16 = DecoderBlock(256, c16, 256, attention)
        self.dec8 = DecoderBlock(256, c8, 128, attention)
        self.dec4 = DecoderBlock(128, c4 + s4, 64, attention)
        self.dec2 = DecoderBlock(64, s2, 32, attention)
        self.dec1 = DecoderBlock(32, s1, 16, attention)
        self.seg_head = nn.Sequential(ConvBNReLU(16, 16), nn.Conv2d(16, 1, 1))
        self.aux4 = nn.Conv2d(64, 1, 1) if deep_sup else None
        self.aux8 = nn.Conv2d(128, 1, 1) if deep_sup else None
        self.bnd_head = nn.Sequential(ConvBNReLU(16, 16), nn.Conv2d(16, 1, 1)) if boundary else None
        self.proj = (nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                   nn.Conv2d(64, proj_dim, 1)) if contrast else None)
        nn.init.constant_(self.seg_head[-1].bias, -2.0)
        if deep_sup:
            nn.init.constant_(self.aux4.bias, -2.0)
            nn.init.constant_(self.aux8.bias, -2.0)
        if boundary:
            nn.init.constant_(self.bnd_head[-1].bias, -3.0)
        # ---- semantic prior (created after all SP-UNet modules, so their initialisation is unchanged) ----
        self.fm_name = fm if fm_enabled(fm) else None
        self.fm_mode, self.fm_pretrained, self.fm_scale = fm_mode, bool(fm_pretrained), float(fm_scale)
        self.fm_fallbacks = 0
        self.distill_head = None
        if self.fm_name:
            vit = get_foundation(self.fm_name, self.fm_pretrained)
            ps = vit.patch_embed.patch_size
            self.fm_patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
            self.fm_idx = fm_indices(len(vit.blocks), fm_layers)
            self.fm_dim = int(vit.embed_dim)
            cfg = getattr(vit, 'pretrained_cfg', None) or {}
            mean = tuple(cfg.get('mean', (0.485, 0.456, 0.406)))
            std = tuple(cfg.get('std', (0.229, 0.224, 0.225)))
            if len(mean) != 3 or len(std) != 3:
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            self.register_buffer('fm_mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
            self.register_buffer('fm_std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
            if fm_mode == 'fuse':
                gated = fm_fusion == 'gated'
                self.fm_adapter = nn.Sequential(
                    nn.Conv2d(self.fm_dim * len(self.fm_idx), fm_channels, 1, bias=False),
                    nn.BatchNorm2d(fm_channels), nn.GELU(),
                    nn.Conv2d(fm_channels, fm_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fm_channels), nn.GELU())
                cs = s4 if gated else 0
                self.inj8 = SemanticInjection(c8, fm_channels, cs, gated)
                self.inj16 = SemanticInjection(c16, fm_channels, cs, gated)
                self.inj32 = SemanticInjection(c32, fm_channels, cs, gated)
            else:  # distill: predict the foundation features from the CNN's 1/16 features (training only)
                self.distill_head = nn.Sequential(nn.Conv2d(c16, 256, 1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
                                                  nn.Conv2d(256, self.fm_dim, 1))

    @torch.no_grad()
    def foundation_features(self, x):
        """Frozen-ViT features of the blocks in self.fm_idx (final LayerNorm applied), FP32 NCHW at 1/patch of
        the foundation-model input (x resized by fm_scale, multiple of the patch size, own normalisation)."""
        vit = get_foundation(self.fm_name, self.fm_pretrained)
        if next(vit.parameters()).device != x.device:
            vit.to(x.device)
        H, W = x.shape[-2:]
        ps = self.fm_patch
        h = max(ps, int(round(H * self.fm_scale / ps)) * ps)
        w = max(ps, int(round(W * self.fm_scale / ps)) * ps)
        xi = x.float()
        if (h, w) != (H, W):
            xi = F.interpolate(xi, size=(h, w), mode='bilinear', align_corners=False, antialias=h < H)
        xi = (xi - self.fm_mean) / self.fm_std
        kw = dict(indices=self.fm_idx, norm=True, output_fmt='NCHW', intermediates_only=True)
        feats = [f.float() for f in vit.forward_intermediates(xi, **kw)]
        if not bool(torch.stack([torch.isfinite(f).all() for f in feats]).all()):
            with torch.autocast(device_type=x.device.type, enabled=False):  # FP16 overflow: recompute in FP32
                feats = [f.float() for f in vit.forward_intermediates(xi, **kw)]
            self.fm_fallbacks += 1
        return feats

    def forward(self, x):
        """x: Bx3xHxW RGB in [0,1] (float32). Training: dict with seg/aux4/aux8/bnd/emb (+ fm_sem / distill
        with a foundation model); eval: {'seg'}."""
        H, W = x.shape[-2:]
        xn = (x - self.encoder.mean) / self.encoder.std
        c4, c8, c16, c32 = self.encoder(xn)
        p1 = p2 = p4 = None
        if self.hires:
            s_in = xn
            if self.use_prior:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    pr = self.prior_bn(self.prior(x.float()))
                s_in = torch.cat([xn.float(), pr], 1)
            p1 = self.stem1(s_in)
            p2 = self.stem2(p1)
            p4 = self.stem4(p2)
            if p4.shape[-2:] != c4.shape[-2:]:
                p4 = F.interpolate(p4, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        fm_sem = None
        if self.fm_name and (self.fm_mode == 'fuse' or self.training):
            feats = self.foundation_features(x)
            if self.training:
                fm_sem = F.normalize(feats[-1], dim=1)
            if self.fm_mode == 'fuse':
                f = self.fm_adapter(torch.cat(feats, 1))
                c8 = self.inj8(c8, f, p4)
                c16 = self.inj16(c16, f, p4)
                c32 = self.inj32(c32, f, p4)
        d16 = self.dec16(self.ppm(c32), c16)
        d8 = self.dec8(d16, c8)
        d4 = self.dec4(d8, c4 if p4 is None else torch.cat([c4, p4.to(c4.dtype)], 1))
        d2 = self.dec2(d4, p2)
        d1 = self.dec1(d2, p1)
        seg = self.seg_head(d1)
        if seg.shape[-2:] != (H, W):
            seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        out = {'seg': seg}
        if self.training:
            if self.aux4 is not None:
                out['aux4'] = self.aux4(d4)
                out['aux8'] = self.aux8(d8)
            if self.bnd_head is not None:
                b = self.bnd_head(d1)
                if b.shape[-2:] != (H, W):
                    b = F.interpolate(b, size=(H, W), mode='bilinear', align_corners=False)
                out['bnd'] = b
            if self.proj is not None:
                out['emb'] = self.proj(d4)
            if fm_sem is not None:
                out['fm_sem'] = fm_sem
            if self.distill_head is not None:
                out['distill'] = self.distill_head(c16)
        return out


class Ensemble(nn.Module):
    """Averages member probabilities; returns logits of the mean probability."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        p = torch.stack([torch.sigmoid(m(x)['seg'].float()) for m in self.models]).mean(0)
        return {'seg': torch.logit(p.clamp(1e-6, 1 - 1e-6))}


def build_model(a, pretrained):
    """a: argparse Namespace of this or an earlier version (missing keys -> LEGACY_DEFAULTS). The foundation
    model is always loaded from timm (pretrained unless the run used --no-pretrained); it is not in checkpoints."""
    fm_size = int(arg(a, 'fm_size') or 0)
    return DualPriorUNet(
        encoder=a.encoder, pretrained=pretrained, hires=not a.no_hires, prior=not a.no_prior,
        boundary=a.boundary_weight > 0, contrast=a.contrast_weight > 0,
        deep_sup=any(w > 0 for w in a.aux_weights), attention=not a.no_attention,
        grad_ckpt=getattr(a, 'grad_ckpt', False),
        fm=arg(a, 'fm'), fm_pretrained=not getattr(a, 'no_pretrained', False), fm_mode=arg(a, 'fm_mode'),
        fm_fusion=arg(a, 'fm_fusion'), fm_channels=int(arg(a, 'fm_channels')), fm_layers=int(arg(a, 'fm_layers')),
        fm_scale=(fm_size / a.img_size) if fm_size else 1.0)


# ----------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------
def bce_dice(logits, target, smooth=1.0):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (p * target).sum() + smooth) / (p.sum() + target.sum() + smooth)
    return bce + dice


def boundary_target(mask, k=3):
    dil = F.max_pool2d(mask, k, 1, k // 2)
    ero = -F.max_pool2d(-mask, k, 1, k // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _sample_idx(cls_mask, hard_mask, n):
    hard = hard_mask.nonzero(as_tuple=False).squeeze(1)
    easy = (cls_mask & ~hard_mask).nonzero(as_tuple=False).squeeze(1)
    n_hard = min(hard.numel(), n // 2)
    n_easy = min(easy.numel(), n - n_hard)
    n_hard = min(hard.numel(), n - n_easy)
    sel = []
    if n_hard > 0:
        sel.append(hard[torch.randperm(hard.numel(), device=hard.device)[:n_hard]])
    if n_easy > 0:
        sel.append(easy[torch.randperm(easy.numel(), device=easy.device)[:n_easy]])
    return torch.cat(sel) if sel else hard[:0]


def _top_frac(cls_mask, score, frac):
    """Boolean mask of the top `frac` fraction (by score) of the pixels in cls_mask."""
    out = torch.zeros_like(cls_mask)
    idx = cls_mask.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return out
    k = max(1, int(math.ceil(frac * idx.numel())))
    out[idx[torch.topk(score[idx], k).indices]] = True
    return out


def _sample_groups(cls_mask, groups, n):
    """Up to n distinct pixels of one class: n // (len(groups) + 1) from each hard group in turn, the rest
    uniformly from the remaining pixels of the class."""
    avail = cls_mask.clone()
    quota = n // (len(groups) + 1)
    sel = []
    for g in groups:
        cand = (g & avail).nonzero(as_tuple=False).squeeze(1)
        k = min(cand.numel(), quota)
        if k > 0:
            pick = cand[torch.randperm(cand.numel(), device=cand.device)[:k]]
            avail[pick] = False
            sel.append(pick)
    rest = avail.nonzero(as_tuple=False).squeeze(1)
    k = min(rest.numel(), n - sum(int(s.numel()) for s in sel))
    if k > 0:
        sel.append(rest[torch.randperm(rest.numel(), device=rest.device)[:k]])
    return torch.cat(sel) if sel else rest[:0]


def fm_semantic_score(fm_sem, mask):
    """Landslide-likeness of every foundation-model cell, s = cos(f, mu_ls) - cos(f, mu_bg), where mu_ls / mu_bg
    are batch prototypes of the L2-normalised frozen features weighted by the landslide / background fraction
    of each cell. High s on background = semantic confuser (bare soil, river bed, road ...). No gradient."""
    with torch.no_grad():
        f = fm_sem.float()
        h, w = f.shape[-2:]
        t = F.adaptive_avg_pool2d(mask.float(), (h, w))
        mu_ls = F.normalize((f * t).sum(dim=(0, 2, 3)), dim=0)
        mu_bg = F.normalize((f * (1.0 - t)).sum(dim=(0, 2, 3)), dim=0)
        s = torch.einsum('bdhw,d->bhw', f, mu_ls - mu_bg)
    return s[:, None]


def pixel_contrast_loss(emb, seg_logits, mask, n_per_class=256, temperature=0.1, pos_thr=0.9, neg_thr=0.1,
                        mining='error', sem_score=None, sem_frac=0.2):
    """Supervised pixel contrastive loss on the 1/4-resolution embedding with hard-anchor mining (FP32).
    mining  'error'    : half of the anchors of each class are pixels the current prediction gets wrong;
            'fm'       : half are semantic hard pixels - the top sem_frac of background pixels the frozen
                         foundation model finds most landslide-like, and of landslide pixels it finds least so;
            'error+fm' : one third prediction errors, one third semantic hard pixels, one third random;
            'random'   : uniformly random anchors.
    Without sem_score, 'fm' / 'error+fm' fall back to 'error'."""
    emb = F.normalize(emb.float(), dim=1)
    _, D, h, w = emb.shape
    zero = emb.sum() * 0.0
    with torch.no_grad():
        t = F.adaptive_avg_pool2d(mask.float(), (h, w)).flatten()
        prob = F.adaptive_avg_pool2d(torch.sigmoid(seg_logits.detach().float()), (h, w)).flatten()
        pos, neg = t >= pos_thr, t <= neg_thr
        err_p, err_n = pos & (prob < 0.5), neg & (prob >= 0.5)
        use_sem = mining in ('fm', 'error+fm') and sem_score is not None
        sem_p = sem_n = pos
        if use_sem:
            s = F.interpolate(sem_score.float(), size=(h, w), mode='bilinear', align_corners=False).flatten()
            sem_p, sem_n = _top_frac(pos, -s, sem_frac), _top_frac(neg, s, sem_frac)
        if mining == 'random':
            none = torch.zeros_like(pos)
            idx_p, idx_n = _sample_idx(pos, none, n_per_class), _sample_idx(neg, none, n_per_class)
        elif use_sem and mining == 'fm':
            idx_p, idx_n = _sample_groups(pos, [sem_p], n_per_class), _sample_groups(neg, [sem_n], n_per_class)
        elif use_sem:
            idx_p = _sample_groups(pos, [err_p, sem_p], n_per_class)
            idx_n = _sample_groups(neg, [err_n, sem_n], n_per_class)
        else:
            idx_p, idx_n = _sample_idx(pos, err_p, n_per_class), _sample_idx(neg, err_n, n_per_class)
    if idx_p.numel() < 2 or idx_n.numel() < 2:
        return zero
    z = emb.permute(0, 2, 3, 1).reshape(-1, D)
    anchors = torch.cat([z[idx_p], z[idx_n]], 0)
    labels = torch.cat([torch.ones(idx_p.numel(), device=emb.device),
                        torch.zeros(idx_n.numel(), device=emb.device)])
    n = anchors.shape[0]
    eye = torch.eye(n, device=emb.device, dtype=torch.bool)
    sim = (anchors @ anchors.t() / temperature).masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean() + zero


def compute_loss(out, mask, a):
    """Returns (total_loss, vector [seg, aux4, aux8, bnd, con, dis]) in FP32."""
    zero = torch.zeros((), device=mask.device)
    l_seg = bce_dice(out['seg'], mask)
    total = l_seg
    l_a4 = l_a8 = l_bnd = l_con = l_dis = zero
    if 'aux4' in out:
        size = mask.shape[-2:]
        l_a4 = bce_dice(F.interpolate(out['aux4'].float(), size=size, mode='bilinear', align_corners=False), mask)
        l_a8 = bce_dice(F.interpolate(out['aux8'].float(), size=size, mode='bilinear', align_corners=False), mask)
        total = total + a.aux_weights[0] * l_a4 + a.aux_weights[1] * l_a8
    if 'bnd' in out:
        l_bnd = bce_dice(out['bnd'], boundary_target(mask))
        total = total + a.boundary_weight * l_bnd
    if 'emb' in out:
        mining = arg(a, 'hard_mining')
        sem = (fm_semantic_score(out['fm_sem'], mask)
               if mining in ('fm', 'error+fm') and 'fm_sem' in out else None)
        l_con = pixel_contrast_loss(out['emb'], out['seg'], mask, a.contrast_anchors, a.contrast_temp,
                                    mining=mining, sem_score=sem)
        total = total + a.contrast_weight * l_con
    if 'distill' in out and 'fm_sem' in out:
        pred = _resize(out['distill'].float(), out['fm_sem'].shape[-2:])
        l_dis = (1.0 - F.cosine_similarity(pred, out['fm_sem'].float(), dim=1)).mean()
        total = total + arg(a, 'distill_weight') * l_dis
    parts = torch.stack([l_seg.detach(), l_a4.detach(), l_a8.detach(), l_bnd.detach(), l_con.detach(),
                         l_dis.detach()])
    return total, parts


# ----------------------------------------------------------------------------------------
# optimisation helpers
# ----------------------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of all floating-point parameters and buffers."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0
        self._pairs = None

    def _build(self, model):
        esd, msd = self.module.state_dict(), model.state_dict()
        fe, fm, oe, om = [], [], [], []
        for k, v in esd.items():
            if v.dtype.is_floating_point:
                fe.append(v)
                fm.append(msd[k])
            else:
                oe.append(v)
                om.append(msd[k])
        self._pairs = (fe, fm, oe, om)

    @torch.no_grad()
    def update(self, model):
        if self._pairs is None:
            self._build(model)
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        fe, fm, oe, om = self._pairs
        if hasattr(torch, '_foreach_mul_') and hasattr(torch, '_foreach_add_'):
            torch._foreach_mul_(fe, d)  # fused multi-tensor kernels (fast path)
            torch._foreach_add_(fe, fm, alpha=1.0 - d)
        else:
            for e, m in zip(fe, fm):
                e.mul_(d).add_(m, alpha=1.0 - d)
        for e, m in zip(oe, om):
            e.copy_(m)

    def state_dict(self):
        return {'module': self.module.state_dict(), 'updates': self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd['module'])
        self.updates = int(sd['updates'])
        self._pairs = None


def build_optimizer(model, a):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        enc = n.startswith('encoder.')
        no_decay = p.ndim <= 1 or n.endswith('.bias')
        groups.setdefault((enc, no_decay), []).append(p)
    param_groups = []
    for (enc, no_decay), params in groups.items():
        lr = a.enc_lr if enc else a.dec_lr
        param_groups.append({'params': params, 'lr': lr, 'base_lr': lr,
                             'weight_decay': 0.0 if no_decay else a.weight_decay,
                             'group_name': ('encoder' if enc else 'decoder') + ('_no_decay' if no_decay else '')})
    return torch.optim.AdamW(param_groups, lr=a.dec_lr, betas=(0.9, 0.999), eps=1e-8)


def lr_factor(step, total, warmup, min_ratio):
    if warmup > 0 and step < warmup:
        return 0.01 + 0.99 * step / warmup
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ----------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------
def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = (float(v) for v in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    iou_bg = tn / (tn + fp + fn) if tn + fp + fn > 0 else 0.0
    f1_bg = 2 * tn / (2 * tn + fp + fn) if tn + fp + fn > 0 else 0.0
    oa = (tp + tn) / n if n > 0 else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n > 0 else 0.0
    kappa = (oa - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1, 'iou': iou, 'iou_bg': iou_bg,
            'miou': 0.5 * (iou + iou_bg), 'f1_bg': f1_bg, 'mf1': 0.5 * (f1 + f1_bg), 'oa': oa,
            'kappa': kappa, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


@torch.no_grad()
def predict_modes(model, x, modes, amp, channels_last, ms_scales=(0.75, 1.0, 1.25)):
    """x: Bx3xHxW float in [0,1]. Returns {mode: Bx1xHxW FP32 probabilities} for modes in
    {'noTTA', 'TTA', 'MSTTA'}, sharing forward passes: noTTA is the identity pass among the 8 D4 passes, TTA
    their mean, MSTTA the mean over scales of [TTA at scale 1, flip-averaged prediction at every other scale]
    (8 + 2 per extra scale forward passes)."""

    def run(inp):
        inp = inp.contiguous(memory_format=torch.channels_last) if channels_last else inp.contiguous()
        with torch.autocast(device_type=inp.device.type, dtype=torch.float16, enabled=amp):
            return torch.sigmoid(model(inp)['seg'].float())

    modes = set(modes)
    if not modes & {'TTA', 'MSTTA'}:
        return {'noTTA': run(x)}
    H, W = x.shape[-2:]
    acc = torch.zeros(x.shape[0], 1, H, W, device=x.device, dtype=torch.float32)
    ident = None
    for flip in (False, True):
        xf = torch.flip(x, dims=[3]) if flip else x
        for k in range(4):
            p = torch.rot90(run(torch.rot90(xf, k, dims=[2, 3])), -k, dims=[2, 3])
            p = torch.flip(p, dims=[3]) if flip else p
            if ident is None:
                ident = p
            acc += p
    d4 = acc / 8.0
    out = {}
    if 'noTTA' in modes:
        out['noTTA'] = ident
    if 'TTA' in modes:
        out['TTA'] = d4
    if 'MSTTA' in modes:
        ps = [d4]
        for s in ms_scales:
            if abs(float(s) - 1.0) < 1e-6:
                continue
            h, w = max(32, int(round(H * s / 32)) * 32), max(32, int(round(W * s / 32)) * 32)
            if (h, w) == (H, W):  # tiny inputs: the scale rounds back to the original size
                continue
            xs = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=h < H)
            q = 0.5 * (run(xs) + torch.flip(run(torch.flip(xs, dims=[3])), dims=[3]))
            ps.append(F.interpolate(q, size=(H, W), mode='bilinear', align_corners=False))
        out['MSTTA'] = torch.stack(ps).mean(0)
    return out


def predict_probs(model, x, tta, amp, channels_last):
    """x: Bx3xHxW float in [0,1]. Returns Bx1xHxW FP32 probabilities (D4-TTA averaged if tta)."""
    mode = 'TTA' if tta else 'noTTA'
    return predict_modes(model, x, (mode,), amp, channels_last)[mode]


@torch.no_grad()
def evaluate(model, loader, d, a, modes=('noTTA',), thresholds=None, region_ids=None, n_regions=0, sink=None,
             sink_mode=None, per_tile=False):
    """Dataset-level confusion counts [tp, fp, fn, tn] for every prediction mode and threshold (and per region /
    per tile), summed over all ranks. thresholds: {mode: [t, ...]} (default [0.5]); per-tile counts use the last
    threshold of each mode. Returns {'counts': {mode: T x 4}, 'regions': {mode: R x T x 4} | None,
    'tiles': {mode: N x 4} | None} (CPU, float64)."""
    model.eval()
    modes = list(modes)
    ths = {m: list((thresholds or {}).get(m, [0.5])) for m in modes}
    dev = d.device
    counts = {m: torch.zeros(len(ths[m]), 4, dtype=torch.float64, device=dev) for m in modes}
    reg = ({m: torch.zeros(n_regions, len(ths[m]), 4, dtype=torch.float64, device=dev) for m in modes}
           if n_regions else None)
    tiles = ({m: torch.zeros(len(loader.dataset), 4, dtype=torch.float64, device=dev) for m in modes}
             if per_tile else None)
    ms_scales = tuple(float(s) for s in arg(a, 'ms_scales'))
    for img, mask, idx in loader:
        x = img.to(dev, non_blocking=True).float().div_(255.0)
        y = mask.to(dev, non_blocking=True) > 0
        probs = predict_modes(model, x, modes, a.amp, a.channels_last, ms_scales)
        B = x.shape[0]
        yf = y.reshape(B, -1)
        npx = float(yf.shape[1])
        rid = region_ids[idx].to(dev) if reg is not None else None
        for m in modes:
            pf = probs[m].reshape(B, -1)
            per = None
            for ti, t in enumerate(ths[m]):
                pr = pf >= t
                tp = (pr & yf).sum(1).double()
                fp = (pr & ~yf).sum(1).double()
                fn = (~pr & yf).sum(1).double()
                per = torch.stack([tp, fp, fn, npx - tp - fp - fn], 1)
                counts[m][ti] += per.sum(0)
                if reg is not None:
                    reg[m][:, ti].index_add_(0, rid, per)
            if tiles is not None:
                tiles[m][idx.to(dev)] = per
        if sink is not None:
            sink(idx, x, y, probs[sink_mode if sink_mode in probs else modes[-1]])
    for m in modes:
        d.sum_(counts[m])
        if reg is not None:
            d.sum_(reg[m])
        if tiles is not None:
            d.sum_(tiles[m])

    def cpu(dd):
        return None if dd is None else {k: v.cpu() for k, v in dd.items()}

    return {'counts': cpu(counts), 'regions': cpu(reg), 'tiles': cpu(tiles)}


class VisSink:
    """Writes RGB | ground truth | prediction-error panels (TP green, FP red, FN blue)."""

    def __init__(self, folder, rows, max_n, threshold, enabled):
        self.folder, self.rows, self.max_n, self.thr = Path(folder), rows, max_n, threshold
        self.enabled, self.n = enabled and max_n > 0, 0
        if self.enabled:
            self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, idx, x, y, p):
        if not self.enabled or self.n >= self.max_n:
            return
        for b in range(x.shape[0]):
            if self.n >= self.max_n:
                return
            img = (x[b].permute(1, 2, 0).float().cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            gt = y[b, 0].cpu().numpy().astype(bool)
            pr = (p[b, 0] >= self.thr).cpu().numpy()
            gt_ov, err = img.copy(), img.copy()
            gt_ov[gt] = (0.45 * gt_ov[gt] + 0.55 * np.array([255, 255, 0])).astype(np.uint8)
            for m, col in ((gt & pr, (0, 255, 0)), (~gt & pr, (255, 0, 0)), (gt & ~pr, (0, 90, 255))):
                err[m] = (0.4 * err[m] + 0.6 * np.array(col)).astype(np.uint8)
            panel = np.concatenate([img, gt_ov, err], 1)
            ok, buf = cv2.imencode('.png', cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            if ok:
                buf.tofile(str(self.folder / f'{safe_name(self.rows[int(idx[b])]["name"])}.png'))
            self.n += 1


def param_report(model):
    """Trainable parameters (incl. training-only heads), frozen foundation-model parameters (counted once even
    if shared by ensemble members) and the parameters actually used at inference."""
    nets = [m for m in model.modules() if isinstance(m, DualPriorUNet)]
    total = sum(p.numel() for p in model.parameters())
    heads = 0
    for net in nets:
        for name in TRAIN_ONLY_HEADS:
            mod = getattr(net, name, None)
            if isinstance(mod, nn.Module):
                heads += sum(p.numel() for p in mod.parameters())
    fm_infer, fm_train = {}, {}
    for net in nets:
        if net.fm_name:
            n = sum(p.numel() for p in get_foundation(net.fm_name, net.fm_pretrained).parameters())
            (fm_infer if net.fm_mode == 'fuse' else fm_train)[net.fm_name] = n
    fm_i = sum(fm_infer.values())
    return {'params_M': total / 1e6, 'train_only_heads_M': heads / 1e6, 'fm_frozen_params_M': fm_i / 1e6,
            'fm_training_only_params_M': sum(fm_train.values()) / 1e6,
            'inference_params_M': (total - heads + fm_i) / 1e6}


@torch.no_grad()
def model_stats(model, size, device, amp, channels_last=False):
    stats = param_report(model)
    model.eval()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.rand(1, 3, size, size, device=device)
        with FlopCounterMode(display=False) as fc:
            model(x)
        flops = fc.get_total_flops()
        stats['gflops'] = flops / 1e9
        stats['gmacs'] = flops / 2e9
    except Exception as e:  # noqa: BLE001
        stats['flops_error'] = str(e)[:200]
    if device.type == 'cuda':
        x = torch.rand(8, 3, size, size, device=device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                model(x)
        torch.cuda.synchronize(device)
        t0, n_it = time.time(), 10
        for _ in range(n_it):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                model(x)
        torch.cuda.synchronize(device)
        stats['fps_batch8'] = 8 * n_it / (time.time() - t0)
        stats['gpu'] = torch.cuda.get_device_name(device)
    return stats


BENCHMARKS = {
    # (method, source, same split?, P, R, IoU, mIoU, OA, mF1, F1)  values in %; F1 = landslide-class F1
    'SAT': [
        ('FCN (VGG16)', 'Xu et al. 2024, Sci. Data', 'no', 62.981, 84.142, 55.716, 75.173, 94.972, 84.391, None),
        ('U-Net (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 61.795, 78.550, 51.179, 72.619, 94.410, 82.316, None),
        ('DeepLabv3+ (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 74.275, 89.187, 68.137, 82.397, 96.881, 89.675, None),
        ('MFFENet (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 74.141, 89.141, 67.998, 82.318, 96.862, 89.621, None),
        ('SegFormer', '2025, quoted by SC-Net', 'no', None, None, 79.30, None, None, None, None),
        ('U-Net (ResNet50)', 'Song et al. 2026 (SC-Net paper)', 'yes', 84.39, 88.02, 75.69, None, None, None, 86.17),
        ('DeepLabV3+ (ResNet50)', 'Song et al. 2026 (SC-Net paper)', 'yes', 86.90, 87.99, 77.68, None, None, None, 87.44),
        ('SC-Net (RegNetY-6.4GF)', 'Song et al. 2026, Remote Sens.', 'yes', 87.77, 89.74, 79.76, None, None, None, 88.74),
    ],
    'UAV': [
        ('FCN (VGG16)', 'Xu et al. 2024, Sci. Data', 'no', 75.045, 84.016, 65.057, 77.456, 91.468, 86.724, None),
        ('U-Net (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 73.694, 86.394, 65.991, 78.019, 91.658, 87.136, None),
        ('DeepLabv3+ (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 89.289, 93.739, 84.261, 90.142, 96.721, 94.715, None),
        ('MFFENet (ResNet50)', 'Xu et al. 2024, Sci. Data', 'no', 89.326, 93.839, 84.375, 90.214, 96.746, 94.756, None),
        ('SegFormer', '2025, quoted by SC-Net', 'no', None, None, 87.80, None, None, None, None),
        ('U-Net (ResNet50)', 'Song et al. 2026 (SC-Net paper)', 'no', 91.78, 91.80, 84.82, None, None, None, 91.79),
        ('DeepLabV3+ (ResNet50)', 'Song et al. 2026 (SC-Net paper)', 'no', 92.76, 93.92, 87.51, None, None, None, 93.34),
        ('SC-Net (RegNetY-6.4GF)', 'Song et al. 2026, Remote Sens.', 'no', 94.82, 94.51, 89.89, None, None, None, 94.67),
    ],
}


def fmt(v):
    return '-' if v is None else f'{v:.2f}'


def describe_semantic_prior(margs):
    fm = arg(margs, 'fm')
    if not fm_enabled(fm):
        return 'none'
    mode = arg(margs, 'fm_mode')
    return f'`{fm}` ({mode}' + (f', {arg(margs, "fm_fusion")} fusion)' if mode == 'fuse' else ', training only)')


def results_markdown(res, subset):
    margs = res.get('model_args') or res['args']
    main_mode = res.get('tta_mode', 'TTA' if res.get('tta') else 'noTTA')
    lines = [f'# Results - {res["run"]}', '',
             f'Encoder `{arg(margs, "encoder")}`; semantic prior: {describe_semantic_prior(margs)}; '
             f'anchor mining: {arg(margs, "hard_mining")}; subset **{subset}**; test tiles: {res["n_test"]}; '
             f'selected weights: {res["weights"]}; main setting: {MODE_LABELS[main_mode]} at decision threshold '
             f't* = {res["threshold"]:.3f} ({res["threshold_source"]}).', '',
             'All values in %. F1 / IoU / P / R are for the **landslide** class (dataset-level pixel counts). '
             'mF1 = mean of landslide and background F1 (the "F1 score" reported by Xu et al. 2024); '
             'mIoU = mean of landslide and background IoU.', '',
             '| Method | Source | Same test split | P | R | F1 | IoU | mIoU | OA | mF1 | Kappa |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    if res.get('tta'):
        ours = [('test_noTTA@0.5', 'no TTA, t=0.5'), ('test_TTA@0.5', 'D4-TTA, t=0.5'), ('test_TTA@t*', 'D4-TTA, t=t*'),
                ('test_MSTTA@0.5', 'D4+MS-TTA, t=0.5'), ('test_MSTTA@t*', 'D4+MS-TTA, t=t*')]
    else:
        ours = [('test_noTTA@0.5', 'no TTA, t=0.5'), ('test_noTTA@t*', 'no TTA, t=t*')]
    for key, label in ours:
        m = res.get(key)
        if m is None:
            continue
        label = f'Ours, {label}' + (' (main)' if key == res.get('main_key') else '')
        lines.append(f'| **{label}** | this work | yes | {100 * m["precision"]:.2f} | {100 * m["recall"]:.2f} | '
                     f'{100 * m["f1"]:.2f} | **{100 * m["iou"]:.2f}** | {100 * m["miou"]:.2f} | {100 * m["oa"]:.2f} | '
                     f'{100 * m["mf1"]:.2f} | {100 * m["kappa"]:.2f} |')
    for (name, src, same, P, R, iou, miou, oa, mf1, f1) in BENCHMARKS.get(subset, []):
        if f1 is None and P is not None and R is not None:
            f1 = 2 * P * R / (P + R)
        if subset == 'UAV':
            same = 'no'
        lines.append(f'| {name} | {src} | {same} | {fmt(P)} | {fmt(R)} | {fmt(f1)} | {fmt(iou)} | {fmt(miou)} | '
                     f'{fmt(oa)} | {fmt(mf1)} | - |')
    lines += ['', 'Notes: Xu et al. (2024) and SegFormer used their own random 7:3 splits. For SAT, the SC-Net '
              'rows use the identical public test split; SC-Net selected its checkpoint on that test split, '
              'whereas this work selects on a validation split carved from the training tiles. '
              'For UAV no public split exists; our seeded split is written by prepare_data.py.', '']
    if res.get('per_region'):
        lines += ['## Per-region test metrics (main setting)', '',
                  '| Region | tiles | P | R | F1 | IoU |', '|---|---|---|---|---|---|']
        for rg, m in res['per_region'].items():
            lines.append(f'| {rg} | {m["tiles"]} | {100 * m["precision"]:.2f} | {100 * m["recall"]:.2f} | '
                         f'{100 * m["f1"]:.2f} | {100 * m["iou"]:.2f} |')
        lines.append('')
    val_modes = [m for m in ('TTA', 'MSTTA', 'noTTA') if res.get(f'val_{m}@0.5')]
    if not val_modes and res.get('val_at_0.5'):  # results written by the SP-UNet version
        res = {**res, f'val_{main_mode}@0.5': res['val_at_0.5'], f'val_{main_mode}@t*': res.get('val_at_t*')}
        val_modes = [main_mode]
    for m in val_modes:
        v, vt = res[f'val_{m}@0.5'], res.get(f'val_{m}@t*') or res[f'val_{m}@0.5']
        t_m = (res.get('thresholds') or {}).get(m, res['threshold'])
        lines.append(f'Validation ({res["n_val"]} tiles, {MODE_LABELS[m]}): IoU {100 * v["iou"]:.2f} / '
                     f'F1 {100 * v["f1"]:.2f} at t=0.5; IoU {100 * vt["iou"]:.2f} / F1 {100 * vt["f1"]:.2f} '
                     f'at t*={t_m:.3f}.')
    if val_modes:
        lines.append('')
    st = res.get('model_stats') or {}
    if st:
        size = arg(margs, 'img_size')
        txt = (f'Model: {st.get("params_M", 0):.2f} M trainable parameters (incl. '
               f'{st.get("train_only_heads_M", 0):.2f} M in training-only heads)')
        if st.get('fm_frozen_params_M'):
            txt += f' + {st["fm_frozen_params_M"]:.1f} M frozen foundation-model parameters'
        if st.get('fm_training_only_params_M'):
            txt += f' (+ {st["fm_training_only_params_M"]:.1f} M frozen foundation-model parameters used only in training)'
        if 'inference_params_M' in st:
            txt += f'; {st["inference_params_M"]:.2f} M parameters used at inference'
        txt += (f'; {st.get("gmacs", float("nan")):.1f} GMACs / {st.get("gflops", float("nan")):.1f} GFLOPs '
                f'at {size}x{size}')
        if 'fps_batch8' in st:
            txt += f', {st["fps_batch8"]:.1f} img/s (batch 8, FP16, {st.get("gpu", "")}, single pass)'
        lines += [txt + '.', '']
    return '\n'.join(lines)


def _ths(t):
    return [0.5] if abs(float(t) - 0.5) < 1e-9 else [0.5, float(t)]


def write_per_tile(path, rows, tiles):
    """Per-tile confusion counts of every evaluated mode at its own t* (input of paired significance tests)."""
    modes = list(tiles)
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name', 'region'] + [f'{m}_{k}' for m in modes for k in ('tp', 'fp', 'fn', 'tn')])
        for i, r in enumerate(rows):
            w.writerow([r['name'], r['region']] + [int(round(float(tiles[m][i, j]))) for m in modes for j in range(4)])
    os.replace(tmp, path)


def final_evaluation(a, d, model, rows, loaders, out_dir, info):
    """TTA mode and threshold chosen on validation (if allowed); test with no TTA, D4-TTA and (--ms-tta)
    D4+MS-TTA in one pass; per-region and per-tile counts; results.json / results.md / per_tile_test.csv."""
    use_tta = not a.no_tta
    ms = use_tta and bool(arg(a, 'ms_tta'))
    sel_modes = (['TTA'] + (['MSTTA'] if ms else [])) if use_tta else ['noTTA']
    regions = sorted({r['region'] for r in rows['test']})
    rid_map = {rg: i for i, rg in enumerate(regions)}
    test_rids = torch.tensor([rid_map[r['region']] for r in rows['test']], dtype=torch.long)
    res = {'run': out_dir.name, 'tag': arg(a, 'tag') or out_dir.name, 'args': vars(a),
           'n_test': len(rows['test']), 'n_val': len(rows['val']), **info}
    thr = {m: float(a.threshold) for m in sel_modes}
    thr_src, val_best = 'fixed (--threshold)', {}
    if loaders.get('val') is not None and info.get('select_threshold', False):
        ev = evaluate(model, loaders['val'], d, a, modes=sel_modes,
                      thresholds={m: SWEEP_THRESHOLDS for m in sel_modes})
        i05 = SWEEP_THRESHOLDS.index(0.5)
        for m in sel_modes:
            c = ev['counts'][m]
            sweep = [metrics_from_counts(*c[i].tolist()) for i in range(len(SWEEP_THRESHOLDS))]
            best = max(range(len(SWEEP_THRESHOLDS)),
                       key=lambda i: (round(sweep[i]['iou'], 6), -abs(SWEEP_THRESHOLDS[i] - 0.5)))
            thr[m], val_best[m] = float(SWEEP_THRESHOLDS[best]), sweep[best]['iou']
            res[f'val_sweep_{m}'] = {f'{t:.3f}': {'iou': s['iou'], 'f1': s['f1']}
                                     for t, s in zip(SWEEP_THRESHOLDS, sweep)}
            res[f'val_{m}@0.5'], res[f'val_{m}@t*'] = sweep[i05], sweep[best]
        m0 = sel_modes[0]
        res['val_sweep'], res['val_at_0.5'], res['val_at_t*'] = (res[f'val_sweep_{m0}'], res[f'val_{m0}@0.5'],
                                                                 res[f'val_{m0}@t*'])
        thr_src = 'best validation IoU' + (' with TTA' if use_tta else '')
    main = sel_modes[0]
    if ms and val_best and val_best['MSTTA'] > val_best['TTA']:
        main = 'MSTTA'
    if ms:
        res['tta_mode_source'] = ('D4+MS-TTA beat D4-TTA on validation' if main == 'MSTTA' else
                                  'D4-TTA >= D4+MS-TTA on validation' if val_best else
                                  'D4-TTA (no validation split to choose on)')
    res['threshold'], res['threshold_source'], res['thresholds'] = thr[main], thr_src, dict(thr)
    log(d, f'[eval] main setting {MODE_LABELS[main]}, decision threshold t* = {thr[main]:.3f} ({thr_src})')
    test_modes = (['noTTA'] if use_tta else []) + sel_modes
    ths = {m: _ths(thr.get(m, thr[sel_modes[0]])) for m in test_modes}  # no-TTA rows use the TTA threshold
    sink = VisSink(out_dir / 'vis_test', rows['test'], a.save_vis, thr[main], d.main)
    ev = evaluate(model, loaders['test'], d, a, modes=test_modes, thresholds=ths, region_ids=test_rids,
                  n_regions=len(regions), sink=sink, sink_mode=main, per_tile=True)
    for m in test_modes:
        c = ev['counts'][m]
        res[f'test_{m}@0.5'] = metrics_from_counts(*c[0].tolist())
        res[f'test_{m}@t*'] = metrics_from_counts(*c[-1].tolist())
    res['tta'], res['tta_mode'] = use_tta, main
    res['main_key'] = f'test_{main}@t*'
    res['main'] = res[res['main_key']]
    res['ablation_key'] = f'test_{sel_modes[0]}@t*'  # the same setting for every ablation row
    n_tiles = {rg: sum(1 for r in rows['test'] if r['region'] == rg) for rg in regions}
    res['per_region_by_mode'] = {}
    for m in test_modes:
        pr = {}
        for rg, i in rid_map.items():
            pr[rg] = metrics_from_counts(*ev['regions'][m][i, -1].tolist())
            pr[rg]['tiles'] = int(n_tiles[rg])
        res['per_region_by_mode'][m] = pr
    res['per_region'] = res['per_region_by_mode'][main]
    if d.main:
        write_per_tile(out_dir / 'per_tile_test.csv', rows['test'], ev['tiles'])
        res['per_tile_file'] = 'per_tile_test.csv'
        try:
            res['model_stats'] = model_stats(model, a.img_size, d.device, a.amp, a.channels_last)
        except Exception as e:  # noqa: BLE001
            res['model_stats'] = {'error': str(e)[:200]}
        subset = info.get('subset', 'SAT')
        write_json(res, out_dir / 'results.json')
        md = results_markdown(res, subset)
        (out_dir / 'results.md').write_text(md, encoding='utf-8')
        print('\n' + md, flush=True)
    d.barrier()
    return res


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='DPF-Net (Dual-Prior Fusion Network) for CAS Landslide segmentation')
    p.add_argument('--splits', required=True, help='folder with train/val/trainval/test.csv (prepare_data.py)')
    p.add_argument('--out', required=True, help='output folder of this run')
    g = p.add_argument_group('model')
    g.add_argument('--encoder', default='convnext_tiny.dinov3_lvd1689m')
    g.add_argument('--no-pretrained', action='store_true', help='random encoder init (smoke tests only)')
    g.add_argument('--no-hires', action='store_true', help='drop the high-resolution stem (implies --no-prior)')
    g.add_argument('--no-prior', action='store_true', help='drop the spectral-structural prior')
    g.add_argument('--no-attention', action='store_true', help='drop scSE attention in the decoder')
    g.add_argument('--boundary-weight', type=float, default=0.3, help='0 disables the boundary head')
    g.add_argument('--contrast-weight', type=float, default=0.1, help='0 disables the contrastive head')
    g.add_argument('--contrast-temp', type=float, default=0.1)
    g.add_argument('--contrast-anchors', type=int, default=256, help='anchors per class per GPU')
    g.add_argument('--aux-weights', type=float, nargs=2, default=[0.4, 0.2], help='deep supervision 1/4, 1/8')
    g.add_argument('--grad-ckpt', action='store_true', help='gradient checkpointing in the encoder')
    g = p.add_argument_group('foundation-model semantic prior')
    g.add_argument('--fm', default='vit_large_patch16_dinov3.sat493m',
                   help="frozen timm ViT used as semantic prior ('none' = SP-UNet), e.g. "
                        'vit_large_patch16_dinov3.lvd1689m (web images) or vit_base_patch16_dinov3.lvd1689m')
    g.add_argument('--fm-mode', choices=['fuse', 'distill'], default='fuse',
                   help='fuse: gated injection of the FM features (FM needed at inference); distill: FM used only '
                        'during training (feature distillation + semantic mining), no inference cost')
    g.add_argument('--fm-fusion', choices=['gated', 'add'], default='gated',
                   help='gated: structure-aware gated injection; add: plain residual addition (ablation)')
    g.add_argument('--fm-size', type=int, default=0,
                   help='FM input size for img-size tiles (0 = img-size; rounded to the patch size), e.g. 384')
    g.add_argument('--fm-layers', type=int, default=4, help='evenly spaced ViT blocks whose features are used')
    g.add_argument('--fm-channels', type=int, default=256, help='width of the FM adapter')
    g.add_argument('--hard-mining', choices=['auto', 'error', 'fm', 'error+fm', 'random'], default='auto',
                   help="anchor mining of the pixel contrastive loss; auto = 'error+fm' with a foundation model, "
                        "otherwise 'error'")
    g.add_argument('--distill-weight', type=float, default=0.5, help='feature-distillation weight (--fm-mode distill)')
    g = p.add_argument_group('training')
    g.add_argument('--img-size', type=int, default=512)
    g.add_argument('--epochs', type=int, default=80)
    g.add_argument('--batch-size', type=int, default=8, help='per GPU')
    g.add_argument('--workers', type=int, default=2, help='DataLoader workers per process')
    g.add_argument('--enc-lr', type=float, default=1e-4)
    g.add_argument('--dec-lr', type=float, default=4e-4)
    g.add_argument('--weight-decay', type=float, default=0.05)
    g.add_argument('--warmup-epochs', type=float, default=2.0)
    g.add_argument('--min-lr-ratio', type=float, default=0.01)
    g.add_argument('--ema-decay', type=float, default=0.9995)
    g.add_argument('--clip-grad', type=float, default=1.0)
    g.add_argument('--no-amp', action='store_true')
    g.add_argument('--no-channels-last', action='store_true')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--full-train', action='store_true',
                   help='train on train+val with a fixed schedule; evaluate the final EMA weights at --threshold')
    g.add_argument('--threshold', type=float, default=0.5, help='decision threshold when not selected on val')
    g.add_argument('--max-train-hours', type=float, default=10.5, help='stop + checkpoint before this budget')
    g.add_argument('--resume', default='', help='last.pth of a previous (possibly interrupted) run')
    g.add_argument('--val-every', type=int, default=1)
    g.add_argument('--track-test', action='store_true',
                   help='also log test IoU every epoch (reporting only - never used for selection)')
    g.add_argument('--log-every', type=int, default=50)
    g.add_argument('--tag', default='', help='experiment id stored in results.json (default: the --out folder name)')
    g.add_argument('--no-eval-on-timeout', action='store_true',
                   help='if the time budget stops training, save last.pth and skip the final evaluation '
                        '(used by ablation.py; the run is resumed in the next session)')
    g.add_argument('--stop-after-epochs', type=int, default=0,
                   help='testing: behave as if the time budget ran out after N epochs of this invocation')
    g = p.add_argument_group('augmentation')
    g.add_argument('--aug-affine-p', type=float, default=0.5)
    g.add_argument('--aug-rotate', type=float, default=30.0)
    g.add_argument('--aug-scale-min', type=float, default=0.75)
    g.add_argument('--aug-scale-max', type=float, default=1.33)
    g.add_argument('--aug-translate', type=float, default=0.1)
    g.add_argument('--aug-color-p', type=float, default=0.8)
    g.add_argument('--aug-brightness', type=float, default=0.2)
    g.add_argument('--aug-contrast', type=float, default=0.2)
    g.add_argument('--aug-saturation', type=float, default=0.2)
    g.add_argument('--aug-channel-gain', type=float, default=0.05)
    g.add_argument('--aug-gamma-p', type=float, default=0.3)
    g.add_argument('--aug-noise-p', type=float, default=0.2)
    g.add_argument('--aug-noise-std', type=float, default=0.02)
    g = p.add_argument_group('evaluation')
    g.add_argument('--no-tta', action='store_true')
    g.add_argument('--ms-tta', action='store_true',
                   help='also evaluate D4 + multi-scale TTA; the main setting (D4 or D4+MS) is chosen on validation')
    g.add_argument('--ms-scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    g.add_argument('--eval-only', action='store_true', help='skip training; evaluate --eval-ckpts (or <out>/best.pth)')
    g.add_argument('--eval-ckpts', nargs='*', default=[], help='several checkpoints = probability ensemble')
    g.add_argument('--save-vis', type=int, default=24, help='qualitative test panels to save (rank 0)')
    g.add_argument('--debug-limit', type=int, default=0, help='use only the first N tiles of every split (tests)')
    a = p.parse_args(argv)
    if a.no_hires and not a.no_prior:
        a.no_prior = True
    a.fm = 'none' if str(a.fm).strip().lower() in ('', 'none', 'null', 'off', 'no') else str(a.fm).strip()
    if a.hard_mining == 'auto':
        a.hard_mining = 'error+fm' if fm_enabled(a.fm) else 'error'
    elif a.hard_mining in ('fm', 'error+fm') and not fm_enabled(a.fm):
        p.error(f'--hard-mining {a.hard_mining} needs a foundation model (--fm)')
    if a.fm_mode == 'distill' and not fm_enabled(a.fm):
        p.error('--fm-mode distill needs a foundation model (--fm)')
    if a.fm_size < 0 or a.fm_layers < 1 or a.fm_channels < 8:
        p.error('--fm-size must be >= 0, --fm-layers >= 1, --fm-channels >= 8')
    a.amp = (not a.no_amp) and torch.cuda.is_available()
    a.channels_last = (not a.no_channels_last) and torch.cuda.is_available()
    return a


def load_rows(a):
    sp = Path(a.splits)
    rows = {'train': read_split(sp / ('trainval.csv' if a.full_train else 'train.csv')),
            'val': [] if a.full_train or not (sp / 'val.csv').exists() else read_split(sp / 'val.csv'),
            'test': read_split(sp / 'test.csv')}
    if a.debug_limit:
        rows = {k: v[:a.debug_limit] for k, v in rows.items()}
    subset = 'SAT'
    if (sp / 'summary.json').exists():
        subset = json.loads((sp / 'summary.json').read_text(encoding='utf-8')).get('subset', 'SAT')
    return rows, subset


def make_eval_loader(rows, a, d, persistent=False):
    if not rows:
        return None
    ds = SegDataset(rows, a.img_size, train=False)
    return DataLoader(ds, batch_size=a.batch_size, sampler=ShardSampler(len(ds), d.rank, d.world),
                      num_workers=a.workers, pin_memory=d.device.type == 'cuda', drop_last=False,
                      persistent_workers=persistent and a.workers > 0)


def eval_only(a, d, rows, subset, out_dir):
    ckpts = a.eval_ckpts or [str(out_dir / 'best.pth')]
    models, full_train_any, names, member_args = [], False, [], []
    for c in ckpts:
        sd = torch.load(c, map_location='cpu', weights_only=True)
        margs = argparse.Namespace(**sd['args'])
        if margs.img_size != a.img_size:
            log(d, f'WARNING: {c} was trained at {margs.img_size}px, evaluating at {a.img_size}px')
        m = build_model(margs, pretrained=False)
        m.load_state_dict(sd['model'])
        models.append(m)
        fm = arg(margs, 'fm')
        names.append(f'{Path(c).parent.name}/{Path(c).name} ({margs.encoder}'
                     + (f' + {fm} [{arg(margs, "fm_mode")}]' if fm_enabled(fm) else '') + ')')
        member_args.append({**{k: LEGACY_DEFAULTS[k] for k in LEGACY_DEFAULTS if k not in sd['args']}, **sd['args']})
        full_train_any |= bool(getattr(margs, 'full_train', False))
    model = models[0] if len(models) == 1 else Ensemble(models)
    model = model.to(d.device).eval()
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    loaders = {'val': make_eval_loader(rows['val'], a, d), 'test': make_eval_loader(rows['test'], a, d)}
    info = {'subset': subset, 'weights': 'ensemble of ' + ', '.join(names) if len(names) > 1 else names[0],
            'select_threshold': not full_train_any and loaders['val'] is not None, 'schedule_complete': None,
            'eval_only': True, 'checkpoints': [str(c) for c in ckpts]}
    if len(member_args) == 1:
        info['model_args'] = member_args[0]  # the trained model's settings (results 'args' = this invocation)
    else:
        info['member_args'] = member_args
    log(d, f'[eval-only] {info["weights"]}')
    final_evaluation(a, d, model, rows, loaders, out_dir, info)
    if d.main:
        write_json({'tag': a.tag or out_dir.name, 'schedule_complete': None, 'evaluated': True, 'eval_only': True},
                   out_dir / 'status.json')


def train(a, d, rows, subset, out_dir):
    t_start = time.time()
    has_val = len(rows['val']) > 0
    split_h = rows_hash(rows['train'], rows['val'], rows['test'])

    # ---- model (rank 0 downloads the pretrained weights first) ----
    model = None
    if d.main:
        model = build_model(a, pretrained=not a.no_pretrained)
    d.barrier()
    if not d.main:
        model = build_model(a, pretrained=not a.no_pretrained)
    if d.enabled and d.device.type == 'cuda':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(d.device)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    ema = ModelEMA(model, a.ema_decay)
    optimizer = build_optimizer(model, a)
    scaler = torch.amp.GradScaler(d.device.type, enabled=a.amp)

    # ---- data ----
    ds_train = SegDataset(rows['train'], a.img_size, train=True)
    sampler = DistributedSampler(ds_train, num_replicas=d.world, rank=d.rank, shuffle=True, seed=a.seed,
                                 drop_last=True)
    dl_train = DataLoader(ds_train, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                          pin_memory=d.device.type == 'cuda', drop_last=True, persistent_workers=a.workers > 0)
    dl_val = make_eval_loader(rows['val'], a, d, persistent=True) if has_val else None
    dl_test_track = make_eval_loader(rows['test'], a, d, persistent=True) if a.track_test else None
    steps_per_epoch = len(dl_train)
    if steps_per_epoch == 0:
        raise SystemExit('Training split smaller than one global batch.')
    total_steps = a.epochs * steps_per_epoch
    warmup_steps = min(int(a.warmup_epochs * steps_per_epoch), max(total_steps - 1, 0))

    # ---- resume ----
    start_epoch, step, best_iou, best_epoch, history, prev_times = 0, 0, -1.0, 0, [], []
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=True)
        old = ck['args']
        for k in ARCH_KEYS + ('full_train', 'hard_mining'):
            if arg(old, k) != arg(a, k):
                raise SystemExit(f'--resume: argument "{k}" differs ({arg(old, k)} vs {arg(a, k)}).')
        for k in ('boundary_weight', 'contrast_weight'):
            if (old.get(k, 0) > 0) != (getattr(a, k) > 0):
                raise SystemExit(f'--resume: head "{k}" enabled/disabled differently from the checkpoint.')
        if (any(w > 0 for w in old.get('aux_weights', [0, 0]))) != any(w > 0 for w in a.aux_weights):
            raise SystemExit('--resume: deep supervision enabled/disabled differently from the checkpoint.')
        if ck.get('split_hash') != split_h:
            log(d, 'WARNING: the data split differs from the checkpoint run (split hash mismatch).')
        if old.get('epochs') != a.epochs or old.get('steps_per_epoch') not in (None, steps_per_epoch):
            log(d, 'WARNING: epochs / steps-per-epoch changed - the LR schedule continues from the saved step.')
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        for gr in optimizer.param_groups:  # keep the base LRs of this invocation
            gr['base_lr'] = a.enc_lr if gr.get('group_name', '').startswith('encoder') else a.dec_lr
        if a.amp and ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        start_epoch, step = int(ck['epoch']), int(ck['step'])
        best_iou, best_epoch = float(ck['best_iou']), int(ck['best_epoch'])
        history = list(ck.get('history', []))
        prev_times = [float(h['seconds']) for h in history[-3:] if h.get('seconds')]
        if d.main:
            prev_best = Path(a.resume).parent / 'best.pth'
            if prev_best.exists() and not (out_dir / 'best.pth').exists():
                shutil.copy2(prev_best, out_dir / 'best.pth')
        log(d, f'[resume] {a.resume}: continuing at epoch {start_epoch + 1}/{a.epochs} (step {step}), '
               f'best val IoU {100 * best_iou:.2f} @ epoch {best_epoch}')
        del ck

    net = model
    if d.enabled:
        net = nn.parallel.DistributedDataParallel(
            model, device_ids=[d.local_rank] if d.device.type == 'cuda' else None,
            broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)

    pr = param_report(model)
    log(d, f'[train] encoder {a.encoder} | {pr["params_M"]:.2f} M trainable params | hires={not a.no_hires} '
           f'prior={not a.no_prior} boundary={a.boundary_weight} contrast={a.contrast_weight} '
           f'(mining {a.hard_mining}) aux={a.aux_weights}')
    log(d, f'[train] semantic prior: {describe_semantic_prior(a)}'
           + (f' | {pr["fm_frozen_params_M"] + pr["fm_training_only_params_M"]:.1f} M frozen parameters, '
              f'blocks {model.fm_idx}, input scale {model.fm_scale:g}' if model.fm_name else ''))
    log(d, f'[train] tiles train/val/test = {len(rows["train"])}/{len(rows["val"])}/{len(rows["test"])} | '
           f'world {d.world} x batch {a.batch_size} = {d.world * a.batch_size} | {steps_per_epoch} steps/epoch | '
           f'{a.epochs} epochs | AMP {a.amp} | channels_last {a.channels_last} | device {d.device}')

    epoch_times, stopped_early = list(prev_times), False  # previous session's epoch times seed the time guard
    names = ['loss', 'seg', 'aux4', 'aux8', 'bnd', 'con', 'dis']
    for epoch in range(start_epoch, a.epochs):
        elapsed = time.time() - t_start
        est = float(np.mean(epoch_times[-3:])) if epoch_times else 0.0
        if d.any_flag(elapsed + 1.05 * est > a.max_train_hours * 3600):
            log(d, f'[train] time budget ({a.max_train_hours} h) reached before epoch {epoch + 1}; '
                   'stopping. Resume later with --resume <out>/last.pth')
            stopped_early = True
            break
        t_ep = time.time()
        sampler.set_epoch(epoch)
        net.train()
        acc = torch.zeros(len(names), device=d.device, dtype=torch.float64)
        n_ok = torch.zeros((), device=d.device, dtype=torch.float64)
        n_skip = 0
        for it, (img, mask, _) in enumerate(dl_train):
            x = img.to(d.device, non_blocking=True).float().div_(255.0)
            y = mask.to(d.device, non_blocking=True).float()
            x, y = gpu_augment(x, y, a)
            if a.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            f = lr_factor(step, total_steps, warmup_steps, a.min_lr_ratio)
            for gr in optimizer.param_groups:
                gr['lr'] = gr['base_lr'] * f
            with torch.autocast(device_type=d.device.type, dtype=torch.float16, enabled=a.amp):
                out = net(x)
            loss, parts = compute_loss(out, y, a)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            scale_before = scaler.get_scale() if a.amp else 1.0
            scaler.step(optimizer)
            scaler.update()
            if a.amp and scaler.get_scale() < scale_before:
                n_skip += 1
            ema.update(model)
            step += 1
            vals = torch.cat([loss.detach().reshape(1), parts]).double()
            ok = torch.isfinite(vals).all()
            acc += torch.where(ok, vals, torch.zeros_like(vals))
            n_ok += ok.double()
            if d.main and a.log_every and (it + 1) % a.log_every == 0:
                m = (acc / n_ok.clamp(min=1)).tolist()
                print(f'  ep {epoch + 1} it {it + 1}/{steps_per_epoch} loss {m[0]:.4f} (seg {m[1]:.4f}) '
                      f'lr {optimizer.param_groups[0]["lr"]:.2e} | {(time.time() - t_ep) / (it + 1):.2f} s/it',
                      flush=True)
        d.sum_(acc)
        d.sum_(n_ok)
        tr = {k: v for k, v in zip(names, (acc / n_ok.clamp(min=1)).tolist())}

        val_m = None
        if dl_val is not None and ((epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs):
            c = evaluate(ema.module, dl_val, d, a, modes=('noTTA',))['counts']['noTTA']
            val_m = metrics_from_counts(*c[0].tolist())
        test_m = None
        if dl_test_track is not None:
            c = evaluate(ema.module, dl_test_track, d, a, modes=('noTTA',))['counts']['noTTA']
            test_m = metrics_from_counts(*c[0].tolist())
        improved = val_m is not None and val_m['iou'] > best_iou
        if improved:
            best_iou, best_epoch = val_m['iou'], epoch + 1
        epoch_times.append(time.time() - t_ep)
        rec = {'epoch': epoch + 1, 'lr_dec': optimizer.param_groups[0]['lr'], 'train': tr,
               'amp_skipped_steps': n_skip, 'amp_scale': scaler.get_scale() if a.amp else None,
               'val': val_m, 'test_tracked': test_m, 'seconds': epoch_times[-1]}
        history.append(rec)
        if d.main:
            if improved:
                save_atomic({'model': ema.module.state_dict(), 'args': vars(a), 'epoch': epoch + 1, 'val': val_m,
                             'split_hash': split_h}, out_dir / 'best.pth')
            save_atomic({'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict() if a.amp else None, 'epoch': epoch + 1, 'step': step,
                         'best_iou': best_iou, 'best_epoch': best_epoch, 'history': history,
                         'args': {**vars(a), 'steps_per_epoch': steps_per_epoch}, 'split_hash': split_h},
                        out_dir / 'last.pth')
            write_json(history, out_dir / 'history.json')
            msg = (f'[ep {epoch + 1:3d}/{a.epochs}] loss {tr["loss"]:.4f} (seg {tr["seg"]:.4f} bnd {tr["bnd"]:.4f} '
                   f'con {tr["con"]:.4f}' + (f' dis {tr["dis"]:.4f}' if model.distill_head is not None else '')
                   + f') lr {optimizer.param_groups[0]["lr"]:.2e}')
            if val_m:
                msg += (f' | val IoU {100 * val_m["iou"]:.2f} F1 {100 * val_m["f1"]:.2f} P {100 * val_m["precision"]:.1f} '
                        f'R {100 * val_m["recall"]:.1f} | best {100 * best_iou:.2f}@{best_epoch}')
            if test_m:
                msg += f' | test(track) IoU {100 * test_m["iou"]:.2f}'
            if n_skip:
                msg += f' | amp-skipped {n_skip}'
            if model.fm_fallbacks:
                msg += f' | FM FP32 fallbacks {model.fm_fallbacks}'
            mean_ep = float(np.mean(epoch_times[-3:]))
            msg += f' | {epoch_times[-1] / 60:.1f} min/ep, ETA {mean_ep * (a.epochs - epoch - 1) / 3600:.2f} h'
            print(msg, flush=True)
            if epoch == start_epoch:
                proj = (time.time() - t_start) + mean_ep * (a.epochs - epoch - 1)
                print(f'[train] projected total training time {proj / 3600:.2f} h (budget {a.max_train_hours} h)'
                      + ('  -> will stop early; resume in a new session with --resume' if proj > a.max_train_hours * 3600 else ''),
                      flush=True)
        d.barrier()
        if a.stop_after_epochs and epoch + 1 - start_epoch >= a.stop_after_epochs and epoch + 1 < a.epochs:
            log(d, f'[train] --stop-after-epochs {a.stop_after_epochs}: stopping as if out of time')
            stopped_early = True
            break

    # ---- choose the weights to evaluate ----
    schedule_complete = not stopped_early and (len(history) == 0 or history[-1]['epoch'] == a.epochs)
    status = {'tag': a.tag or out_dir.name, 'schedule_complete': schedule_complete,
              'epochs_done': history[-1]['epoch'] if history else 0, 'epochs': a.epochs, 'evaluated': False,
              'train_hours_total': sum(float(h.get('seconds') or 0.0) for h in history) / 3600}
    if not schedule_complete and a.no_eval_on_timeout:
        if d.main:
            write_json(status, out_dir / 'status.json')
        log(d, f'[train] stopped by the time budget after epoch {status["epochs_done"]}/{a.epochs}; final evaluation '
               f'skipped (--no-eval-on-timeout). Resume with --resume {out_dir / "last.pth"}')
        return
    if has_val and (out_dir / 'best.pth').exists():
        ck = torch.load(out_dir / 'best.pth', map_location='cpu', weights_only=True)
        ema.module.load_state_dict(ck['model'])
        weights = f'EMA weights of epoch {ck["epoch"]} (best validation IoU {100 * ck["val"]["iou"]:.2f})'
        del ck
    else:
        weights = f'final EMA weights (epoch {history[-1]["epoch"] if history else 0})'
        if d.main:
            save_atomic({'model': ema.module.state_dict(), 'args': vars(a),
                         'epoch': history[-1]['epoch'] if history else 0, 'val': None, 'split_hash': split_h},
                        out_dir / 'final.pth')
    d.barrier()
    log(d, f'[eval] evaluating {weights}')
    loaders = {'val': dl_val, 'test': make_eval_loader(rows['test'], a, d)}
    info = {'subset': subset, 'weights': weights, 'schedule_complete': schedule_complete,
            'select_threshold': has_val and not a.full_train,
            'train_hours': (time.time() - t_start) / 3600, 'train_hours_total': status['train_hours_total'],
            'epochs_done': status['epochs_done']}
    final_evaluation(a, d, ema.module, rows, loaders, out_dir, info)
    if d.main:
        write_json({**status, 'evaluated': True}, out_dir / 'status.json')
    if not schedule_complete:
        log(d, f'NOTE: the {a.epochs}-epoch schedule is incomplete; resume with --resume {out_dir / "last.pth"}')


def main(argv=None):
    a = parse_args(argv)
    d = Dist()
    try:
        set_seed(a.seed + 1000 * d.rank)
        if d.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        out_dir = Path(a.out)
        if d.main:
            out_dir.mkdir(parents=True, exist_ok=True)
        d.barrier()
        rows, subset = load_rows(a)
        if a.eval_only:
            eval_only(a, d, rows, subset, out_dir)
        else:
            train(a, d, rows, subset, out_dir)
        d.barrier()
    finally:
        d.close()


if __name__ == '__main__':
    main()
