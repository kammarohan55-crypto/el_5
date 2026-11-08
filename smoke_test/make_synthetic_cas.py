#!/usr/bin/env python3
"""Creates a small synthetic copy of the CAS Landslide folder layout (Zenodo 10294997 / Kaggle mirror)
for local tests: real SC-Net SAT tile names, full-width parentheses in folder names, decoy folders,
hidden files, a few RGBA / 16-bit tiles, and 7 UAV regions with clashing tile names."""
import argparse
import csv
import io
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SCNET = 'https://raw.githubusercontent.com/LiaoChengCRCC/SC-Net/HEAD/datasets/{}.csv'
SAT_DIRS = [('Haiti_Planet', 'Tiburon Peninsula（planet）'), ('Haiti', 'Tiburon Peninsula（Sentinel）'),
            ('Hokkaido', 'Hokkaido Iburi-Tobu'), ('Lombok', 'Lombok'), ('LongxiheSAT', 'Longxi River(SAT)'),
            ('Mengdong', 'Mengdong Township'), ('Moxitaidi_Satellite', 'Moxitaidi (SAT)'), ('Palu', 'palu'),
            ('wenchuan', 'Wenchuan')]
UAV_DIRS = [('Jiuzhai', 'Jiuzhai valley (UAV-0.2m)'), ('Jiuzhai', 'Jiuzhai valley (UAV-0.5m)'),
            ('LongxiheUAV', 'Longxi River（UAV）'), ('Moxi', 'Moxi town（UAV-0.2m）'), ('Moxi', 'Moxi town（UAV-1m）'),
            ('Moxitaidi', 'Moxitaidi (UAV-0.6m)'), ('Moxitaidi', 'Moxitaidi (UAV-1m)')]


def write_img(path, arr):
    ok, buf = cv2.imencode('.tif', arr)
    assert ok, path
    buf.tofile(str(path))


def tile(rng, size):
    """Green/grey terrain with brown landslide ellipses; returns (BGR uint8, mask 0/1)."""
    base = np.zeros((size, size, 3), np.float32)
    base[..., 1] = rng.uniform(70, 140)
    base[..., 2] = rng.uniform(40, 90)
    base[..., 0] = rng.uniform(30, 80)
    base += rng.normal(0, 12, base.shape)
    mask = np.zeros((size, size), np.uint8)
    for _ in range(rng.integers(1, 4)):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        ax = (int(rng.integers(3, size // 3)), int(rng.integers(3, size // 4)))
        cv2.ellipse(mask, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    soil = np.array([60, 110, 160], np.float32) + rng.normal(0, 15, 3)  # BGR brownish
    img = np.where(mask[..., None] > 0, soil + rng.normal(0, 18, base.shape), base)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='becomes the fake /kaggle/input')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--uav-per-region', type=int, default=30)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    root = Path(a.root)
    cas = root / 'cas-landslide' / '10294997'
    csv_dir = root.parent / 'scnet_csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for split in ('train', 'test'):
        text = urllib.request.urlopen(SCNET.format(split), timeout=60).read().decode('utf-8-sig')
        (csv_dir / f'{split}.csv').write_text(text, encoding='utf-8')
        stems += [Path(r['img']).stem for r in csv.DictReader(io.StringIO(text))]
    print(f'{len(stems)} SC-Net SAT tile names')

    count = 0
    for prefix, folder in SAT_DIRS:
        for sub in ('img', 'mask', 'label'):
            (cas / folder / sub).mkdir(parents=True, exist_ok=True)
    for s in stems:
        prefix, folder = next((pf, fd) for pf, fd in SAT_DIRS if s.startswith(pf))
        img, mask = tile(rng, a.size)
        if count == 5:      # one RGBA tile
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        if count == 6:      # one 16-bit tile
            img = (img.astype(np.uint16) * 257)
        write_img(cas / folder / 'img' / f'{s}.tif', img)
        write_img(cas / folder / 'mask' / f'{s}.tif', mask)
        if count % 50 == 0:
            write_img(cas / folder / 'label' / f'{s}.tif', mask * 255)
        count += 1
    # decoys
    write_img(cas / 'Tiburon Peninsula（Sentinel）' / 'img' / '._Haiti002.tif', np.zeros((4, 4), np.uint8))
    (cas / 'study areas shp' / 'shp').mkdir(parents=True, exist_ok=True)
    (cas / 'study areas shp' / 'shp' / 'areas.shp').write_bytes(b'fake')
    (root / 'cas-landslide' / 'rasters_dieu_kien').mkdir(parents=True, exist_ok=True)
    (cas / 'CAS Landslide Dataset README.html').write_text('<html></html>', encoding='utf-8')
    for prefix, folder in UAV_DIRS:
        for sub in ('img', 'mask'):
            (cas / folder / sub).mkdir(parents=True, exist_ok=True)
        for i in range(a.uav_per_region):
            img, mask = tile(rng, a.size)
            write_img(cas / folder / 'img' / f'{prefix}{i:04d}.tif', img)
            write_img(cas / folder / 'mask' / f'{prefix}{i:04d}.tif', mask)
    print(f'wrote {count} SAT tiles and {len(UAV_DIRS) * a.uav_per_region} UAV tiles below {cas}')
    print(f'SC-Net CSVs saved in {csv_dir}')


if __name__ == '__main__':
    main()
