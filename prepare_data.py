#!/usr/bin/env python3
"""
prepare_data.py - build train / val / test split CSVs for the CAS Landslide Dataset.

  SAT : the public SC-Net split (5,204 train / 2,218 test tiles) from
        https://github.com/LiaoChengCRCC/SC-Net/tree/main/datasets
        -> identical test set to SC-Net (Song et al., Remote Sens. 2026, 18(13), 2216).
  UAV : seeded random 7:3 split per region (13,443 tiles in 7 UAV regions),
        same protocol as Xu et al. (Sci. Data 2024). Save the CSVs and publish them.

A region-stratified validation subset (default 10 %) is carved out of the training
split.  The test split is never used for model selection.

Outputs  <out>/<SUBSET>/train.csv     training tiles (train minus val)
         <out>/<SUBSET>/val.csv       validation tiles (model selection / threshold)
         <out>/<SUBSET>/trainval.csv  train + val (for --full-train final runs)
         <out>/<SUBSET>/test.csv      held-out test tiles
         <out>/<SUBSET>/summary.json  counts per region, split hash, source folders
CSV columns: name, region, img, mask   (img / mask are absolute paths)

Works with the Zenodo record 10294997 folder layout
    <root>/<Region name>/{img,label,mask}/<tile>.tif
(e.g. the Kaggle mirror "ngxunninh/cas-landslide").  Region folders are found
automatically anywhere below --input-root.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
import unicodedata
import urllib.request
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

# normalised folder-name prefix -> readable region name
SAT_REGIONS = OrderedDict([
    ('tiburonpeninsulasentinel', 'Tiburon Peninsula (Sentinel)'),
    ('tiburonpeninsulaplanet', 'Tiburon Peninsula (Planet)'),
    ('hokkaidoiburitobu', 'Hokkaido Iburi-Tobu'),
    ('lombok', 'Lombok'),
    ('longxiriversat', 'Longxi River (SAT)'),
    ('mengdong', 'Mengdong'),
    ('moxitaidisat', 'Moxitaidi (SAT)'),
    ('palu', 'Palu'),
    ('wenchuan', 'Wenchuan'),
])
UAV_REGIONS = OrderedDict([
    ('jiuzhaivalleyuav02m', 'Jiuzhai Valley (UAV-0.2m)'),
    ('jiuzhaivalleyuav05m', 'Jiuzhai Valley (UAV-0.5m)'),
    ('longxiriveruav', 'Longxi River (UAV)'),
    ('moxitownuav02m', 'Moxi Town (UAV-0.2m)'),
    ('moxitownuav1m', 'Moxi Town (UAV-1m)'),
    ('moxitaidiuav06m', 'Moxitaidi (UAV-0.6m)'),
    ('moxitaidiuav1m', 'Moxitaidi (UAV-1m)'),
])
EXPECTED_SAT = {'train': 5204, 'test': 2218}
EXPECTED_UAV_TOTAL = 13443
SCNET_URLS = {
    'train': 'https://raw.githubusercontent.com/LiaoChengCRCC/SC-Net/HEAD/datasets/train.csv',
    'test': 'https://raw.githubusercontent.com/LiaoChengCRCC/SC-Net/HEAD/datasets/test.csv',
}
SKIP_DIRS = {'img', 'imgs', 'image', 'images', 'mask', 'masks', 'label', 'labels'}


def norm_name(s):
    """NFKC-normalise, case-fold and keep only alphanumerics (handles full-width parentheses)."""
    return ''.join(c for c in unicodedata.normalize('NFKC', s).casefold() if c.isalnum())


def find_region_dirs(input_root, regions, max_depth=8):
    """Breadth-first search for folders whose normalised name starts with a region key
    and that contain both an 'img' and a 'mask' sub-folder."""
    found = defaultdict(list)
    queue = deque([(Path(input_root), 0)])
    while queue:
        d, depth = queue.popleft()
        try:
            subdirs = [e for e in os.scandir(d) if e.is_dir()]
        except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
            continue
        lower = {e.name.lower() for e in subdirs}
        nd = norm_name(d.name)
        if 'img' in lower and 'mask' in lower:
            for key in regions:
                if nd.startswith(key):
                    found[key].append(d)
        if depth < max_depth:
            for e in subdirs:
                if e.name.lower() in SKIP_DIRS:
                    continue
                queue.append((Path(e.path), depth + 1))
    return found


def pick_dir(key, candidates):
    cands = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
    if len(cands) > 1:
        print(f'  WARNING: {len(cands)} folders match region "{key}", using the first:')
        for c in cands:
            print('     ', c)
    return cands[0]


def child_dir(region_dir, name):
    for e in os.scandir(region_dir):
        if e.is_dir() and e.name.lower() == name:
            return Path(e.path)
    raise FileNotFoundError(f'No "{name}" folder in {region_dir}')


def index_images(folder):
    """stem -> absolute path for all image files below folder (hidden files ignored)."""
    out, dups = {}, []
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if f.startswith('.'):
                continue
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXT:
                continue
            if stem in out:
                dups.append(stem)
            out[stem] = str(Path(dirpath, f).resolve())
    return out, dups


def index_region(region_dir):
    imgs, d1 = index_images(child_dir(region_dir, 'img'))
    masks, d2 = index_images(child_dir(region_dir, 'mask'))
    if d1 or d2:
        raise ValueError(f'Duplicate tile stems inside {region_dir}: {(d1 + d2)[:10]}')
    common = sorted(set(imgs) & set(masks))
    no_mask = sorted(set(imgs) - set(masks))
    no_img = sorted(set(masks) - set(imgs))
    return {s: (imgs[s], masks[s]) for s in common}, no_mask, no_img


def read_text(url_or_path, retries=3):
    if os.path.exists(url_or_path):
        return Path(url_or_path).read_text(encoding='utf-8-sig')
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url_or_path, timeout=60) as r:
                return r.read().decode('utf-8-sig')
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f'Could not download {url_or_path}: {last}. '
                       'Turn Internet ON, or pass --csv-dir with SC-Net train.csv/test.csv.')


def scnet_stems(split, csv_dir):
    src = str(Path(csv_dir) / f'{split}.csv') if csv_dir else SCNET_URLS[split]
    text = read_text(src)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or 'img' not in rows[0]:
        raise ValueError(f'{src}: expected a CSV with an "img" column')
    stems = [Path(r['img'].strip().replace('\\', '/')).stem for r in rows]
    if len(set(stems)) != len(stems):
        raise ValueError(f'{src}: duplicate tile names')
    return stems, src, text


def round_half_up(x):
    return int(math.floor(x + 0.5))


def make_row(name, region, pair):
    return {'name': name, 'region': region, 'img': pair[0], 'mask': pair[1]}


def stratified_val(rows, frac, seed):
    """Carve a region-stratified validation subset out of the training rows."""
    by_region = defaultdict(list)
    for r in rows:
        by_region[r['region']].append(r)
    fit, val = [], []
    for region in sorted(by_region):
        rr = sorted(by_region[region], key=lambda r: r['name'])
        random.Random(f'{seed}-val-{region}').shuffle(rr)
        n_val = round_half_up(len(rr) * frac) if frac > 0 else 0
        n_val = min(n_val, max(len(rr) - 1, 0))
        val += rr[:n_val]
        fit += rr[n_val:]
    return fit, val


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'region', 'img', 'mask'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def split_hash(splits):
    h = hashlib.sha256()
    for name in ('train', 'val', 'test'):
        for r in sorted(splits[name], key=lambda r: r['name']):
            h.update(f'{name}:{r["region"]}:{r["name"]}\n'.encode('utf-8'))
    return h.hexdigest()


def imread_any(path):
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def sanity_check(rows, n, seed):
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    shapes, dtypes, chans, mvals = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    fg, problems = [], []
    for r in sample:
        img, m = imread_any(r['img']), imread_any(r['mask'])
        if img is None or m is None:
            problems.append(f'unreadable: {r["img"]} / {r["mask"]}')
            continue
        shapes[str(img.shape[:2])] += 1
        dtypes[str(img.dtype)] += 1
        chans[1 if img.ndim == 2 else img.shape[2]] += 1
        mm = m[..., 0] if m.ndim == 3 else m
        for v in np.unique(mm)[:6].tolist():
            mvals[v] += 1
        if mm.shape != img.shape[:2]:
            problems.append(f'mask/image size mismatch: {r["name"]} {mm.shape} vs {img.shape[:2]}')
        thr = 0 if mm.max() <= 1 else 127
        fg.append(float((mm > thr).mean()))
    print(f'  sanity check on {len(sample)} random tiles:')
    print(f'    image sizes   : {dict(shapes)}')
    print(f'    image dtypes  : {dict(dtypes)}    channels: {dict(chans)}')
    print(f'    mask values   : {dict(mvals)}')
    if fg:
        print(f'    landslide px  : mean {100 * np.mean(fg):.2f} %  (min {100 * np.min(fg):.2f} %, max {100 * np.max(fg):.2f} %)')
    for p in problems[:10]:
        print('    PROBLEM:', p)
    bad_vals = [v for v in mvals if v not in (0, 1, 255)]
    if bad_vals:
        print(f'    NOTE: unexpected mask values {bad_vals[:6]}; masks are binarised as >0 (0/1) or >127 (0/255).')
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subset', choices=['SAT', 'UAV'], default='SAT')
    p.add_argument('--input-root', default='/kaggle/input', help='folder searched for the CAS region folders')
    p.add_argument('--out', default='/kaggle/working/splits')
    p.add_argument('--val-frac', type=float, default=0.1, help='fraction of each region\'s training tiles used for validation')
    p.add_argument('--test-frac', type=float, default=0.3, help='UAV only: test fraction per region')
    p.add_argument('--seed', type=int, default=2024, help='seed of the validation split (and of the UAV split)')
    p.add_argument('--csv-dir', default='', help='SAT only: local folder with SC-Net train.csv / test.csv (no download)')
    p.add_argument('--check-n', type=int, default=40, help='number of random tiles to sanity-check')
    p.add_argument('--allow-count-mismatch', action='store_true')
    a = p.parse_args(argv)

    t0 = time.time()
    regions = SAT_REGIONS if a.subset == 'SAT' else UAV_REGIONS
    print(f'[prepare] subset={a.subset}  searching region folders below {a.input_root} ...', flush=True)
    found = find_region_dirs(a.input_root, regions)
    missing = [k for k in regions if not found.get(k)]
    if missing:
        raise SystemExit(
            f'Could not find the {a.subset} region folders {missing} below {a.input_root}.\n'
            'Attach the CAS Landslide dataset (e.g. Kaggle "ngxunninh/cas-landslide" or the Zenodo 10294997 zips '
            'extracted) so that each region folder contains img/ and mask/ sub-folders.')

    pairs, region_dir = {}, {}
    for key, rname in regions.items():
        d = pick_dir(key, found[key])
        region_dir[rname] = str(d)
        pr, no_mask, no_img = index_region(d)
        pairs[rname] = pr
        msg = f'  {rname:30s} {len(pr):6d} tiles   <- {d}'
        if no_mask or no_img:
            msg += f'   (ignored: {len(no_mask)} images without mask, {len(no_img)} masks without image)'
        print(msg, flush=True)

    if a.subset == 'SAT':
        stem_map, clashes = {}, set()
        for rname, pr in pairs.items():
            for stem, pair in pr.items():
                if stem in stem_map:
                    clashes.add(stem)
                stem_map[stem] = (rname, pair)
        tr_stems, tr_src, tr_text = scnet_stems('train', a.csv_dir)
        te_stems, te_src, te_text = scnet_stems('test', a.csv_dir)
        overlap = set(tr_stems) & set(te_stems)
        if overlap:
            raise SystemExit(f'SC-Net train/test overlap: {sorted(overlap)[:10]}')
        for split, stems in (('train', tr_stems), ('test', te_stems)):
            if len(stems) != EXPECTED_SAT[split]:
                msg = f'SC-Net {split}.csv has {len(stems)} tiles, expected {EXPECTED_SAT[split]}'
                if not a.allow_count_mismatch:
                    raise SystemExit(msg + ' (use --allow-count-mismatch to continue)')
                print('  WARNING:', msg)
        need = set(tr_stems) | set(te_stems)
        amb = sorted(need & clashes)
        if amb:
            raise SystemExit(f'Ambiguous tile names found in several SAT regions: {amb[:10]}')
        lost = sorted(s for s in need if s not in stem_map)
        if lost:
            raise SystemExit(f'{len(lost)} SC-Net tiles not found in the dataset, e.g. {lost[:10]}')
        train_rows = [make_row(s, *stem_map[s]) for s in tr_stems]
        test_rows = [make_row(s, *stem_map[s]) for s in te_stems]
        source = {'train_csv': tr_src, 'test_csv': te_src,
                  'train_csv_sha256': hashlib.sha256(tr_text.encode('utf-8')).hexdigest(),
                  'test_csv_sha256': hashlib.sha256(te_text.encode('utf-8')).hexdigest()}
    else:
        train_rows, test_rows = [], []
        for key, rname in regions.items():
            stems = sorted(pairs[rname])
            random.Random(f'{a.seed}-uav-{key}').shuffle(stems)
            n_test = round_half_up(len(stems) * a.test_frac)
            for i, s in enumerate(stems):
                row = make_row(f'{key}__{s}', rname, pairs[rname][s])
                (test_rows if i < n_test else train_rows).append(row)
        total = len(train_rows) + len(test_rows)
        if total != EXPECTED_UAV_TOTAL:
            msg = f'UAV subset has {total} tiles, expected {EXPECTED_UAV_TOTAL}'
            if not a.allow_count_mismatch:
                raise SystemExit(msg + ' (use --allow-count-mismatch to continue)')
            print('  WARNING:', msg)
        source = {'protocol': f'random {1 - a.test_frac:.0%}/{a.test_frac:.0%} split per region, seed {a.seed}'}

    fit_rows, val_rows = stratified_val(train_rows, a.val_frac, a.seed)
    splits = {'train': fit_rows, 'val': val_rows, 'trainval': fit_rows + val_rows, 'test': test_rows}
    names = [r['name'] for r in fit_rows + val_rows + test_rows]
    assert len(names) == len(set(names)), 'duplicate tile names across splits'

    out = Path(a.out) / a.subset
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_csv(out / f'{name}.csv', rows)

    counts = {name: dict(sorted(((rg, sum(r['region'] == rg for r in rows)) for rg in pairs), key=lambda kv: kv[0]))
              for name, rows in splits.items()}
    summary = {
        'subset': a.subset, 'seed': a.seed, 'val_frac': a.val_frac,
        'n': {k: len(v) for k, v in splits.items()},
        'per_region': counts, 'region_dirs': region_dir, 'split_hash': split_hash(splits),
        'source': source, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'\n[prepare] {a.subset} split written to {out}')
    print(f'  {"region":30s} {"train":>7s} {"val":>6s} {"test":>6s}')
    for rg in sorted(pairs):
        print(f'  {rg:30s} {counts["train"][rg]:7d} {counts["val"][rg]:6d} {counts["test"][rg]:6d}')
    print(f'  {"TOTAL":30s} {len(fit_rows):7d} {len(val_rows):6d} {len(test_rows):6d}')
    print(f'  split hash: {summary["split_hash"][:16]}...')
    problems = sanity_check(fit_rows + val_rows + test_rows, a.check_n, a.seed)
    if problems:
        print('  Sanity check reported problems (see above). Training will still try to read every tile.')
    print(f'[prepare] done in {time.time() - t0:.1f} s', flush=True)
    return summary


if __name__ == '__main__':
    main()
