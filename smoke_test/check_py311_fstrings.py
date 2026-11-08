#!/usr/bin/env python3
"""Flags f-string syntax that only Python >= 3.12 accepts (quote reuse / backslashes inside replacement
fields), so the scripts also run on Kaggle images with Python 3.10 / 3.11. Needs Python >= 3.12 to run."""
import io
import json
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKSLASH = chr(92)


def check(src, label):
    problems, stack = [], []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.FSTRING_START:
            q = t.string.lstrip('fFrRbB')[:1]
            if stack and q == stack[-1]:
                problems.append((t.start[0], 'nested f-string reuses the enclosing quote', t.line.strip()[:100]))
            stack.append(q)
        elif t.type == tokenize.FSTRING_END:
            stack.pop()
        elif stack and t.type == tokenize.STRING:
            q = t.string.lstrip('rRbBuU')[:1]
            if q == stack[-1]:
                problems.append((t.start[0], 'string inside an f-string uses the same quote', t.line.strip()[:100]))
            if BACKSLASH in t.string:
                problems.append((t.start[0], 'backslash inside an f-string expression', t.line.strip()[:100]))
    print(f'{label}: {len(problems)} problem(s)')
    for p in problems[:20]:
        print('   ', p)
    return problems


def main():
    total = []
    for f in ['prepare_data.py', 'train_landslide.py', 'build_notebook.py', 'smoke_test/unit_checks.py',
              'smoke_test/run_smoke.py', 'smoke_test/make_synthetic_cas.py']:
        total += check((ROOT / f).read_text(encoding='utf-8'), f)
    nb_path = ROOT / 'CAS_Landslide_StructuralPrior_Kaggle.ipynb'
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] == 'code':
                s = ''.join(c['source'])
                if s.startswith('%%'):
                    s = s.split('\n', 1)[1]
                total += check(s, f'notebook cell {i}')
    print('TOTAL problems:', len(total))
    sys.exit(1 if total else 0)


if __name__ == '__main__':
    main()
