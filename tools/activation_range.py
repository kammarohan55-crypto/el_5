"""Measures activation magnitudes inside the pretrained DINOv3 ConvNeXt-T encoder (FP32, CPU) on a natural
image, to judge FP16 overflow / precision risk (FP16 max 65504, 11-bit significand)."""
import sys
import numpy as np
import torch
import timm
import matplotlib.cbook as cbook
from PIL import Image

torch.manual_seed(0)
img = Image.open(cbook.get_sample_data('grace_hopper.jpg')).convert('RGB').resize((512, 512))
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 255.0
m = timm.create_model('convnext_tiny.dinov3_lvd1689m', pretrained=True).eval()
cfg = m.pretrained_cfg
mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1); std = torch.tensor(cfg['std']).view(1, 3, 1, 1)
xn = (x - mean) / std
stats = []
def hook(name):
    def f(mod, inp, out):
        if torch.is_tensor(out):
            a = out.detach().abs()
            stats.append((name, float(a.max()), float(a.mean()), float(out.detach().std())))
    return f
for n, mod in m.named_modules():
    if n.count('.') <= 3 and n and (n.startswith('stages') or n.startswith('stem')):
        if n.endswith(('blocks', 'downsample')) or n.count('.') == 3 or n.startswith('stem'):
            mod.register_forward_hook(hook(n))
with torch.no_grad():
    m.forward_features(xn)
print(f'{"module":34s} {"max|a|":>10s} {"mean|a|":>10s} {"std":>10s}')
for n, mx, mn, sd in stats:
    if n.count('.') == 3 and not n.endswith('.0'):  # every block of a stage except the first is listed below
        pass
    print(f'{n:34s} {mx:10.1f} {mn:10.2f} {sd:10.2f}')
# layer-scale gammas
for n, p in m.named_parameters():
    if n.endswith('gamma'):
        pass
g = [(n, float(p.abs().max())) for n, p in m.named_parameters() if 'gamma' in n]
print('largest layer-scale |gamma|:', sorted(g, key=lambda t: -t[1])[:5])
