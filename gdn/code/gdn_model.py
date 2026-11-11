"""Pinned official architecture with explicit RGB/CMF padding only.

This does not set backend determinism flags or download/load weights. Both
gyro arguments receive the same ideal start-CMF, epoch=100; no noise curriculum.
Caller must disclose this common-budget adaptation separately from native GDN.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_HASHES = {
    'model.py': 'e189e154bb48307c18604ee5dbeae50b2f31fae22ab0c050f120b2e424b745d4',
    'arch_util.py': 'e1792f1a8cdb79d6fb728a52f2cc8eabb0e2700439b7e3c2a5eaa2c649873f1b',
}

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def official_class():
    paths = {name: ROOT/'vendor/official'/name for name in OFFICIAL_HASHES}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_HASHES[name]:
            raise ValueError('Pinned official source differs: '+name)
    arch = _load('_imda_gdn_pinned_arch', paths['arch_util.py'])
    # The official file has an absolute `from arch_util import ...`. Resolve
    # that one import to the pinned local module, then restore any prior alias.
    # The source files themselves are unchanged; classes retain their objects.
    missing = object()
    previous = sys.modules.get('arch_util', missing)
    try:
        sys.modules['arch_util'] = arch
        module = _load('_imda_gdn_pinned_model', paths['model.py'])
    finally:
        if previous is missing:
            sys.modules.pop('arch_util', None)
        else:
            sys.modules['arch_util'] = previous
    return module.GyroDeblurNet

class GDNStart(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = official_class()(img_channel=3, gyro_channel=16, width=32,
                                      enc_blk_nums=[2, 2, 2],
                                      middle_blk_num=16, dec_blk_nums=[1, 1, 1])

    def forward(self, blur, cmf):
        if blur.ndim != 4 or blur.shape[1] != 3:
            raise ValueError('BCHW RGB required')
        batch, _, height, width = blur.shape
        hp, wp = 8*((height+7)//8), 8*((width+7)//8)
        if tuple(cmf.shape) != (batch, 16, hp//2, wp//2):
            raise ValueError('CMF must cover the padded half lattice')
        if (blur.dtype != torch.float32 or cmf.dtype != torch.float32
                or blur.device != cmf.device):
            raise ValueError('Same-device float32 RGB and CMF required')
        if not torch.isfinite(blur).all() or not torch.isfinite(cmf).all():
            raise ValueError('Finite inputs required; no replacement')
        padded = F.pad(blur, (0, wp-width, 0, hp-height))
        result = self.model(padded, cmf, cmf, 100)
        if result.shape != padded.shape or not torch.isfinite(result).all():
            raise ValueError('Native network output is invalid')
        return result[:, :, :height, :width]
