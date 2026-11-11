#!/usr/bin/env python3
"""Candidate GDN-start common-budget trainer; no resume, TEST forward or downloads.

Default is a read-only full-data preflight. Formal writes use exactly three
declared seeds and the fixed 20k budget; a failure preserves partial artifacts.
The small optimize() function is separately callable for invented CPU tests.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision

from cmf_start import from_q16_metadata
from gdn_model import GDNStart

ROOT = Path(__file__).resolve().parents[1]
TRAINING_HASHES = {
    'float_training.py': '25e2c5a58b3afed323e15b2bc31098be3ce23bc6f2dda64c21880eaad45abb1b',
    'float_dataset.py': 'cde8c7549760ae0323c74ca0e16c427ada101363791385324d4a5ef26b94100f',
}
ALLOCATION_SHA = '2539210b6a48caea3e0745b9dbd2e16571a087e98c754eab08e0f1538637adb5'
COMPILE_SUMMARY_SHA = '20b7e50ed76008919db8185b151f734478688c4af79cefad551c6ea8a7ab0ade'
COMPILE_RECORDS_SHA = '3dd43ed2ea89788e638309fadb2a2c47b157aab3ca8289973b27d44500339a48'
CHECKPOINT_SCHEMA = 'gdn-start-training-checkpoint-v1'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_training_helpers():
    for name, expected in TRAINING_HASHES.items():
        if file_hash(ROOT/'vendor/training'/name) != expected:
            raise ValueError('Frozen E3 helper differs: '+name)
    spec = importlib.util.spec_from_file_location('gdn_start_e3_helpers', ROOT/'vendor/training/float_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e3 = load_training_helpers()


def source_hashes():
    paths = ['code/train_gdn_start.py', 'code/cmf_start.py', 'code/gdn_model.py',
             'vendor/reference/imu_synthesis.py', 'vendor/official/model.py',
             'vendor/official/arch_util.py', 'vendor/training/float_training.py',
             'vendor/training/float_dataset.py', 'protocols/comparison-candidate-v1.json']
    return {name: file_hash(ROOT/name) for name in paths}


def prepare_official(allocation_path):
    """Full official byte/schema provenance check, with no model/optimizer/forward."""
    allocation_path = Path(allocation_path).resolve()
    if file_hash(allocation_path) != ALLOCATION_SHA:
        raise ValueError('Only completed frozen C3 allocation is accepted by formal entry')
    folder = allocation_path.parent
    for name, expected in [('compile-summary.json', COMPILE_SUMMARY_SHA),
                           ('compile-records.jsonl', COMPILE_RECORDS_SHA)]:
        if file_hash(folder/name) != expected:
            raise ValueError('Completed compiler identity differs: '+name)
    prepared = e3.prepare_allocation(allocation_path)
    if {k: len(v) for k, v in prepared['splits'].items()} != {'train': 2890, 'valid': 2172, 'test': 2406}:
        raise ValueError('Incomplete official split coverage')
    rows = [json.loads(line) for line in (folder/'compile-records.jsonl').read_text().splitlines()]
    if len(rows) != len(prepared['npz_records']):
        raise ValueError('Compiler record coverage differs')
    for actual, compiled in zip(prepared['npz_records'], rows):
        if (actual['sample_id'], actual['npz'], actual['sha256']) != (compiled['sample_id'], compiled['npz'], compiled['npz_sha256']):
            raise ValueError('NPZ identity/order differs from compiler receipt')
    # Validate the new metadata/prior interface everywhere before optimization.
    # This reads TEST bytes/geometry only; no model prediction or target selection.
    for record in prepared['npz_records']:
        sample = e3._read_record(prepared, record)
        from_q16_metadata(sample['metadata'], record['height'], record['width'])
        del sample
    return prepared


def tensor_batch(samples, crops, device):
    """Use original RGB/target/mask, construct only the separate gyro input."""
    blur, target, masks, fields = [], [], [], []
    for sample, (top, left, height, width) in zip(samples, crops, strict=True):
        h, w = sample['blur_linear_rgb'].shape[:2]
        field, _ = from_q16_metadata(sample['metadata'], h, w, crop=(top, left, height, width))
        region = np.s_[top:top+height, left:left+width]
        blur.append(sample['blur_linear_rgb'][region].copy())
        target.append(sample['sharp_linear_rgb'][region].copy())
        masks.append(sample['valid_mask'][region].copy())
        fields.append(field)
    if not samples:
        raise ValueError('Nonempty batch required')
    def nchw(items):
        return torch.from_numpy(np.stack(items).transpose(0, 3, 1, 2).copy()).to(device)
    batch = {'blur': nchw(blur), 'cmf': nchw(fields), 'target': nchw(target),
             'valid_mask': torch.from_numpy(np.stack(masks)[:, None]).to(device)}
    if batch['valid_mask'].dtype != torch.bool or not batch['valid_mask'].flatten(1).any(1).all():
        raise ValueError('Empty planned crop; no resampling or fallback')
    return batch


def training_batch(prepared, plan, device):
    # Keep at most one batch, never a full image pool, and no TEST sampling.
    samples = [e3._read_record(prepared, prepared['splits']['train'][i]) for i, *_ in plan]
    return tensor_batch(samples, [tuple(p[1:]) for p in plan], device)


def evaluate_valid(model, prepared, device):
    model.eval()
    absolute, squared, elements = 0., 0., 0
    with torch.no_grad():
        for record in prepared['splits']['valid']:
            sample = e3._read_record(prepared, record)
            batch = tensor_batch([sample], [(0, 0, record['height'], record['width'])], device)
            del sample
            prediction = model(batch['blur'], batch['cmf'])
            e3.masked_spatial_l1(prediction, batch['target'], batch['valid_mask'])
            residual = (prediction-batch['target'])[batch['valid_mask'].expand_as(prediction)].double()
            absolute += residual.abs().sum().item()
            squared += residual.square().sum().item()
            elements += residual.numel()
    if not elements:
        raise ValueError('Empty validation coverage')
    return {'mae': absolute/elements, 'mse': squared/elements,
            'valid_rgb_elements': elements, 'sample_count': len(prepared['splits']['valid'])}


def configure_backend(device, seed, allow_nondeterministic_cuda):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and allow_nondeterministic_cuda):
        raise ValueError('Explicit CPU/CUDA device; CPU determinism cannot be waived')
    if device.type == 'cuda' and not allow_nondeterministic_cuda:
        raise ValueError('GDN CUDA DCN backward needs explicit nondeterministic exception')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no fallback')
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Frozen CUDA configuration requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(device.type == 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def optimize(model, optimizer, prepared, output, contract, *, updates, batch_size,
             patch_size, validation_interval, checkpoint_interval, seed, device):
    """Fixed-budget loop. No resume/early stopping; earliest exact MAE tie wins."""
    if any(type(v) is not int or v < 1 for v in (updates, batch_size, patch_size, validation_interval, checkpoint_interval)):
        raise ValueError('Positive integer budgets required')
    if checkpoint_interval % validation_interval or updates % validation_interval or updates % checkpoint_interval:
        raise ValueError('All checkpoints and final update must be frozen validation points')
    output = Path(output)
    output.mkdir(exist_ok=False)
    history, best_state, best_update, best_mae = [], None, None, float('inf')
    initial_hash = e3.parameter_hash(model)
    begun = time.perf_counter()
    for update in range(1, updates+1):
        model.train()
        plan = e3.sample_plan(prepared, seed, update, batch_size, patch_size)
        batch = training_batch(prepared, plan, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['blur'], batch['cmf'])
        loss = e3.masked_spatial_l1(prediction, batch['target'], batch['valid_mask'])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        row = {'update': update, 'train_masked_l1': loss.item(), 'gradient_l2_norm': norm.item(),
               'sampling_plan_sha256': hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()}
        if update % validation_interval == 0:
            row['valid'] = evaluate_valid(model, prepared, device)
            row['model_sha256'] = e3.parameter_hash(model)
            if row['valid']['mae'] < best_mae:
                best_mae, best_update = row['valid']['mae'], update
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        history.append(row)
        if update % checkpoint_interval == 0:
            checkpoint = {'checkpoint_schema': CHECKPOINT_SCHEMA, 'contract': contract,
                          'update': update, 'model_state': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(), 'history': history,
                          'best_mae': best_mae, 'best_update': best_update,
                          'model_sha256': e3.parameter_hash(model), 'best_model_state': best_state,
                          'best_model_sha256': e3.state_hash(best_state),
                          'torch_rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
            if not e3.finite_tree(checkpoint):
                raise ValueError('Nonfinite optimizer/model/history; run incomplete')
            with (output/f'update-{update:08d}.pt').open('xb') as stream:
                torch.save(checkpoint, stream)
        print(json.dumps({'event': 'gdn_start_training_progress', **row}, allow_nan=False), flush=True)
    final = output/f'update-{updates:08d}.pt'
    return {'contract': contract, 'complete_budget': True, 'completed_update': updates,
            'resumed_from': None, 'history': history, 'best_update': best_update,
            'best_valid_mae': best_mae, 'initial_parameter_sha256': initial_hash,
            'final_parameter_sha256': e3.parameter_hash(model),
            'best_model': {'checkpoint': str(final.resolve()), 'state_key': 'best_model_state',
                           'sha256': e3.state_hash(best_state), 'selected_update': best_update},
            'checkpoint_files': {p.name: file_hash(p) for p in output.glob('update-*.pt')},
            'wall_seconds_including_validation_checkpointing': time.perf_counter()-begun}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-nondeterministic-cuda', action='store_true')
    parser.add_argument('--write', type=Path)
    args = parser.parse_args(argv)
    if args.device not in ('cpu', 'cuda:0'):
        raise ValueError('This fixed study supports CPU readonly preflight or cuda:0')
    if args.write is not None and args.device != 'cuda:0':
        raise ValueError('Formal fixed protocol writes require cuda:0; CPU is for engineering helpers')
    if args.write is not None and not args.allow_nondeterministic_cuda:
        raise ValueError('Formal CUDA write requires the explicit nondeterministic exception')
    if args.write is not None and args.write.exists():
        raise FileExistsError(args.write)
    before = source_hashes()
    prepared = prepare_official(args.allocation)
    if before != source_hashes():
        raise ValueError('Source changed during preflight')
    contract = {'schema': 'gdn-start-common-budget-training-v1', 'architecture': 'GDN-start adapted official architecture',
                'allocation_sha256': prepared['allocation_sha256'], 'npz_records': prepared['npz_records'],
                'code_sha256': before, 'seed': args.seed, 'total_updates': 20000, 'batch_size': 4,
                'patch_size': 224, 'validation_interval': 1000, 'checkpoint_interval': 5000,
                'optimizer': {'name': 'Adam', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
                              'weight_decay': 0., 'foreach': False, 'fused': False,
                              'capturable': False, 'amsgrad': False, 'maximize': False},
                'sampling': 'unchanged E3 stateless PCG64 source/crop plan',
                'loss': 'masked final-resolution unclipped linear-RGB L1',
                'validation': 'all2172 normal-valid, scalar-weighted MAE, earliest minimum',
                'reference': 'original exposure_start B/S0/ROI/mask, no reference warp',
                'device': str(torch.device(args.device)), 'precision': 'float32',
                'deterministic_algorithms': torch.device(args.device).type == 'cpu',
                'TF32': False, 'cudnn_benchmark': False, 'resume': 'unsupported; fresh run only'}
    if args.write is None:
        print(json.dumps({'action': 'readonly_full_preflight_no_model_or_optimization', 'contract': contract}))
        return
    device = configure_backend(args.device, args.seed, args.allow_nondeterministic_cuda)
    model = GDNStart().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False, capturable=False,
                                 amsgrad=False, maximize=False)
    report = optimize(model, optimizer, prepared, args.write, contract, updates=20000, batch_size=4,
                      patch_size=224, validation_interval=1000, checkpoint_interval=5000,
                      seed=args.seed, device=device)
    if source_hashes() != before:
        raise ValueError('Source changed during optimization; no completed report')
    report['environment'] = {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
                             'CUDA': torch.version.cuda, 'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                             'GPU': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    report['TEST'] = 'byte/metadata preflight only; no forward, optimization or selection'
    with (args.write/'run.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
