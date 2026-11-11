#!/usr/bin/env python3
"""Evaluation preparation and executable per-image helpers, not a completed study.

No implicit TEST inference, bank fitting, tuning or metrics placeholders. Main CLI
only verifies immutable compile metadata and emits a read-only planned schedule.
A full durable multi-run execution/resume/aggregation CLI remains a later stage.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODES = ('full_maps', 'rgb_zero_maps', 'endpoint_only')
SEEDS = (0, 1, 2)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


metrics = module('industrial_evaluation_metrics', HERE/'industrial_metrics.py')
geometry = module('industrial_evaluation_geometry', HERE/'industrial_geometry.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def checked_bytes(path, expected_sha256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Explicit lowercase SHA256 required')
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError('File identity differs: '+str(path))
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected_sha256 or sha(path) != expected_sha256:
        raise ValueError('File changed during read')
    return value


def safe_relative(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Explicit safe relative path required')
    rel = Path(relative);root = Path(root).resolve();path = (root/rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
        raise ValueError('Relative path traversal')
    cursor = root
    for part in rel.parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError('Symlink in compiled path')
    if not path.is_file():
        raise ValueError('Missing/nonregular compiled sample')
    return path


def prepare_evaluation_inputs(compiled_directory, source_manifest, *, expected_compile_summary_sha256):
    """Metadata-only verification, never evaluates TEST or builds a model."""
    root = Path(compiled_directory).resolve()
    summary = strict_json(checked_bytes(root/'compile-summary.json', expected_compile_summary_sha256))
    if summary.get('complete') is not True or summary.get('wrote_files') is not True:
        raise ValueError('A complete compiled artifact is required')
    allocation = strict_json(checked_bytes(root/'allocation.json', summary['allocation_sha256']))
    protocol = strict_json(checked_bytes(root/'protocol.json', summary['protocol_sha256']))
    records = [strict_json(line) for line in checked_bytes(root/'compile-records.jsonl', summary['compile_records_sha256']).decode().splitlines()]
    sources = [strict_json(line) for line in checked_bytes(source_manifest, summary['source']['source_manifest_sha256']).decode().splitlines()]
    source_by_id = {row['source_image_id']: row for row in sources}
    if len(source_by_id) != len(sources) or len(records) != summary['compiled_samples'] or allocation.get('allocation_schema') != 'imda-float-allocation-v2':
        raise ValueError('Compiled/source count or schema mismatch')
    entries = {row['sample_id']: row for row in allocation['samples']}
    if len(entries) != len(records) or len({r['sample_id'] for r in records}) != len(records) or set(entries) != {r['sample_id'] for r in records}:
        raise ValueError('Exact one-to-one allocation/record binding required')
    rows, source_split, seen_paths = [], {}, set()
    for record in records:
        entry = entries[record['sample_id']]
        if entry['npz'] != record['npz'] or record['npz'] in seen_paths:
            raise ValueError('Compiled NPZ path mismatch/duplicate')
        safe_relative(root, record['npz']);seen_paths.add(record['npz'])
        sid = entry['source_image_id'];source = source_by_id[sid]
        label, split = source['label'], entry['split_name']
        expected_source = {'image': source['image'], 'image_file_sha256': source['image_info']['file_sha256'],
            'image_decoded_pixel_sha256': source['image_info']['decoded_pixel_sha256'], 'mask': source['mask'] or None,
            'mask_file_sha256': source['mask_info']['file_sha256'] if source['mask_info'] else None,
            'original_image_label': label, 'original_split': source['split'], 'category': source['object']}
        if record['source'] != expected_source or split not in ('train','valid','test') or (split != 'test' and (label != 'normal' or source['split'] != 'train')) or (split == 'test' and source['split'] != 'test'):
            raise ValueError('Original label/source/split identity differs')
        if sid in source_split and source_split[sid] != split:
            raise ValueError('Source crosses split')
        source_split[sid] = split
        levels = [a for a in protocol['angle_levels_deg'] if 'industrial-'+digest([entry['trajectory_family_id'], a]) == entry['sample_id']]
        if len(levels) != 1 or record['trajectory_family_id'] != entry['trajectory_family_id']:
            raise ValueError('Unable to bind source level/family to sample identity')
        rows.append(dict(record, split_name=split, source_image_id=sid, angle_deg=levels[0], category=source['object']))
    test_sources = {row['source_image_id'] for row in rows if row['split_name'] == 'test'}
    if test_sources != {row['source_image_id'] for row in sources if row['split'] == 'test'}:
        raise ValueError('Evaluation must preserve every official TEST source')
    for split in ('valid', 'test'):
        selected = [row for row in rows if row['split_name'] == split]
        pairs = [(row['source_image_id'], row['angle_deg']) for row in selected]
        if len(set(pairs)) != len(pairs) or len(selected) != len({s for s,_ in pairs})*len(protocol['angle_levels_deg']):
            raise ValueError('Every validation/TEST source must cover every frozen level exactly once')
    if protocol['source_scope'] == 'visa-pcb1-4-official' and (len(test_sources) != 802 or protocol['angle_levels_deg'] != [.5,1.,1.5] or {r['category'] for r in rows} != {'pcb1','pcb2','pcb3','pcb4'}):
        raise ValueError('Formal scope requires all 802 TEST x 3 frozen levels')
    return {'root': str(root), 'summary': summary, 'protocol': protocol, 'rows': rows,
            'sources': source_by_id, 'entries': entries, 'compile_summary_sha256': expected_compile_summary_sha256}


def evaluation_schedule(prepared, split_name='test'):
    """Plan only: baselines once and all three modes x three seeds, paired by source."""
    if split_name not in ('valid', 'test'):
        raise ValueError('Only independent valid/TEST evaluation schedules')
    for row in sorted(prepared['rows'], key=lambda r:(r['source_image_id'],r['angle_deg'])):
        if row['split_name'] != split_name:
            continue
        for branch, seed in [('sharp', None), ('blur', None)]+[(mode, seed) for mode in MODES for seed in SEEDS]:
            yield {'sample_id': row['sample_id'], 'source_image_id': row['source_image_id'],
                   'category': row['category'], 'angle_deg': row['angle_deg'], 'split_name': split_name,
                   'branch': branch, 'seed': seed, 'cluster_unit': row['source_image_id']}


def select_bank_candidate(prepared, category, *, limit, selection_seed='normal-bank-sourcehash-v1'):
    """Only a 64/128-source train-only candidate, pending normal resource pilot."""
    if limit not in (64,128):
        raise ValueError('Only explicit 64 or 128 normal-source candidates')
    selected = [r for r in prepared['rows'] if r['split_name']=='train' and r['category']==category]
    if not selected or any(r['source']['original_image_label']!='normal' for r in selected):
        raise ValueError('Nonempty clean-normal train category required')
    if len({r['source_image_id'] for r in selected}) != len(selected):
        raise ValueError('One sharp bank source record per normal train source required')
    selected.sort(key=lambda r:digest([selection_seed,category,r['source_image_id'],r['source']['image_decoded_pixel_sha256']]))
    records = [{'source_image_id':r['source_image_id'], 'sample_id':r['sample_id'],
                'npz_sha256':r['npz_sha256'], 'source_decoded_sha256':r['source']['image_decoded_pixel_sha256']} for r in selected[:limit]]
    value = {'schema':'industrial-sharp-normal-bank-candidate-v1','category':category,'source_limit':limit,
             'selection_seed':selection_seed,'sources':records,'allocation_sha256':prepared['summary']['allocation_sha256'],
             'compile_summary_sha256':prepared['compile_summary_sha256'], 'candidate_only':True,
             'fit_input':'sharp_linear_rgb encoded once to sRGB; one per selected train normal',
             'shared_by':'same category sharp/blur/all restored modes/seeds/levels; never mix categories'}
    return dict(value,selection_sha256=digest(value))


def load_shared_bank(directory, *, expected_metadata_sha256, expected_training_manifest_sha256, factory_kwargs):
    """Use detector's real public reload API; never set private fitted flags."""
    detector_module = module('industrial_evaluation_detector', HERE/'industrial_detector.py')
    kwargs = dict(factory_kwargs)
    config = kwargs.pop('config', None)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError('factory_kwargs.config must be a plain frozen configuration dict')
        kwargs['config'] = detector_module.DetectorConfig(**config)
    detector = detector_module.create_detector(**kwargs)
    detector.load_fitted(directory, expected_training_manifest_sha256=expected_training_manifest_sha256,
                         expected_metadata_sha256=expected_metadata_sha256)
    return detector


@contextlib.contextmanager
def fp32_inference():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old[:3]
        torch.use_deterministic_algorithms(old[3], warn_only=old[4])


class BestRestorer:
    def __init__(self, trainer, model, contract, identity, device):
        self.trainer,self.model,self.contract,self.identity,self.device=trainer,model,contract,identity,device

    def predict_linear(self, sample):
        batch=self.trainer.float_io.to_torch_bchw([sample],device=self.device)
        batch=self.trainer._apply_mode(batch,self.contract['input_mode'])
        self.model.eval()
        with torch.no_grad(), fp32_inference(), torch.autocast(device_type=self.device.type,enabled=False):
            value=self.model(batch['input'])[-1]
        if value.dtype!=torch.float32 or value.shape!=batch['target'].shape or not torch.isfinite(value).all():
            raise ValueError('Finite FP32 final-resolution restoration required')
        return value[0].detach().cpu().permute(1,2,0).contiguous().numpy()


def restoration_identity(report, *, expected_checkpoint_sha256, expected_run_report_sha256,
                         expected_allocation_sha256, input_mode, seed, device):
    """Derive inference identity from a caller-pinned run report, not a receipt."""
    contract=report['contract'];selected=report['best_model']
    return {'checkpoint_sha256':expected_checkpoint_sha256,'run_report_sha256':expected_run_report_sha256,
            'best_model_sha256':selected['sha256'],'selected_update':selected['selected_update'],
            'mode':input_mode,'seed':seed,'allocation_sha256':expected_allocation_sha256,
            'num_res':contract['num_res'],'total_updates':contract['total_updates'],
            'inference_torch':str(torch.__version__),'inference_numpy':np.__version__,
            'inference_device':str(torch.device(device)),'inference_precision':'FP32/TF32off/deterministic'}


def load_best_restorer(checkpoint_path, run_report_path, *, expected_checkpoint_sha256,
                       expected_run_report_sha256, expected_allocation_sha256, input_mode, seed,
                       device='cpu', trainer_path=HERE/'float_training.py'):
    """Inference loader for a COMPLETE fixed-budget run, including its early best.

    The trusted run report file hash binds the whole expected training contract;
    source/mode/seed/allocation and actual code bytes are checked independently.
    This does not call resume's optimizer loader or reject final-budget states.
    """
    report=strict_json(checked_bytes(run_report_path,expected_run_report_sha256))
    contract=report['contract']
    if input_mode not in MODES or type(seed) is not int or seed not in SEEDS or contract.get('input_mode')!=input_mode or contract.get('seed')!=seed or contract.get('allocation_sha256')!=expected_allocation_sha256:
        raise ValueError('Wrong restoration source/mode/seed contract')
    if report.get('complete_budget') is not True or report.get('completed_update')!=contract['total_updates'] or report.get('requested_total_updates')!=contract['total_updates']:
        raise ValueError('Only a completed frozen total budget can be evaluated')
    trainer=module('industrial_evaluation_trainer',Path(trainer_path))
    if trainer.code_hashes()!=contract['code_sha256']:
        raise ValueError('Trainer/model/IO code identity differs from checkpoint contract')
    if sha(checkpoint_path)!=expected_checkpoint_sha256 or report['checkpoint_files'].get(Path(checkpoint_path).name)!=expected_checkpoint_sha256:
        raise ValueError('Checkpoint file not bound by the completed run report')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if sha(checkpoint_path)!=expected_checkpoint_sha256 or checkpoint.get('checkpoint_schema')!=trainer.CHECKPOINT_SCHEMA or checkpoint.get('contract')!=contract or checkpoint.get('update')!=contract['total_updates']:
        raise ValueError('Final checkpoint identity/schema/budget differs')
    history=checkpoint['history']
    if report.get('history') != history:
        raise ValueError('Run report/checkpoint history differs')
    valid=[r for r in history if 'valid' in r]
    if len(history)!=contract['total_updates'] or [r['update'] for r in history]!=list(range(1,len(history)+1)) or not valid or not trainer.finite_tree(history):
        raise ValueError('Complete finite ordered selection history required')
    best_mae=min(r['valid']['mae'] for r in valid)
    best=next(r for r in valid if r['valid']['mae']==best_mae)
    state=checkpoint['best_model_state'];state_sha=trainer.state_hash(state)
    selected=report['best_model']
    if not trainer.finite_tree(state) or state_sha!=checkpoint['best_model_sha256'] or state_sha!=best.get('model_sha256') or checkpoint['best_update']!=best['update'] or checkpoint['best_mae']!=best_mae or selected['state_key']!='best_model_state' or selected['sha256']!=state_sha or selected['selected_update']!=best['update'] or report['best_update']!=best['update'] or report['best_valid_mae']!=best_mae:
        raise ValueError('Best model identity differs from min-valid history/report')
    device=torch.device(device)
    if device.type not in ('cpu','cuda') or device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CPU/CUDA inference device unavailable')
    # Construction RNG is isolated: loading weights does not alter the caller's dropout RNG.
    with torch.random.fork_rng(devices=[]):
        model=trainer.build_plus(contract['num_res'],device='cpu').float()
    expected=model.state_dict()
    if set(state)!=set(expected) or any(state[k].shape!=v.shape or state[k].dtype!=v.dtype for k,v in expected.items()):
        raise ValueError('Best state shape/dtype differs from fixed architecture')
    model.load_state_dict(state,strict=True);model.to(device).eval().requires_grad_(False)
    identity=restoration_identity(report,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_run_report_sha256=expected_run_report_sha256,expected_allocation_sha256=expected_allocation_sha256,
        input_mode=input_mode,seed=seed,device=device)
    return BestRestorer(trainer,model,contract,identity,device)


def load_compiled_sample(prepared,sample_id):
    row=next(r for r in prepared['rows'] if r['sample_id']==sample_id)
    path=safe_relative(prepared['root'],row['npz'])
    if sha(path)!=row['npz_sha256']:
        raise ValueError('Compiled sample hash differs')
    io=module('industrial_evaluation_float_io',HERE/'float_dataset.py')
    sample=io.load_npz_sample(path,expected_split=row['split_name'],require_target=True,
          expected_manifest_sha256=prepared['summary']['allocation_sha256'])
    source=sample['metadata']['input_source']
    if sha(path)!=row['npz_sha256'] or source['source']!=row['source'] or source['mapping']!=row['source_mapping']:
        raise ValueError('Sample source/geometry differs from compiled record')
    return sample,row


def read_original_annotation(prepared,source_image_id,data_root,mapping):
    source=prepared['sources'][source_image_id]
    compiler=module('industrial_evaluation_compiler',HERE/'industrial_compile.py')
    if source['label']=='anomaly':
        mask=compiler.checked_image(data_root,source['mask'],source['mask_info'],mask=True)
    else:
        mask=np.zeros((source['image_info']['height'],source['image_info']['width']),dtype=np.uint8)
    return geometry.original_annotation(mask,source['label'],mapping)


def evaluate_one(sample, row, annotation, *, branch, detector, restorer=None,
                 image_threshold=None,pixel_threshold=None,return_crop_score=False):
    """Actually evaluate one compiled image; does not write or tune any threshold.

    Caller persists pixel scores in shards and aggregates within class/level/
    branch/seed first. Original source ID is the later bootstrap cluster unit.
    """
    provenance=sample['metadata']['provenance']
    source=sample['metadata']['input_source']
    if provenance['sample_id']!=row['sample_id'] or provenance['source_image_id']!=row['source_image_id'] or provenance['split_name']!=row['split_name'] or source['source']!=row['source'] or source['mapping']!=row['source_mapping'] or annotation['statistics']!=row['annotation']:
        raise ValueError('Per-image source/label/geometry/annotation binding differs')
    if branch in ('sharp','blur') and restorer is not None:
        raise ValueError('Deterministic baselines have no restoration seed/model')
    if branch=='sharp':
        prediction=sample['sharp_linear_rgb']
    elif branch=='blur':
        prediction=sample['blur_linear_rgb']
    elif branch in MODES and restorer is not None and restorer.contract['input_mode']==branch:
        prediction=restorer.predict_linear(sample)
    else:
        raise ValueError('Explicit baseline or matching restored mode required')
    encoded,clipping=metrics.detection_encoding(prediction)
    detected=detector.predict_srgb(encoded)
    mapping=sample['metadata']['input_source']['mapping']
    score=geometry.remap_score_to_original_roi(detected['score_roi'],mapping)
    original_roi=annotation['roi_mask_original_grid']
    if score.shape!=original_roi.shape or annotation['statistics']['image_label']!=row['source']['original_image_label']:
        raise ValueError('Original label/ROI annotation mismatch')
    record={'sample_id':row['sample_id'],'npz_sha256':row['npz_sha256'],'source_image_id':row['source_image_id'],'category':row['category'],
            'angle_deg':row['angle_deg'],'branch':branch,'seed':restorer.contract['seed'] if restorer else None,
            'split_name':row['split_name'],'original_image_label':row['source']['original_image_label'],
            'image_score':detected['image_score'],'restoration':metrics.restoration_metrics(prediction,sample['sharp_linear_rgb'],sample['valid_mask']),
            'clipping':clipping,'annotation':annotation['statistics'],
            'pixels':metrics.pixel_statistics(score,original_roi,pixel_threshold),
            'detector_contract_sha256':detected['contract_sha256'],'bank_identity':detected['bank_identity'],'bootstrap_cluster':row['source_image_id'],
            'unobserved_policy':'Outside common original-grid ROI has no score; retained/lost defects reported separately'}
    if restorer:
        record['restorer']=restorer.identity
    if image_threshold is not None:
        stats=annotation['statistics']
        record['image_threshold_statistics']=metrics.threshold_image_statistics(detected['image_score'],record['original_image_label'],image_threshold,
            original_defect_pixels=stats['original_defect_pixels'],retained_defect_pixels=stats['retained_defect_pixels'])
    if return_crop_score:
        return record,score,(original_roi!=0),detected['score_roi']
    return record,score,(original_roi!=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled',required=True,type=Path)
    parser.add_argument('--source-manifest',required=True,type=Path)
    parser.add_argument('--compile-summary-sha256',required=True)
    args=parser.parse_args()
    prepared=prepare_evaluation_inputs(args.compiled,args.source_manifest,expected_compile_summary_sha256=args.compile_summary_sha256)
    planned=list(evaluation_schedule(prepared))
    print(json.dumps({'action':'metadata preflight only; no TEST inference/fitting/writes',
          'planned_image_evaluations':len(planned),'test_sources':len({r['source_image_id'] for r in planned}),
          'levels':prepared['protocol']['angle_levels_deg'],'branches':['sharp','blur',*MODES],
          'restoration_seeds':list(SEEDS),'complete_evaluation_driver':False},indent=2))
