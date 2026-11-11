"""New TWO-condition exploratory arithmetic, not the old four-mode normalizer.

These APIs are arithmetic fixtures only. Actual complete-GDN receipt adapter,
formal CLI and 802-source authentication live in gdn_result_statistics.py. No result from this module
is a verified industrial statistic without that adapter's actual producer/source checks.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

S2_SHA = 'e3730c748f7c317155288ced71ff5921d8bf8dfc4cc5a4b036618f5c6d9bc93e'
CONDITIONS = ('full_maps', 'gdn_start')
CATEGORIES = ('pcb1', 'pcb2', 'pcb3', 'pcb4')
SEEDS = (0, 1, 2)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_math(root):
    """Reuse actual fixed S2 source ordering/resampling/tie arithmetic unchanged."""
    root = Path(root).resolve()
    if sha(root/'manifest.json') != S2_SHA:
        raise ValueError('Only frozen S2 arithmetic dependency accepted')
    m = json.loads((root/'manifest.json').read_text())
    for row in m['files']:
        rel = Path(row['path']); path = root/rel
        if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe fixed math source')
        cursor = root
        for part in rel.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError('Symlink math source')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Frozen math payload changed')
    spec = importlib.util.spec_from_file_location('gdn_exploratory_fixed_s2_math', root/'code/revision/industrial_statistics.py')
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def normalize_gdn_record(record, *, math, proof=None):
    """Schema arithmetic adapter only; formal producer authentication is separate."""
    required = {'sample_id', 'npz_sha256', 'source_image_id', 'category', 'original_image_label', 'angle_deg',
                'branch', 'seed', 'split_name', 'restoration', 'image_score', 'restorer', 'bank_identity', 'detector_contract_sha256'}
    if not isinstance(record, dict) or not required <= record.keys() or record['split_name'] != 'test' or record['branch'] != 'gdn_start':
        raise ValueError('Explicit GDN TEST record required')
    if (not isinstance(record['source_image_id'], str) or not record['source_image_id']
            or record['category'] not in CATEGORIES or record['original_image_label'] not in ('normal', 'anomaly')):
        raise ValueError('Explicit original source/category/label domain required')
    seed = record['seed']; model = record['restorer']
    if (type(seed) is not int or seed not in SEEDS or not isinstance(model, dict)
            or model.get('schema') != 'gdn-start-restorer-identity-v1' or model.get('mode') != 'gdn_start'
            or model.get('seed') != seed or 'num_res' in model or model.get('target_reference') != 'exposure_start'
            or model.get('model_parameter_count') != 16311296 or model.get('total_updates') != 20000):
        raise ValueError('Explicit adapted GDN architecture/budget identity required')
    if 'counterfactual' in record:
        raise ValueError('GDN main branch cannot masquerade as CF')
    def digest(v):
        return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
    for key in ('checkpoint_sha256', 'run_report_sha256', 'best_model_sha256', 'allocation_sha256', 'trainer_manifest_sha256', 'inference_adapter_sha256'):
        if not digest(model.get(key)):
            raise ValueError('Actual GDN model digest required')
    if type(model.get('selected_update')) is not int or not 1 <= model['selected_update'] <= 20000:
        raise ValueError('Actual normal-VALID-selected update required')
    bank = record['bank_identity']
    if not isinstance(bank, dict) or any(not digest(bank.get(k)) for k in ('training_manifest_sha256', 'bank_sha256', 'metadata_sha256')):
        raise ValueError('Actual shared class bank required')
    if not digest(record['npz_sha256']) or not digest(record['detector_contract_sha256']) or not isinstance(record['sample_id'], str) or not record['sample_id']:
        raise ValueError('Observation/detector identity required')
    return {'source_image_id': record['source_image_id'], 'category': record['category'],
            'original_image_label': record['original_image_label'], 'angle_deg': math._finite_scalar(record['angle_deg'], 'angle', True),
            'seed': seed, 'condition': 'gdn_start', 'linear_mse': math._finite_scalar(record['restoration']['mse'], 'MSE', True),
            'image_score': math._finite_scalar(record['image_score'], 'image score'),
            'identity': {k: copy.deepcopy(record[k]) for k in ('sample_id', 'npz_sha256', 'restorer', 'bank_identity', 'detector_contract_sha256')},
            'counterfactual': None, 'proof': copy.deepcopy(proof)}


def prepare_arithmetic_panel(rows, sources, levels, *, math):
    """Strict paired cells. This generic fixture API does not certify 802 sources."""
    sources = math.ordered_sources(sources); levels = tuple(math._finite_scalar(x, 'level', True) for x in levels)
    if levels != (.5, 1., 1.5) or any(not any(r['category'] == c and r['original_image_label'] == label for r in sources)
                                  for c in CATEGORIES for label in ('normal', 'anomaly')):
        raise ValueError('Fixed3 levels and all4 categories/both label strata required')
    index = {r['source_image_id']: i for i, r in enumerate(sources)}; by_id = {r['source_image_id']: r for r in sources}
    mse = np.empty((2, 3, 3, len(sources)), np.float64); scores = np.empty_like(mse)
    seen, observations, banks, models = {}, {}, {}, {}; allocation = None
    keys = {'source_image_id', 'category', 'original_image_label', 'angle_deg', 'seed', 'condition', 'linear_mse', 'image_score', 'identity', 'counterfactual', 'proof'}
    for row in rows:
        if set(row) != keys or row['condition'] not in CONDITIONS or type(row['seed']) is not int or row['seed'] not in SEEDS or row['angle_deg'] not in levels:
            raise ValueError('Exact new2condition normalized cell required')
        sid = row['source_image_id']; original = by_id.get(sid)
        if original is None or any(row[k] != original[k] for k in ('category', 'original_image_label')) or row['counterfactual'] is not None:
            raise ValueError('Original source/label/condition differs')
        key = (sid, row['angle_deg'], row['seed'], row['condition'])
        if key in seen:
            raise ValueError('Duplicate source/level/seed/condition cell')
        seen[key] = row; identity = row['identity']; model = identity['restorer']
        if model['mode'] != row['condition'] or model['seed'] != row['seed']:
            raise ValueError('New panel model mode/seed differs')
        observation = (identity['sample_id'], identity['npz_sha256'])
        if observations.setdefault((sid, row['angle_deg']), observation) != observation:
            raise ValueError('Paired original sample/NPZ differs')
        bank = math.canonical([identity['bank_identity'], identity['detector_contract_sha256']])
        if banks.setdefault(row['category'], bank) != bank:
            raise ValueError('Full/GDN must share the actual same class bank')
        model_key = (row['condition'], row['seed']); value = math.canonical(model)
        if models.setdefault(model_key, value) != value:
            raise ValueError('Condition/seed weights change across sources/levels')
        if allocation is None:
            allocation = model['allocation_sha256']
        if allocation != model['allocation_sha256']:
            raise ValueError('Matched allocation differs')
        cell = (CONDITIONS.index(row['condition']), row['seed'], levels.index(row['angle_deg']), index[sid])
        mse[cell] = math._finite_scalar(row['linear_mse'], 'MSE', True); scores[cell] = math._finite_scalar(row['image_score'], 'score')
    if len(seen) != len(sources)*3*3*2:
        raise ValueError('Missing paired source/level/seed/condition cells')
    ordered = sorted(seen, key=lambda x: (index[x[0]], levels.index(x[1]), x[2], CONDITIONS.index(x[3])))
    return {'sources': sources, 'levels': levels, 'mse': mse, 'scores': scores, 'row_count': len(seen),
            'allocation_sha256': allocation, 'normalized_rows': [seen[k] for k in ordered]}


def exploratory_arithmetic(panel, *, math, replicates=5000, return_replicates=False):
    """One source multiplicity shared across2 conditions x3seeds x3levels.

    Separate GDN exploratory95% intervals; never change f247 co-primary alpha.
    """
    sources = panel['sources']; weights, design = math.draw_source_weights(sources, replicates=replicates)
    allweights = np.vstack([np.ones((1, len(sources)), np.int64), weights])
    labels = np.array([r['original_image_label'] == 'anomaly' for r in sources], bool)
    values = {k: [] for k in ('mse_reduction', 'image_auroc_delta', 'image_ap_delta')}; detail = []
    for category in CATEGORIES:
        indices = np.array([i for i, row in enumerate(sources) if row['category'] == category]); w = allweights[:, indices]
        for li, level in enumerate(panel['levels']):
            for si, seed in enumerate(SEEDS):
                rank = [math._rank_rows(panel['scores'][ci, si, li, indices], labels[indices], w)[:2] for ci in range(2)]
                endpoints = ((w@(panel['mse'][1, si, li, indices]-panel['mse'][0, si, li, indices]))/w.sum(axis=1),
                             rank[0][0]-rank[1][0], rank[0][1]-rank[1][1])
                if any(not np.isfinite(x).all() for x in endpoints):
                    raise ValueError('Nonfinite resampled endpoint; no dropping')
                row = {'category': category, 'angle_deg': level, 'seed': seed}
                for key, endpoint in zip(values, endpoints):
                    values[key].append(endpoint); row[key] = float(endpoint[0])
                detail.append(row)
    result = {}; distributions = {}
    for key, pieces in values.items():
        macro = np.stack(pieces).mean(axis=0); distributions[key] = macro[1:].copy()
        seed_points = {str(seed): float(np.mean([r[key] for r in detail if r['seed'] == seed])) for seed in SEEDS}
        by_level = {}
        for level in panel['levels']:
            selected = [v for v, row in zip(pieces, detail) if row['angle_deg'] == level]
            level_macro = np.stack(selected).mean(axis=0)
            by_level[str(level)] = {'point': float(level_macro[0]), 'interval_95_exploratory': np.quantile(level_macro[1:], [.025, .975], method='linear').tolist()}
        result[key] = {'point': float(macro[0]), 'interval': np.quantile(macro[1:], [.025, .975], method='linear').tolist(),
                       'quantiles': [.025, .975], 'interval_coverage': .95, 'status': 'exploratory',
                       'seed_point_effects': seed_points, 'seed_point_range': [min(seed_points.values()), max(seed_points.values())],
                       'severity_specific_exploratory': by_level,
                       'positive_favors': 'full_maps',
                       'effect_definition': 'GDN MSE - full MSE' if key == 'mse_reduction' else 'full image rank metric - GDN image rank metric'}
    output = {'schema': 'gdn-start-exploratory-arithmetic-v1', 'arithmetic_only': True, 'formal_result_coverage_verified': False,
              'statistics_performed': False, 'source_count': len(sources), 'conditions': list(CONDITIONS), 'conditioned_seeds': list(SEEDS),
              'design': design, 'endpoints': result, 'point_by_category_level_seed': detail, 'numpy_version': np.__version__,
              'pairing': 'same source multiplicity across all3strengths/all3trainedseeds/bothconditions; equal category/strength/seed macro',
              'interpretation': 'conditional on realized fixed trained seeds/benchmark label prevalence; source-image cluster, physical board/session unknown; training-seed/nondeterministicCUDA population uncertainty not estimated',
              'pixel_bootstrap': False, 'p_values': False, 'must_win_rule': False, 'original_f247_modified': False}
    return (output, distributions) if return_replicates else output
