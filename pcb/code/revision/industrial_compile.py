#!/usr/bin/env python3
"""Read-only industrial preflight; explicit --write freezes allocation then compiles.

Only already inspected regular image/mask files are decoded, never archives.
This prototype rejects ALL exact file/decoded duplicates pending adjudication.
Source identity, simulated family and NPZ hashes live in separate manifests.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import csv
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OFFICIAL_COMMIT = '2a692ab575001cbde74d402d897a7286086c6199'
OFFICIAL_SPLIT_SHA = 'a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995'
PROTOCOL_FIELDS = ('schema_version', 'source_scope', 'official_split_csv', 'dataset_id', 'dataset_version',
                   'long_side', 'margin', 'horizontal_fov_degrees', 'exposure_seconds', 'map_scale_px',
                   'samples_per_interval', 'angle_levels_deg', 'profile_catalog', 'orientation_rule',
                   'split_seed', 'trajectory_seed')


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_module('industrial_compile_geometry', 'industrial_geometry.py')
float_io = load_module('industrial_compile_float_io', 'float_dataset.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_protocol(path):
    path = Path(path).resolve()
    protocol = strict_json(path.read_text())
    if not isinstance(protocol, dict) or set(protocol) != set(PROTOCOL_FIELDS) or protocol['schema_version'] != 'industrial-compile-v1':
        raise ValueError('Exact industrial-compile-v1 protocol required; paired profiles/extra options unsupported')
    if protocol['source_scope'] not in ('visa-pcb1-4-official', 'artificial-fixture'):
        raise ValueError('Explicit official or artificial-fixture source_scope required')
    for key in ('dataset_id', 'dataset_version', 'split_seed', 'trajectory_seed', 'official_split_csv'):
        if not isinstance(protocol[key], str) or not protocol[key].strip():
            raise ValueError('Nonempty protocol field required: '+key)
    if protocol['orientation_rule'] != 'sha256-pcg64-unit-quaternion-v1':
        raise ValueError('Only explicitly source-hashed PCG64 quaternion orientations are supported')
    for key in ('exposure_seconds', 'map_scale_px', 'horizontal_fov_degrees'):
        value = protocol[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite protocol scalar required: '+key)
    if protocol['horizontal_fov_degrees'] >= 180:
        raise ValueError('FOV must be below 180 degrees')
    if type(protocol['samples_per_interval']) is not int or protocol['samples_per_interval'] < 1:
        raise ValueError('Positive samples_per_interval integer required')
    levels = protocol['angle_levels_deg']
    if (not isinstance(levels, list) or not levels or any(type(a) not in (int, float) or not math.isfinite(a) or a <= 0 for a in levels)
            or len(set(levels)) != len(levels)):
        raise ValueError('Unique positive finite angle_levels_deg required')
    profiles = protocol['profile_catalog']
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('Named four-interval profile_catalog required')
    for name, rates in profiles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(rates, list):
            raise ValueError('Each profile requires a nonempty name and a four-interval template')
        _template(rates)
    # Validate dimensions/margin without reading a source or choosing a new ROI.
    geometry.roi_geometry(geometry.resize_geometry((protocol['long_side'], protocol['long_side']), protocol['long_side']), protocol['margin'])
    return protocol


def safe_source_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('Unsafe source path')
    rel = Path(relative)
    root = Path(root).resolve()
    path = root/rel
    if rel.is_absolute() or '..' in rel.parts or ':' in relative or not path.resolve().is_relative_to(root):
        raise ValueError('Source path traversal')
    cursor = root
    for part in rel.parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError('Source symlink refused')
    if not path.is_file():
        raise ValueError('Missing/nonregular source file: '+relative)
    return path


def checked_image(root, relative, info, *, mask=False):
    """Hash before/read/hash after; decode exactly the bytes whose digest matched."""
    if not isinstance(info, dict) or not valid_sha(info.get('file_sha256')) or not valid_sha(info.get('decoded_pixel_sha256')):
        raise ValueError('Inspected source file/pixel SHA required')
    path = safe_source_path(root, relative)
    expected = info['file_sha256']
    if sha(path) != expected:
        raise ValueError('Source file changed before read: '+relative)
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected or sha(path) != expected or len(value) != info['file_bytes']:
        raise ValueError('Source file changed during read: '+relative)
    with Image.open(io.BytesIO(value)) as opened:
        opened.load()
        array = np.asarray(opened)
        actual_pixel = hashlib.sha256((opened.mode+'|'+str(opened.size)+'|'+str(array.dtype)+'|').encode()+array.tobytes()).hexdigest()
        if actual_pixel != info['decoded_pixel_sha256'] or (opened.width, opened.height) != (info['width'], info['height']):
            raise ValueError('Decoded source differs from inspection: '+relative)
        if mask:
            if array.ndim != 2 or array.dtype.kind not in 'biu' or np.any(array < 0):
                raise ValueError('Original mask must retain nonnegative scalar classes')
            return array.copy()
        if opened.mode != 'RGB':
            raise ValueError('Source must be explicit RGB')
        return opened.copy()


def _row_key(row):
    return tuple(row[key] for key in ('object', 'split', 'label', 'image', 'mask'))


def source_inventory(summary_path, manifest_path, data_root, protocol, protocol_path):
    summary_path, manifest_path = Path(summary_path).resolve(), Path(manifest_path).resolve()
    summary = strict_json(summary_path.read_text())
    manifest_sha = sha(manifest_path)
    if summary.get('inspection_complete') is not True or summary.get('source_manifest_sha256') != manifest_sha:
        raise ValueError('Complete inspection and exact source-manifest SHA binding required')
    for field in ('official_split_sha256', 'full_archive_sha256', 'source_inspector_sha256'):
        if not valid_sha(summary.get(field)):
            raise ValueError('Completed official/archive source identity missing: '+field)
    if (not isinstance(summary.get('official_commit'), str) or len(summary['official_commit']) != 40
            or not summary.get('completed_utc') or not summary.get('archive_checksum_authority')):
        raise ValueError('Complete official commit/archive authority/completion identity required')
    if summary['source_inspector_sha256'] != sha(HERE.parents[1]/'tools/visa_source.py'):
        raise ValueError('Source inspector version does not match local inspected schema')
    split_path = Path(protocol['official_split_csv'])
    if not split_path.is_absolute():
        split_path = Path(protocol_path).resolve().parent/split_path
    if sha(split_path) != summary['official_split_sha256']:
        raise ValueError('Official split file SHA differs from inspection')
    if protocol['source_scope'] == 'visa-pcb1-4-official' and (summary['official_commit'] != OFFICIAL_COMMIT or summary['official_split_sha256'] != OFFICIAL_SPLIT_SHA):
        raise ValueError('Fixed official VisA commit/split identity differs')
    with split_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['object', 'split', 'label', 'image', 'mask']:
            raise ValueError('Unexpected official split columns')
        official = [row for row in reader if row['object'] in ('pcb1', 'pcb2', 'pcb3', 'pcb4')]
    rows = [strict_json(line) for line in manifest_path.read_text().splitlines()]
    if len({_row_key(row) for row in official}) != len(official) or sorted(map(_row_key, rows)) != sorted(map(_row_key, official)):
        raise ValueError('Source manifest does not preserve the exact official PCB source set')
    if len({row['image'] for row in rows}) != len(rows) or len(rows) != summary.get('selected_source_images'):
        raise ValueError('Source count or image ID duplicates differ')
    if summary.get('selected_stored_masks') != sum(row['mask_info'] is not None for row in rows):
        raise ValueError('Source mask count differs from complete inspection')
    counts = dict(sorted(Counter(row['object']+'|'+row['split']+'|'+row['label'] for row in rows).items()))
    if counts != summary.get('counts') or summary.get('annotation_label_mismatch_count') != 0:
        raise ValueError('Source category/split/label counts or annotation consistency differ')
    test_count = sum(row['split'] == 'test' for row in rows)
    if protocol['source_scope'] == 'visa-pcb1-4-official' and test_count != 802:
        raise ValueError('All 802 official TEST source images must remain')
    duplicates = []
    for field in ('file_sha256', 'decoded_pixel_sha256'):
        families = defaultdict(list)
        for row in rows:
            families[row['image_info'][field]].append(row['image'])
        duplicates += [(field, key, values) for key, values in families.items() if len(values) > 1]
    if duplicates:
        raise ValueError('Exact source duplicate families require adjudication; no silent deletion or split: '+repr(duplicates[:5]))
    for row in rows:
        if (row.get('source_image_id') != row['image'] or row.get('group_kind') != 'source_image'
                or any(row.get(key, 'missing') is not None for key in ('physical_scene_id', 'physical_entity_id', 'session_id'))):
            raise ValueError('Source physical scene/entity/session must explicitly remain null')
        if row['split'] not in ('train', 'test') or row['label'] not in ('normal', 'anomaly') or (row['split'] == 'train' and row['label'] != 'normal'):
            raise ValueError('Official one-class split/label contract violated')
        if row['image_info'].get('path') != row['image'] or (row['mask_info'] is not None and row['mask_info'].get('path') != row['mask']):
            raise ValueError('Inspection file path differs from official source row')
        image = checked_image(data_root, row['image'], row['image_info'])
        mapping = geometry.roi_geometry(geometry.resize_geometry(image.size, protocol['long_side']), protocol['margin'])
        # Decode color during preflight, so invalid ICC/modes cannot create NPZ.
        geometry.decode_rgb(image)
        if row['label'] == 'normal':
            if row['mask'] or row['mask_info'] is not None:
                raise ValueError('Normal source unexpectedly has a stored defect mask')
        else:
            mask = checked_image(data_root, row['mask'], row['mask_info'], mask=True)
            if mask.shape != (image.height, image.width) or not mask.any():
                raise ValueError('Anomaly mask empty or dimensions differ')
        del image
    return rows, summary, {'summary_sha256': sha(summary_path), 'source_manifest_sha256': manifest_sha,
        'official_split_sha256': sha(split_path), 'official_commit': summary['official_commit'],
        'full_archive_sha256': summary['full_archive_sha256'], 'official_test_sources': test_count,
        'official_split_csv_resolved': str(split_path.resolve()), 'source_manifest_resolved': str(manifest_path),
        'inspection_summary_resolved': str(summary_path)}


def _template(rates):
    raw = np.asarray(rates, dtype=object)
    if any(type(value) not in (int, float) for value in raw.flat):
        raise ValueError('Profile coefficients must be numeric scalars, not booleans/strings')
    try:
        array = np.asarray(rates, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('Profile must be numeric 4-scalar or 4x3 template') from error
    if array.shape == (4,):
        array = np.stack([array, np.zeros(4), np.zeros(4)], axis=1)
    if array.shape != (4, 3) or not np.isfinite(array).all():
        raise ValueError('Profile must be finite 4-scalar or 4x3 template')
    length = float(np.linalg.norm(array, axis=1).mean())
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Profile total angular path length must be positive')
    return array/length


def _orientation(seed):
    rng = np.random.Generator(np.random.PCG64(int(digest(seed)[:32], 16)))
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return q.tolist(), rotation


def _choices(row, split, protocol, protocol_sha):
    family = 'simulated:'+digest([protocol['trajectory_seed'], protocol_sha, split, row['source_image_id'], row['image_info']['decoded_pixel_sha256']])
    quaternion, rotation = _orientation([protocol['trajectory_seed'], protocol_sha, split, row['image_info']['decoded_pixel_sha256']])
    profile_names = sorted(protocol['profile_catalog'])
    profile = profile_names[int(digest([family, 'profile']), 16) % len(profile_names)]
    normalized = _template(protocol['profile_catalog'][profile])
    base_rates = normalized@rotation.T/protocol['exposure_seconds']
    levels = protocol['angle_levels_deg']
    selected = [levels[int(digest([family, 'strength']), 16) % len(levels)]] if split == 'train' else levels
    return family, profile, quaternion, base_rates, selected


def preflight(summary_path, source_manifest_path, data_root, protocol_path):
    """Read every inspected source; build deterministic allocation without writing."""
    protocol_path = Path(protocol_path).resolve()
    protocol = read_protocol(protocol_path)
    protocol_sha = sha(protocol_path)
    rows, summary, source = source_inventory(summary_path, source_manifest_path, data_root, protocol, protocol_path)
    assignments = {row['source_image_id']: 'test' for row in rows if row['split'] == 'test'}
    for category in sorted({row['object'] for row in rows}):
        normals = [row for row in rows if row['object'] == category and row['split'] == 'train']
        if len(normals) < 2:
            raise ValueError('Each category requires >=2 normal train sources for train/valid')
        ordered = sorted(normals, key=lambda row: digest([protocol['split_seed'], category, row['image_info']['decoded_pixel_sha256']]))
        nvalid = math.ceil(.2*len(ordered))
        for index, row in enumerate(ordered):
            assignments[row['source_image_id']] = 'valid' if index < nvalid else 'train'
    samples, recipes, base_seen, actual_seen = [], {}, {}, {}
    renderer = geometry._renderer()
    for row in sorted(rows, key=lambda row: row['source_image_id']):
        split = assignments[row['source_image_id']]
        family, profile, quaternion, base, levels = _choices(row, split, protocol, protocol_sha)
        base_sha = digest(base.tolist())
        if base_sha in base_seen:
            raise ValueError('Actual base trajectory repeated across source families')
        base_seen[base_sha] = family
        for angle in levels:
            sid = 'industrial-'+digest([family, angle])
            provenance = dict(sample_id=sid, source_dataset_id=protocol['dataset_id'], source_dataset_version=protocol['dataset_version'],
                source_image_id=row['source_image_id'], group_kind='source_image', group_id=row['source_image_id'],
                trajectory_family_id=family, trajectory_id='simulated:'+digest([family, angle]),
                original_split_id=summary['official_split_sha256']+':'+row['split'], split_name=split,
                physical_scene_id=None, physical_session_id=None)
            float_io.validate_provenance(dict(provenance, split_manifest_sha256='0'*64))
            samples.append(dict(provenance, npz='samples/'+sid+'.npz'))
            rates = (base*math.radians(angle)).tolist()
            poses = renderer.integrate_body_gyro(np.linspace(0, protocol['exposure_seconds'], 5), rates)
            rate_sha, pose_sha = digest(rates), digest(poses.tolist())
            for kind, value in (('rates', rate_sha), ('poses', pose_sha)):
                if (kind, value) in actual_seen and actual_seen[kind, value] != family:
                    raise ValueError('Actual rate/pose curve reused across source families')
                actual_seen[kind, value] = family
            recipes[sid] = dict(source=row, profile=profile, quaternion_wxyz=quaternion, angle_deg=angle,
                base_rate_sha256=base_sha, rates_rad_s=rates, rates_sha256=rate_sha, poses_sha256=pose_sha)
    allocation = {'allocation_schema': 'imda-float-allocation-v2',
        'allocation_id': 'industrial-'+digest([source, protocol_sha]), 'samples': samples}
    return {'protocol': protocol, 'protocol_sha256': protocol_sha, 'source': source, 'allocation': allocation,
            'recipes': recipes, 'data_root': str(Path(data_root).resolve()),
            'split_source_counts': dict(Counter(assignments.values())),
            'npz_split_counts': dict(Counter(x['split_name'] for x in samples)),
            'code_sha256': {name: sha(HERE/name) for name in ('industrial_compile.py', 'industrial_geometry.py', 'imu_synthesis.py', 'float_dataset.py')}}


def _write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)+b'\n')


def _compile_one_entry(entry, recipe, protocol, data_root, output, allocation_sha,
                       protocol_sha256, inspection_identity, code_sha256):
    """Render/write/check one already allocated entry; no allocation or preflight."""
    output = Path(output)
    row = recipe['source']
    image = checked_image(data_root, row['image'], row['image_info'])
    mask = (checked_image(data_root, row['mask'], row['mask_info'], mask=True) if row['mask']
            else np.zeros((image.height, image.width), dtype=np.uint8))
    result, target, annotation = geometry.prepare_industrial_sample(image, mask, row['label'],
        long_side=protocol['long_side'], margin=protocol['margin'],
        times_seconds=np.linspace(0, protocol['exposure_seconds'], 5).tolist(),
        angular_rates_rad_s=recipe['rates_rad_s'], horizontal_fov_degrees=protocol['horizontal_fov_degrees'],
        samples_per_interval=protocol['samples_per_interval'], map_scale_px=protocol['map_scale_px'])
    source_record = {'image': row['image'], 'image_file_sha256': row['image_info']['file_sha256'],
        'image_decoded_pixel_sha256': row['image_info']['decoded_pixel_sha256'], 'mask': row['mask'] or None,
        'mask_file_sha256': row['mask_info']['file_sha256'] if row['mask_info'] else None,
        'original_image_label': row['label'], 'original_split': row['split'], 'category': row['object']}
    result['metadata']['input_source'].update(source=source_record, protocol_sha256=protocol_sha256,
        compiler_identity={'source_scope': protocol['source_scope'], 'inspection': inspection_identity,
                           'code_sha256': code_sha256},
        trajectory={'profile': recipe['profile'], 'quaternion_wxyz': recipe['quaternion_wxyz'],
                    'angular_path_length_deg': recipe['angle_deg'], 'base_rate_sha256': recipe['base_rate_sha256'],
                    'rates_rad_s': recipe['rates_rad_s'], 'rates_sha256': recipe['rates_sha256'], 'poses_sha256': recipe['poses_sha256']})
    minimal = {key: result[key] for key in float_io.REQUIRED_ARRAYS}
    minimal['metadata'] = result['metadata']
    provenance = {key: value for key, value in entry.items() if key != 'npz'}
    provenance['split_manifest_sha256'] = allocation_sha
    path = output/entry['npz']
    float_io.write_npz_sample(path, minimal, provenance, target)
    checked = float_io.load_npz_sample(path, require_target=True, expected_manifest_sha256=allocation_sha)
    if checked['metadata']['provenance'] != provenance:
        raise ValueError('Compiled provenance differs')
    record = dict(sample_id=entry['sample_id'], npz=entry['npz'], npz_sha256=sha(path),
        source=source_record, annotation=annotation['statistics'], trajectory_family_id=entry['trajectory_family_id'],
        source_mapping=result['metadata']['input_source']['mapping'],
        trajectory_base_rate_sha256=recipe['base_rate_sha256'],
        rates_sha256=recipe['rates_sha256'], poses_sha256=recipe['poses_sha256'])
    return record, os.getpid()


def _entry_arguments(plan, entry, output, allocation_sha):
    return (entry, plan['recipes'][entry['sample_id']], plan['protocol'], plan['data_root'],
            str(output.resolve()), allocation_sha, plan['protocol_sha256'], plan['source'], plan['code_sha256'])


def _ordered_entry_results(plan, output, allocation_sha, workers):
    """At most workers in flight; parent consumes results in frozen allocation order."""
    entries = plan['allocation']['samples']
    if workers == 1:
        for entry in entries:
            yield _compile_one_entry(*_entry_arguments(plan, entry, output, allocation_sha))
        return
    # Spawn needs an importable function even when a caller used spec_from_file_location.
    # Keep this explicit path until the executor closes so spawned interpreters inherit it.
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(HERE))
        worker_module = importlib.import_module('industrial_compile')
        if Path(worker_module.__file__).resolve() != Path(__file__).resolve():
            raise ValueError('Spawn worker module resolves to a different compiler')
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            next_submit = 0
            try:
                for index in range(min(workers, len(entries))):
                    pending[index] = pool.submit(worker_module._compile_one_entry,
                        *_entry_arguments(plan, entries[index], output, allocation_sha))
                    next_submit += 1
                for index in range(len(entries)):
                    yield pending.pop(index).result()
                    if next_submit < len(entries):
                        pending[next_submit] = pool.submit(worker_module._compile_one_entry,
                            *_entry_arguments(plan, entries[next_submit], output, allocation_sha))
                        next_submit += 1
            except BaseException:
                for future in pending.values():
                    future.cancel()
                raise
    finally:
        sys.path[:] = old_path


def compile_dataset(summary_path, source_manifest_path, data_root, protocol_path, *, output_dir=None, workers=1):
    """Default only preflights. Explicit new output freezes allocation BEFORE NPZ."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer in [1,4]')
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError('Compile output must be a new directory')
    plan = preflight(summary_path, source_manifest_path, data_root, protocol_path)
    report = {key: plan[key] for key in ('source', 'protocol_sha256', 'split_source_counts', 'npz_split_counts', 'code_sha256')}
    report['execution'] = {'workers': workers, 'start_method': 'spawn' if workers > 1 else 'inline',
                           'max_in_flight_entries': workers, 'record_order': 'frozen allocation order',
                           'participating_worker_process_ids': [], 'participating_worker_count': 0,
                           'worker_pid_scope': 'committed entries only; failed/uncommitted child PIDs may be absent'}
    report['environment'] = {'numpy': np.__version__, 'Pillow': Image.__version__}
    report['protocol_source_resolved'] = str(Path(protocol_path).resolve())
    if output_dir is None:
        return dict(report, preflight_passed=True, wrote_files=False, complete=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'samples').mkdir()
    _write_new(output/'allocation.json', plan['allocation'])
    allocation_sha = sha(output/'allocation.json')
    with (output/'protocol.json').open('xb') as copied_protocol:
        copied_protocol.write(Path(protocol_path).read_bytes())
    completed = 0
    worker_pids = set()
    started = time.monotonic()
    try:
        with (output/'compile-records.jsonl').open('x') as records:
            with closing(_ordered_entry_results(plan, output, allocation_sha, workers)) as results:
                for record, worker_pid in results:
                    entry = plan['allocation']['samples'][completed]
                    if record['sample_id'] != entry['sample_id'] or record['npz'] != entry['npz']:
                        raise ValueError('Worker result differs from frozen allocation order')
                    records.write(canonical(record).decode()+'\n')
                    records.flush()
                    completed += 1
                    worker_pids.add(worker_pid)
                    if completed == 1 or completed % 25 == 0 or completed == len(plan['allocation']['samples']):
                        print(json.dumps({'event': 'compile_progress', 'compiled_samples': completed,
                            'allocated_samples': len(plan['allocation']['samples']), 'workers': workers,
                            'elapsed_seconds': time.monotonic()-started, 'sample_id': entry['sample_id']}), flush=True)
        # Immutable inputs/code must remain fixed through the entire compilation.
        if sha(protocol_path) != plan['protocol_sha256'] or any(sha(HERE/name) != value for name, value in plan['code_sha256'].items()):
            raise ValueError('Protocol/compiler dependency changed during compile')
        if sha(source_manifest_path) != plan['source']['source_manifest_sha256'] or sha(summary_path) != plan['source']['summary_sha256']:
            raise ValueError('Inspection inputs changed during compile')
        report['execution'].update(participating_worker_process_ids=sorted(worker_pids),
            participating_worker_count=len(worker_pids), elapsed_seconds=time.monotonic()-started)
        final = dict(report, complete=True, wrote_files=True, compiled_samples=completed, allocation_sha256=allocation_sha,
            compile_records_sha256=sha(output/'compile-records.jsonl'), source_scope=plan['protocol']['source_scope'])
        _write_new(output/'compile-summary.json', final)
        return final
    except BaseException as error:
        _write_new(output/'partial.json', dict(complete=False, compiled_samples=completed,
            allocation_sha256=allocation_sha, allocated_samples=len(plan['allocation']['samples']),
            execution=dict(report['execution'], participating_worker_process_ids=sorted(worker_pids),
                           participating_worker_count=len(worker_pids)),
            sample_npz_files_present=len(list((output/'samples').glob('*.npz'))),
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('summary', 'source-manifest', 'data-root', 'protocol'):
        parser.add_argument('--'+argument, type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1, help='CPU entry workers; 1 inline, 2..4 spawn')
    parser.add_argument('--write', type=Path, help='Explicit NEW output directory; absent means read-only preflight')
    args = parser.parse_args()
    print(json.dumps(compile_dataset(args.summary, args.source_manifest, args.data_root, args.protocol, output_dir=args.write, workers=args.workers), indent=2))
