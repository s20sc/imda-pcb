#!/usr/bin/env python3
"""Pinned VisA source acquisition and safe PCB inventory; never execute archives."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
import time
import urllib.request

ARCHIVE_URL = 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'
ARCHIVE_SIZE = 1929840640
OFFICIAL_COMMIT = '2a692ab575001cbde74d402d897a7286086c6199'
OFFICIAL_SPLIT_SHA256 = 'a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995'
PCB_CATEGORIES = ('pcb1', 'pcb2', 'pcb3', 'pcb4')
SPLIT_FIELDS = ['object', 'split', 'label', 'image', 'mask']
# Literal PCB subset of pinned official utils/id2class.py, read as text, never executed.
OFFICIAL_CLASSMAP_SHA256 = '4b862131cb8c080f864815fb16ab906165691618fd4734edf35131b759eb25e3'
PCB_CLASS_NAMES = {
    **{c: {1: 'bent', 2: 'melt', 3: 'scratch', 4: 'missing'} for c in ('pcb1', 'pcb2', 'pcb3')},
    'pcb4': {1: 'burnt', 2: 'scratch', 3: 'missing', 4: 'damage',
             5: 'extra', 6: 'wrong place', 7: 'dirt'},
}
MAX_MEMBER_BYTES = 64*1024*1024
MAX_IMAGE_PIXELS = 32_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


LOADED_SOURCE_SHA256 = sha256_file(Path(__file__))


def fetch_archive(output, workers=16):
    output = Path(output)
    if not 1 <= workers <= 16 or output.suffix != '.tar':
        raise ValueError('Expected .tar output and 1..16 workers')
    output.parent.mkdir(parents=True, exist_ok=True)
    source_sha = LOADED_SOURCE_SHA256
    with urllib.request.urlopen(urllib.request.Request(ARCHIVE_URL, method='HEAD'), timeout=60) as response:
        if response.status != 200 or int(response.headers.get('Content-Length', -1)) != ARCHIVE_SIZE or response.headers.get('Accept-Ranges') != 'bytes':
            raise ValueError('Official archive HEAD identity/range support differs')
        etag = response.headers.get('ETag')
        headers = dict(response.headers)
    sidecar = output.with_suffix('.tar.json')
    if output.exists():
        if not sidecar.exists():
            raise FileExistsError('Archive exists without verified completion metadata')
        old = json.loads(sidecar.read_text())
        if output.stat().st_size != ARCHIVE_SIZE or sha256_file(output) != old['downloaded_archive_sha256']:
            raise ValueError('Existing archive length/SHA differs')
        print(json.dumps(dict(old, reused=True)), flush=True)
        return old
    lock = output.with_suffix('.download.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Download lock exists; inspect its PID/actual process before any resume: '+str(lock))
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    try:
        parts = output.with_suffix('.tar.parts')
        parts.mkdir(exist_ok=True)
        chunk = 8*1024*1024
        spans = [(i, start, min(ARCHIVE_SIZE, start+chunk)-1)
                 for i, start in enumerate(range(0, ARCHIVE_SIZE, chunk))]
        started = time.monotonic()

        def download(span):
            number, start, end = span
            data_path = parts/('%05d.bin' % number)
            record_path = data_path.with_suffix('.json')
            expected = 'bytes %d-%d/%d' % (start, end, ARCHIVE_SIZE)
            if data_path.exists() and record_path.exists():
                record = json.loads(record_path.read_text())
                if data_path.stat().st_size == end-start+1 and record['content_range'] == expected and record['etag'] == etag and sha256_file(data_path) == record['sha256']:
                    return dict(record, reused=True)
                raise ValueError('Existing part integrity/ETag differs: %d' % number)
            error = None
            for attempt in range(1, 5):
                try:
                    request_headers = {'Range': 'bytes=%d-%d' % (start, end), 'Accept-Encoding': 'identity'}
                    if etag:
                        request_headers['If-Match'] = etag
                    with urllib.request.urlopen(urllib.request.Request(ARCHIVE_URL, headers=request_headers), timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != expected or response.headers.get('ETag') != etag:
                            raise ValueError('Exact range/ETag not honored')
                        data = response.read(end-start+2)
                        if len(data) != end-start+1:
                            raise ValueError('Truncated/oversized range body')
                    record = {'part': number, 'bytes': len(data), 'content_range': expected,
                              'etag': etag, 'http_status': 206, 'attempt': attempt,
                              'completed_utc': datetime.now(timezone.utc).isoformat(),
                              'sha256': hashlib.sha256(data).hexdigest()}
                    if data_path.exists() and sha256_file(data_path) != record['sha256']:
                        raise FileExistsError('Refusing different existing part')
                    temporary = data_path.with_suffix('.partial')
                    temporary.write_bytes(data)
                    temporary.replace(data_path)
                    record_path.write_text(json.dumps(record, indent=2)+'\n')
                    return record
                except Exception as exc:
                    error = exc
                    print(json.dumps({'event': 'part_retry', 'part': number, 'attempt': attempt, 'error': repr(exc)}), flush=True)
                    if attempt < 4:
                        time.sleep(min(2**attempt, 8))
            raise RuntimeError('Part exhausted four attempts: %d %r' % (number, error))

        records = []
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(download, span) for span in spans]
        try:
            for future in as_completed(futures):
                records.append(future.result())
                print(json.dumps({'event': 'download_progress', 'parts': len(records), 'total_parts': len(spans),
                                  'verified_bytes': sum(r['bytes'] for r in records), 'total_bytes': ARCHIVE_SIZE,
                                  'elapsed_seconds': time.monotonic()-started}), flush=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        assembled = output.with_suffix('.assembling')
        with assembled.open('wb') as stream:
            for number, _, _ in spans:
                with (parts/('%05d.bin' % number)).open('rb') as part:
                    for block in iter(lambda: part.read(1024*1024), b''):
                        stream.write(block)
        if assembled.stat().st_size != ARCHIVE_SIZE:
            raise ValueError('Assembled archive length differs')
        digest = sha256_file(assembled)
        assembled.replace(output)
        result = {'complete': True, 'url': ARCHIVE_URL, 'bytes': ARCHIVE_SIZE,
                  'downloaded_archive_sha256': digest, 'checksum_authority': 'computed by this task; not an official published checksum',
                  'HEAD_headers': headers, 'etag_is_not_assumed_MD5': True,
                  'download_source_sha256': source_sha, 'workers': workers,
                  'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic()-started,
                  'parts': sorted(records, key=lambda r: r['part'])}
        sidecar.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'parts'}), flush=True)
        return result
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid') == os.getpid():
            lock.unlink()


def safe_name(name):
    if not isinstance(name, str) or '\x00' in name or '\\' in name or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError('Unsafe archive/source path')
    parts = PurePosixPath(name).parts
    if '..' in parts:
        raise ValueError('Archive/source path traversal')
    return '/'.join(part for part in parts if part != '.')


def split_rows(path, categories=PCB_CATEGORIES):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SPLIT_FIELDS:
            raise ValueError('Unexpected fixed official split columns')
        all_rows = list(reader)
    selected, seen = [], set()
    for original in all_rows:
        if original['object'] not in categories:
            continue
        row = dict(original)
        category = row['object']
        if row['split'] not in ('train', 'test') or row['label'] not in ('normal', 'anomaly'):
            raise ValueError('Unknown split/label in official PCB row')
        row['image'] = safe_name(row['image'])
        if not row['image'].startswith(category+'/Data/Images/'+row['label'].title()+'/') or row['image'] in seen:
            raise ValueError('Duplicate/image label/path mismatch')
        seen.add(row['image'])
        if row['label'] == 'normal':
            if row['mask']:
                raise ValueError('Normal row unexpectedly declares a stored mask')
        else:
            row['mask'] = safe_name(row['mask'])
            if not row['mask'].startswith(category+'/Data/Masks/Anomaly/'):
                raise ValueError('Anomaly lacks a corresponding whitelisted mask')
        if row['split'] == 'train' and row['label'] != 'normal':
            raise ValueError('Official one-class train unexpectedly includes anomalies')
        selected.append(row)
    if not selected:
        raise ValueError('No selected industrial source rows')
    return selected, {'all_official_rows': len(all_rows), 'columns': SPLIT_FIELDS,
                      'selected_rows': len(selected),
                      'counts': dict(sorted(Counter(r['object']+'|'+r['split']+'|'+r['label'] for r in selected).items())),
                      'physical_scene_session_fields_present': False,
                      'grouping': 'source_image; physical PCB entity/scene/session unknown'}


def safe_write(root, relative, value):
    root = Path(root).resolve()
    relative = safe_name(relative)
    if not relative:
        raise ValueError('Empty file path')
    destination = root/relative
    if not destination.resolve().is_relative_to(root):
        raise ValueError('Destination escapes controlled data root')
    for parent in [destination]+list(destination.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink in controlled extraction target')
    digest = hashlib.sha256(value).hexdigest()
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise FileExistsError('Refusing different existing content: '+str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name+'.visa-part-'+str(os.getpid()))
        try:
            with temporary.open('xb') as stream:
                stream.write(value)
            os.link(temporary, destination)  # atomic and refuses overwrite
        finally:
            if temporary.exists():
                temporary.unlink()
    return digest


def inspect_image(value, is_mask=False):
    import numpy as np
    from PIL import Image, ExifTags
    with Image.open(io.BytesIO(value)) as image:
        if image.width*image.height > MAX_IMAGE_PIXELS or min(image.size) <= 0:
            raise ValueError('Decoded pixel budget exceeded')
        image.load()  # actual decode, not only a header probe
        array = np.asarray(image)
        profile = image.info.get('icc_profile')
        info = {'mode': image.mode, 'format': image.format, 'width': image.width, 'height': image.height,
                'array_dtype': str(array.dtype), 'array_shape': list(array.shape),
                'icc_present': 'icc_profile' in image.info, 'icc_bytes': len(profile) if profile else 0,
                'icc_sha256': hashlib.sha256(profile).hexdigest() if profile else None,
                'exif_tag_names': sorted(ExifTags.TAGS.get(k, str(k)) for k in image.getexif().keys()),
                'declared_srgb_tag': image.info.get('srgb'), 'declared_gamma': image.info.get('gamma'),
                'decoded_pixel_sha256': hashlib.sha256((image.mode+'|'+str(image.size)+'|'+str(array.dtype)+'|').encode()+array.tobytes()).hexdigest()}
        if profile:
            try:
                from PIL import ImageCms
                info['icc_description'] = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip()
                info['icc_parse_error'] = None
            except Exception as exc:
                info['icc_description'] = None
                info['icc_parse_error'] = repr(exc)
        if is_mask:
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) and array.dtype != np.bool_:
                raise ValueError('Mask must expose original scalar class IDs; no silent RGB/color conversion')
            values, counts = np.unique(array, return_counts=True)
            if np.any(values < 0):
                raise ValueError('Negative mask class ID')
            info.update(raw_value_counts={str(int(v)): int(n) for v, n in zip(values, counts)},
                        positive_pixels=int(np.count_nonzero(array)),
                        binary_semantics='nonzero raw class ID denotes anomaly; original file unchanged')
        return info


def annotation_metadata(value, category, selected, records):
    """Compare original multiclass annotations with official binary labels and masks."""
    text = value.decode('utf-8-sig')
    if '\x00' in text:
        raise ValueError('Binary annotation metadata rejected')
    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    if columns != ['image', 'label', 'mask']:
        raise ValueError('Original annotation schema differs')
    rows, by_image = list(reader), {}
    for row in rows:
        path = safe_name(row['image'])
        if not path.startswith(category+'/'):
            path = category+'/'+path
        if path in by_image:
            raise ValueError('Repeated image in category annotation metadata')
        by_image[path] = row
    compared = [r for r in selected if r['object'] == category]
    if set(by_image) != {r['image'] for r in compared}:
        raise ValueError('Annotation image membership differs from official category list')
    annotations, literal_differences = {}, []
    for official in compared:
        row = by_image[official['image']]
        raw_label = row['label']
        tokens = [t.strip().lower() for t in raw_label.split(',')]
        if (not all(tokens) or len(tokens) != len(set(tokens)) or
                ('normal' in tokens and tokens != ['normal']) or
                (tokens != ['normal'] and not set(tokens) <= set(PCB_CLASS_NAMES[category].values()))):
            raise ValueError('Unknown/mixed/duplicate original defect label tokens: '+repr(raw_label))
        binary_label = 'normal' if tokens == ['normal'] else 'anomaly'
        if binary_label != official['label']:
            raise ValueError('Annotation binary label differs from official split: '+official['image'])
        mask_path = safe_name(row['mask']) if row['mask'] else ''
        if mask_path and not mask_path.startswith(category+'/'):
            mask_path = category+'/'+mask_path
        if mask_path != official['mask']:
            raise ValueError('Annotation mask path differs from official split: '+official['image'])
        ids = sorted(int(k) for k in records[mask_path]['raw_value_counts'] if int(k) != 0) if mask_path else []
        if binary_label == 'anomaly' and not ids:
            raise ValueError('Anomaly label has an empty/missing mask')
        if any(k not in PCB_CLASS_NAMES[category] for k in ids):
            raise ValueError('Unknown original mask class ID: '+official['image'])
        mask_classes = [PCB_CLASS_NAMES[category][k] for k in ids]
        if set(mask_classes) != (set(tokens) if binary_label == 'anomaly' else set()):
            raise ValueError('Original label tokens differ from mask class IDs: '+official['image'])
        if raw_label != official['label']:
            literal_differences.append(official['image'])
        annotations[official['image']] = {
            'raw_label': raw_label, 'raw_label_tokens': tokens, 'mapped_binary_label': binary_label,
            'original_mask_path': row['mask'], 'canonical_mask_path': mask_path,
            'nonzero_mask_class_ids': ids, 'mask_class_names': mask_classes,
        }
    return {'columns': columns, 'rows': len(rows),
            'scene_like_columns': [k for k in columns if re.search(r'scene|session|sequence|board_id|part_id|capture_id|lot_id', k, re.I)],
            'official_label_comparison': 'exact after documented multiclass-to-binary mapping',
            'label_mapping': 'normal alone -> normal; one or more known defect tokens -> anomaly',
            'raw_label_counts': dict(Counter(row['label'] for row in rows)),
            'class_names_source': 'pinned official utils/id2class.py (PCB subset, lower case)',
            'class_names_source_sha256': OFFICIAL_CLASSMAP_SHA256,
            'compared_images': len(compared), 'label_mismatches': [],
            'mask_path_mismatches': [], 'label_token_mask_class_mismatches': [],
            'literal_official_label_differences': literal_differences,
            'image_annotations': annotations}


def inspect_tar(archive, rows, data_root, evidence_dir):
    """Helper also accepts tiny artificial tar fixtures for safety tests."""
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    requested = {r['image'] for r in rows} | {r['mask'] for r in rows if r['mask']}
    records, metadata, inventory = {}, {}, []
    with tarfile.open(archive, mode='r:') as source:
        members = source.getmembers()
        named = {}
        for member in members:
            name = safe_name(member.name)
            inventory.append({'name': name, 'size': member.size, 'type': member.type.decode('ascii', errors='replace')})
            if member.isdir():
                continue
            if name in named:
                raise ValueError('Duplicate normalized tar member')
            named[name] = member
        anchor = next(iter(sorted(requested)))
        prefixes = ['' if name == anchor else name[:-len(anchor)] for name in named if name == anchor or name.endswith('/'+anchor)]
        if len(prefixes) != 1:
            raise ValueError('Missing/ambiguous archive root for fixed official paths')
        prefix = prefixes[0]
        missing = sorted(p for p in requested if prefix+p not in named)
        if missing:
            raise ValueError('Official source members missing: '+repr(missing[:10]))
        allowed_metadata = {category+'/'+filename for category in PCB_CATEGORIES
                            for filename in ('image_anno.csv', 'image_annot.csv')}
        selected_paths = sorted(requested | {p for p in allowed_metadata if prefix+p in named})
        total_bytes = 0
        for number, path in enumerate(selected_paths, 1):
            member = named[prefix+path]
            if not member.isfile() or member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise ValueError('Nonregular or oversized whitelisted member')
            total_bytes += member.size
            if total_bytes > ARCHIVE_SIZE:
                raise ValueError('Selected extraction byte budget exceeded')
            with source.extractfile(member) as stream:
                value = stream.read(MAX_MEMBER_BYTES+1)
            if len(value) != member.size:
                raise ValueError('Truncated/oversized member')
            digest = safe_write(data_root, path, value)
            if path in allowed_metadata:
                metadata[path] = dict(annotation_metadata(value, path.split('/')[0], rows, records), sha256=digest)
            else:
                info = inspect_image(value, is_mask='/Masks/' in path)
                records[path] = dict(info, path=path, file_sha256=digest, file_bytes=len(value))
            if number % 500 == 0:
                print(json.dumps({'event': 'inspect_progress', 'members': number, 'total_selected_members': len(selected_paths)}), flush=True)
    if set(metadata) != {c+'/image_anno.csv' for c in {r['object'] for r in rows}}:
        raise ValueError('Missing/ambiguous original category annotation file')
    manifest = []
    for row in rows:
        image = records[row['image']]
        mask = records[row['mask']] if row['mask'] else None
        if mask and (mask['width'], mask['height']) != (image['width'], image['height']):
            raise ValueError('Image/mask dimensions differ: '+row['image'])
        if row['label'] == 'anomaly' and (mask is None or mask['positive_pixels'] == 0):
            raise ValueError('Anomaly label has an empty/missing mask')
        manifest.append(dict(row, source_image_id=row['image'], group_kind='source_image',
                             source_annotation=metadata[row['object']+'/image_anno.csv']['image_annotations'][row['image']],
                             physical_scene_id=None, physical_entity_id=None, session_id=None,
                             image_info=image, mask_info=mask,
                             normal_mask_policy='implicit all-zero; file absent as declared officially' if mask is None else None))
    duplicates = {}
    for key in ('file_sha256', 'decoded_pixel_sha256'):
        groups = defaultdict(list)
        for r in manifest:
            groups[r['image_info'][key]].append({'image': r['image'], 'object': r['object'], 'split': r['split'], 'label': r['label']})
        duplicates[key] = [{'sha256': digest, 'images': values, 'label_conflict': len({v['label'] for v in values}) > 1}
                           for digest, values in groups.items() if len({v['split'] for v in values}) > 1]
    (evidence_dir/'pcb-source-manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in manifest))
    (evidence_dir/'archive-inventory.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in inventory))
    (evidence_dir/'pcb-annotation-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (evidence_dir/'exact-cross-split-duplicates.json').write_text(json.dumps(duplicates, indent=2)+'\n')
    summary = {'inspection_complete': True, 'selected_source_images': len(manifest),
               'selected_stored_masks': sum(r['mask_info'] is not None for r in manifest),
               'selected_annotation_files': len(metadata), 'archive_entries': len(inventory),
               'archive_prefix_for_official_paths': prefix, 'selected_file_bytes': total_bytes,
               'counts': dict(sorted(Counter(r['object']+'|'+r['split']+'|'+r['label'] for r in manifest).items())),
               'image_modes': dict(Counter(r['image_info']['mode'] for r in manifest)),
               'image_dimensions_wh': dict(Counter(str(r['image_info']['width'])+'x'+str(r['image_info']['height']) for r in manifest)),
               'images_with_ICC': sum(r['image_info']['icc_present'] for r in manifest),
               'ICC_profile_sha256_counts': dict(Counter(r['image_info']['icc_sha256'] for r in manifest if r['image_info']['icc_present'])),
               'ICC_description_counts': dict(Counter(r['image_info'].get('icc_description', 'not present') or 'unparsed' for r in manifest)),
               'images_with_ICC_parse_error': sum(bool(r['image_info'].get('icc_parse_error')) for r in manifest),
               'image_EXIF_tag_names': sorted({tag for r in manifest for tag in r['image_info']['exif_tag_names']}),
               'mask_modes': dict(Counter(r['mask_info']['mode'] for r in manifest if r['mask_info'])),
               'raw_mask_values_by_category': {c: sorted({int(k) for r in manifest if r['object'] == c and r['mask_info'] for k in r['mask_info']['raw_value_counts']}) for c in PCB_CATEGORIES},
               'cross_split_duplicate_file_groups': len(duplicates['file_sha256']),
               'cross_split_duplicate_decoded_pixel_groups': len(duplicates['decoded_pixel_sha256']),
               'physical_scene_session_status': 'unknown; no inference from source-image names/categories/EXIF',
               'annotation_metadata_scene_like_fields': {k: v['scene_like_columns'] for k, v in metadata.items()},
               'annotation_label_comparisons': {k: v['official_label_comparison'] for k, v in metadata.items()},
               'annotation_label_mismatch_count': sum(len(v['label_mismatches']) for v in metadata.values()),
               'annotation_literal_label_difference_count': sum(len(v['literal_official_label_differences']) for v in metadata.values()),
               'annotation_mask_path_mismatch_count': sum(len(v['mask_path_mismatches']) for v in metadata.values()),
               'annotation_label_token_mask_class_mismatch_count': sum(len(v['label_token_mask_class_mismatches']) for v in metadata.values()),
               'annotation_multiclass_mapping_source_sha256': OFFICIAL_CLASSMAP_SHA256,
               'source_manifest_sha256': sha256_file(evidence_dir/'pcb-source-manifest.jsonl'),
               'inventory_sha256': sha256_file(evidence_dir/'archive-inventory.jsonl')}
    (evidence_dir/'inspection-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def verify_source(archive, split, official_manifest, data_root, evidence_dir):
    archive, split, official_manifest = Path(archive), Path(split), Path(official_manifest)
    fixed = json.loads(official_manifest.read_text())
    if fixed['commit'] != OFFICIAL_COMMIT:
        raise ValueError('Official source commit differs')
    split_record = next(r for r in fixed['files'] if r['repo_path'] == 'split_csv/1cls.csv')
    if split_record['sha256'] != OFFICIAL_SPLIT_SHA256 or sha256_file(split) != OFFICIAL_SPLIT_SHA256:
        raise ValueError('Pinned official split SHA differs')
    for record in fixed['files']:
        path = official_manifest.parent/safe_name(record['evidence_file'])
        if sha256_file(path) != record['sha256']:
            raise ValueError('Pinned official metadata byte hash differs')
    download = json.loads(archive.with_suffix('.tar.json').read_text())
    if not download['complete'] or archive.stat().st_size != ARCHIVE_SIZE or sha256_file(archive) != download['downloaded_archive_sha256']:
        raise ValueError('Full archive not complete/SHA-verified')
    class_record = next(r for r in fixed['files'] if r['repo_path'] == 'utils/id2class.py')
    if class_record['sha256'] != OFFICIAL_CLASSMAP_SHA256:
        raise ValueError('Pinned official class names source SHA differs')
    rows, split_summary = split_rows(split)
    result = inspect_tar(archive, rows, data_root, evidence_dir)
    if sha256_file(Path(__file__)) != LOADED_SOURCE_SHA256:
        raise ValueError('Inspector source changed while the process was running')
    result.update(official_commit=OFFICIAL_COMMIT, official_split_sha256=sha256_file(split),
                  full_archive_sha256=download['downloaded_archive_sha256'],
                  archive_checksum_authority=download['checksum_authority'],
                  source_inspector_sha256=LOADED_SOURCE_SHA256,
                  completed_utc=datetime.now(timezone.utc).isoformat(), split_summary=split_summary)
    (Path(evidence_dir)/'inspection-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def wait_verify(args):
    started = time.monotonic()
    while not (args.archive.exists() and args.archive.with_suffix('.tar.json').exists()):
        process = Path('/proc')/str(args.download_pid)/'cmdline'
        if not process.exists() or b'download-start-source.py' not in process.read_bytes():
            raise RuntimeError('Observed download exited before verified archive completion; do not restart a still-live handle')
        if time.monotonic()-started > args.timeout_seconds:
            raise TimeoutError('Bounded observer wait expired; download was not stopped or restarted')
        print(json.dumps({'event': 'waiting_for_existing_download', 'pid': args.download_pid,
                          'elapsed_seconds': time.monotonic()-started,
                          'UTC': datetime.now(timezone.utc).isoformat()}), flush=True)
        time.sleep(30)
    return verify_source(args.archive, args.split, args.official_manifest, args.data_root, args.evidence_dir)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch')
    fetch.add_argument('--output', type=Path, required=True)
    fetch.add_argument('--workers', type=int, default=16)
    for name in ('verify', 'wait-verify'):
        verify = sub.add_parser(name)
        verify.add_argument('--archive', type=Path, required=True)
        verify.add_argument('--split', type=Path, required=True)
        verify.add_argument('--official-manifest', type=Path, required=True)
        verify.add_argument('--data-root', type=Path, required=True)
        verify.add_argument('--evidence-dir', type=Path, required=True)
        if name == 'wait-verify':
            verify.add_argument('--download-pid', type=int, required=True)
            verify.add_argument('--timeout-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.command == 'fetch':
        fetch_archive(args.output, args.workers)
    elif args.command == 'verify':
        verify_source(args.archive, args.split, args.official_manifest, args.data_root, args.evidence_dir)
    elif args.command == 'wait-verify':
        wait_verify(args)


if __name__ == '__main__':
    main()
