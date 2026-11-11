#!/usr/bin/env python3
"""Durable industrial evaluation stages. Explicit --write only; no implicit TEST.

fit-bank -> freeze-thresholds(all normal VALID branches) -> test(one/all branch)
-> aggregate. Formal TEST gates pin all nine completed training runs and all
four class-specific bank receipts. Engineering limits remain visibly incomplete.
Permanent outputs retain native crop-grid F32 scores, never 600GB pixel shards.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import numpy as np
import scipy
import torch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('industrial_durable_evaluation',HERE/'industrial_evaluate.py')
v=importlib.util.module_from_spec(s);sys.modules[s.name]=v;s.loader.exec_module(v)
BRANCHES=[('sharp',None),('blur',None)]+[(m,s) for m in v.MODES for s in v.SEEDS]
MAX_EXTRA_DEFAULT=32*1024**3


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('xb') as f:f.write(v.canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def atomic_npy(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def read_json(path):return v.strict_json(Path(path).read_text())
def resolve(base,value):return str((Path(base)/value).resolve())
def branch_key(branch,seed):
    if (branch,seed) not in BRANCHES:raise ValueError('Expected sharp/blur or mode+seed0/1/2')
    return branch if seed is None else branch+'-seed'+str(seed)
def group_key(category,level,branch,seed):
    if not re.fullmatch('[A-Za-z0-9_-]+',category):raise ValueError('Unsafe category')
    return category+'__a'+format(level,'.12g').replace('.','p')+'__'+branch_key(branch,seed)
def source_id(row):return row['source_image_id']


def code_identity():
    return {name:v.sha(HERE/name) for name in ('industrial_execution.py','industrial_evaluate.py','industrial_metrics.py','industrial_geometry.py','float_dataset.py')}


def factory_kwargs(raw,base):
    kwargs=dict(raw)
    for key in ('weights_path','source_root','source_manifest'):
        if key in kwargs and kwargs[key] is not None:kwargs[key]=resolve(base,kwargs[key])
    return kwargs


def instantiate(raw,base):
    d=v.module('industrial_execution_detector',HERE/'industrial_detector.py');kwargs=factory_kwargs(raw,base)
    if 'config' in kwargs:kwargs['config']=d.DetectorConfig(**kwargs['config'])
    return d.create_detector(**kwargs)


def validate_selection(prepared,path,expected_sha256,category):
    selection=v.strict_json(v.checked_bytes(path,expected_sha256))
    expected=v.select_bank_candidate(prepared,category,limit=selection['source_limit'],selection_seed=selection['selection_seed'])
    if selection!=expected:raise ValueError('Bank selection differs from frozen deterministic train-only candidate')
    return selection


def fit_bank(compiled_directory,source_manifest,summary_sha256,selection_path,selection_sha256,
             detector_spec_path,detector_spec_sha256,output,*,write=False):
    if not write:raise ValueError('Bank fitting requires explicit --write')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    prepared=v.prepare_evaluation_inputs(compiled_directory,source_manifest,expected_compile_summary_sha256=summary_sha256)
    raw=v.strict_json(v.checked_bytes(selection_path,selection_sha256))
    selection=validate_selection(prepared,selection_path,selection_sha256,raw['category'])
    spec=v.strict_json(v.checked_bytes(detector_spec_path,detector_spec_sha256))
    detector=instantiate(spec,Path(detector_spec_path).resolve().parent)
    ledger=[]
    def inputs():
        for chosen in selection['sources']:
            sample,row=v.load_compiled_sample(prepared,chosen['sample_id'])
            if row['split_name']!='train' or row['category']!=selection['category'] or row['source']['original_image_label']!='normal':
                raise ValueError('Bank source is not same-category sharp normal TRAIN')
            encoded,clipping=v.metrics.detection_encoding(sample['sharp_linear_rgb'])
            if clipping['clipped_rgb_components']:raise ValueError('Sharp normal targets unexpectedly require clipping')
            ledger.append(dict(chosen,srgb_sha256=hashlib.sha256(encoded.tobytes()).hexdigest()))
            yield torch.from_numpy(encoded.transpose(2,0,1).copy())
    metadata=detector.fit_features(inputs(),output,training_manifest_sha256=selection_sha256)
    if [r['srgb_sha256'] for r in ledger]!=[r['srgb_sha256'] for r in metadata['training_records']]:
        raise ValueError('Actual fit tensors differ from source ledger')
    receipt={'schema':'industrial-sharp-normal-bank-fit-v1','category':selection['category'],
        'compile_summary_sha256':summary_sha256,'selection_sha256':selection_sha256,
        'metadata_sha256':v.sha(output/'metadata.json'),'bank_sha256':metadata['bank_sha256'],
        'detector_contract_sha256':metadata['contract_sha256'],'training_sources':ledger,
        'engineering_only':metadata['contract']['engineering_only'],'code_sha256':code_identity(),
        'industrial_results':False}
    atomic_json(output/'fit-receipt.json',receipt)
    return receipt


def read_plan(path,expected_sha256):
    path=Path(path).resolve();raw=v.strict_json(v.checked_bytes(path,expected_sha256));base=path.parent
    if set(raw)!={'schema','compiled','detector','banks','runs','engineering_only'} or raw['schema']!='industrial-evaluation-plan-v1' or type(raw['engineering_only']) is not bool:
        raise ValueError('Exact frozen industrial-evaluation-plan-v1 required')
    comp=raw['compiled']
    if set(comp)!={'directory','source_manifest','data_root','summary_sha256'}:raise ValueError('Exact compiled binding required')
    prepared=v.prepare_evaluation_inputs(resolve(base,comp['directory']),resolve(base,comp['source_manifest']),expected_compile_summary_sha256=comp['summary_sha256'])
    categories=sorted({r['category'] for r in prepared['rows']})
    if prepared['protocol']['source_scope']!='visa-pcb1-4-official' and not raw['engineering_only']:
        raise ValueError('Artificial source scope can only be engineering')
    if set(raw['banks'])!=set(categories):raise ValueError('Exactly one bank per category required')
    config=raw['detector'].get('config',{})
    if not raw['engineering_only'] and (config.get('engineering_test',False) or config.get('backbone','wide_resnet50_2')!='wide_resnet50_2'):
        raise ValueError('Formal plan cannot use engineering detector')
    banks={}
    for category,item in raw['banks'].items():
        if set(item)!={'directory','selection_path','selection_sha256','fit_receipt_sha256','metadata_sha256'}:
            raise ValueError('Exact bank selection/fit receipt/metadata bindings required')
        selection=validate_selection(prepared,resolve(base,item['selection_path']),item['selection_sha256'],category)
        directory=Path(resolve(base,item['directory']))
        receipt=v.strict_json(v.checked_bytes(directory/'fit-receipt.json',item['fit_receipt_sha256']))
        metadata=v.strict_json(v.checked_bytes(directory/'metadata.json',item['metadata_sha256']))
        if receipt['schema']!='industrial-sharp-normal-bank-fit-v1' or receipt['category']!=category or receipt['compile_summary_sha256']!=comp['summary_sha256'] or receipt['selection_sha256']!=item['selection_sha256'] or receipt['metadata_sha256']!=item['metadata_sha256'] or receipt['bank_sha256']!=metadata['bank_sha256'] or receipt['detector_contract_sha256']!=metadata['contract_sha256'] or metadata['training_manifest_sha256']!=item['selection_sha256']:
            raise ValueError('Bank receipt provenance mismatch')
        if [{k:r[k] for k in selection['sources'][0]} for r in receipt['training_sources']]!=selection['sources'] or [r['srgb_sha256'] for r in receipt['training_sources']]!=[r['srgb_sha256'] for r in metadata['training_records']]:
            raise ValueError('Bank actual source/tensor ledger differs')
        if v.sha(directory/'bank.npy')!=receipt['bank_sha256'] or receipt['industrial_results'] is not False:
            raise ValueError('Bank bytes/status differ')
        if not raw['engineering_only'] and receipt['engineering_only']:raise ValueError('Engineering bank in formal plan')
        banks[category]=dict(item,directory=str(directory),
            expected_bank_identity={'training_manifest_sha256':item['selection_sha256'],'bank_sha256':receipt['bank_sha256'],'metadata_sha256':item['metadata_sha256']},
            expected_detector_contract_sha256=metadata['contract_sha256'])
    if not isinstance(raw['runs'],list) or len(raw['runs'])!=9:raise ValueError('All nine completed training receipts required before evaluation')
    runs={};common=None
    for item in raw['runs']:
        if set(item)!={'mode','seed','checkpoint_path','checkpoint_sha256','run_report_path','run_report_sha256','trainer_path'}:
            raise ValueError('Exact nine-run file bindings required')
        key=(item['mode'],item['seed'])
        if key not in BRANCHES or key[0] not in v.MODES or key in runs:raise ValueError('Wrong/duplicate mode/seed')
        job=dict(item)
        for field in ('checkpoint_path','run_report_path','trainer_path'):job[field]=resolve(base,job[field])
        report=v.strict_json(v.checked_bytes(job['run_report_path'],job['run_report_sha256']));c=report['contract']
        if c['input_mode']!=key[0] or c['seed']!=key[1] or c['allocation_sha256']!=prepared['summary']['allocation_sha256'] or report.get('complete_budget') is not True or report['completed_update']!=c['total_updates'] or report['requested_total_updates']!=c['total_updates']:
            raise ValueError('Nine-run source/mode/seed/full budget mismatch')
        if v.sha(job['checkpoint_path'])!=job['checkpoint_sha256'] or report['checkpoint_files'].get(Path(job['checkpoint_path']).name)!=job['checkpoint_sha256']:
            raise ValueError('Final training file hash differs')
        normalized={k:value for k,value in c.items() if k not in ('input_mode','seed')}
        if common is None:common=normalized
        elif common!=normalized:raise ValueError('Nine runs do not share the same frozen budget/data/model/optimizer contract')
        for name,sha in c['code_sha256'].items():
            if v.sha(Path(job['trainer_path']).parent/name)!=sha:raise ValueError('Run trainer dependency differs')
        job['expected_restorer_identity']=v.restoration_identity(report,expected_checkpoint_sha256=job['checkpoint_sha256'],
            expected_run_report_sha256=job['run_report_sha256'],expected_allocation_sha256=prepared['summary']['allocation_sha256'],
            input_mode=key[0],seed=key[1],device=raw['detector'].get('device','cpu'))
        runs[key]=job
    if set(runs)!={(m,s) for m in v.MODES for s in v.SEEDS}:raise ValueError('Incomplete nine-run grid')
    return {'raw':raw,'base':str(base),'plan_sha256':expected_sha256,'prepared':prepared,'categories':categories,
            'banks':banks,'runs':runs,'factory_kwargs':factory_kwargs(raw['detector'],base),
            'data_root':resolve(base,comp['data_root']),'engineering_only':raw['engineering_only']}


def group_rows(plan,category,level,split,limit):
    rows=sorted([r for r in plan['prepared']['rows'] if r['category']==category and r['angle_deg']==level and r['split_name']==split],key=source_id)
    if not rows:raise ValueError('Required class/level/split is empty')
    if split=='valid' and any(r['source']['original_image_label']!='normal' for r in rows):raise ValueError('Calibration must be VALID normal only')
    return rows if limit is None else rows[:limit]


def resources(plan,split,limit,max_extra_bytes,output):
    if type(max_extra_bytes) is not int or max_extra_bytes<1:raise ValueError('Explicit positive memory cap required')
    all_rows=[];max_pixels=0
    for category in plan['categories']:
        for level in plan['prepared']['protocol']['angle_levels_deg']:
            rows=group_rows(plan,category,level,split,limit);all_rows+=rows
            pixels=0
            for row in rows:
                x0,y0,x1,y1=row['source_mapping']['original_roi_index_xyxy'];pixels+=(x1-x0)*(y1-y0)
            max_pixels=max(max_pixels,pixels)
    if split=='test' and max_pixels*128>max_extra_bytes:raise MemoryError('Exact per-group native ranking exceeds explicit extra-memory estimate; choose reviewed resources, never silently approximate')
    permanent=sum(np.prod(r['source_mapping']['crop_size_wh'])*4+16384 for r in all_rows)*len(BRANCHES)
    ancestor=Path(output).resolve()
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free < permanent+1024**3:raise OSError('Insufficient free disk for permanent crop scores/receipts plus 1GiB reserve')
    return {'max_original_roi_pixels_per_group':max_pixels,'native_peak_extra_estimate_bytes':int(max_pixels*128),
            'extra_memory_cap_bytes':max_extra_bytes,'permanent_crop_score_receipt_estimate_bytes_all_branches':int(permanent),
            'temporary_original_score_shards':False,'estimate_not_profile':True}


def execution_context(plan,stage,threshold_sha,limit):
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('Positive explicit engineering limit per group required')
    return {'schema':'industrial-durable-execution-v1','stage':stage,'plan_sha256':plan['plan_sha256'],
            'threshold_sha256':threshold_sha,'engineering_limit_per_group':limit,
            'engineering_only':bool(plan['engineering_only'] or limit is not None),'code_sha256':code_identity(),
            'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
            'device':plan['factory_kwargs'].get('device','cpu'),'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(torch.device(plan['factory_kwargs']['device'])) if str(plan['factory_kwargs'].get('device','cpu')).startswith('cuda') else None,
            'cpu_threads':torch.get_num_threads(),'cpu_interop_threads':torch.get_num_interop_threads()}


def open_execution(output,context,resume):
    output=Path(output)
    if output.exists():
        if not resume or not (output/'execution.json').is_file() or read_json(output/'execution.json')!=context:
            raise ValueError('Existing execution may resume only with exactly matching plan/threshold/code/environment/limit identity')
    else:
        output.mkdir(parents=True,exist_ok=False);atomic_json(output/'execution.json',context)
    return output


def native_rankings(scores,truth,max_extra_bytes):
    """Production per-group exact NumPy ranking; conservatively budget 128B/pixel.

    No Python per-pixel heap or approximation. Call separately per class/level/
    branch/seed; this function never mixes different bank scales.
    """
    if not isinstance(scores,np.ndarray) or scores.dtype not in (np.dtype('float32'),np.dtype('float64')) or not isinstance(truth,np.ndarray) or truth.dtype!=bool or scores.ndim!=1 or truth.shape!=scores.shape or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Flat finite F32/F64 scores and matching bool truth required')
    if len(scores)*128>max_extra_bytes:raise MemoryError('Native exact ranking exceeds memory cap')
    positives=int(truth.sum());negatives=len(truth)-positives
    if not positives or not negatives:return {'auroc':None,'average_precision':None,'positive_count':positives,'negative_count':negatives,'undefined_reason':'Ranking requires both original classes'}
    order=np.argsort(-scores,kind='stable');ordered=scores[order];cum=np.cumsum(truth[order],dtype=np.int64)
    ends=np.r_[np.flatnonzero(ordered[1:]!=ordered[:-1]),len(scores)-1]
    tp=cum[ends].astype(np.float64);fp=(ends+1)-tp
    del order,ordered,cum
    dtp=np.diff(np.r_[0.,tp]);dfp=np.diff(np.r_[0.,fp]);oldtp=np.r_[0.,tp[:-1]]
    auc=float(np.sum(dfp/negatives*(tp+oldtp)/(2*positives)))
    ap=float(np.sum(dtp/positives*tp/(tp+fp)))
    return {'auroc':auc,'average_precision':ap,'positive_count':positives,'negative_count':negatives,
            'tie_policy':'Exact equal scores enter one global threshold group; noninterpolated AP','implementation':'NumPy native sort/cumulative, explicit 128B/pixel estimate'}


def expected_image_identity(plan,row,context,branch,seed):
    """Authoritative join from pinned plan metadata + compiled source row.

    The same expected identity serves fresh execution, resume, calibration and
    aggregation. Nothing in a completion receipt supplies its own expectation.
    """
    branch_key(branch,seed)
    if row['split_name']!=context['stage']:raise ValueError('Source row split differs from execution stage')
    bank=plan['banks'][row['category']]
    restorer=plan['runs'][branch,seed]['expected_restorer_identity'] if branch in v.MODES else None
    return {'execution':context,'sample_id':row['sample_id'],'npz_sha256':row['npz_sha256'],
        'group':group_key(row['category'],row['angle_deg'],branch,seed),
        'source_image_id':row['source_image_id'],'category':row['category'],'angle_deg':row['angle_deg'],
        'split_name':row['split_name'],'original_image_label':row['source']['original_image_label'],
        'branch':branch,'seed':seed,'bank_identity':bank['expected_bank_identity'],
        'detector_contract_sha256':bank['expected_detector_contract_sha256'],'restorer':restorer,
        'score_crop_hw':row['source_mapping']['crop_size_wh'][::-1]}


def record_provenance(record,expected):
    keys=('sample_id','npz_sha256','group','source_image_id','category','angle_deg','split_name',
          'original_image_label','branch','seed','bank_identity','detector_contract_sha256','restorer')
    if not isinstance(record,dict) or any(key not in record or record[key]!=expected[key] for key in keys):
        raise ValueError('Per-image record provenance differs from pinned plan/source/model identities')


def completed_image(directory,identity):
    receipt_path=directory/'complete.json'
    if not receipt_path.exists():return None
    receipt=read_json(receipt_path)
    if receipt.get('schema')!='industrial-evaluated-image-v1' or receipt.get('identity')!=identity:
        raise ValueError('Completed image identity mismatch; no cross-run cache reuse')
    record=v.strict_json(v.checked_bytes(directory/'record.json',receipt['record_sha256']))
    record_provenance(record,identity)
    if receipt['score_crop_hw']!=identity['score_crop_hw']:raise ValueError('Completed crop geometry differs from compiled source')
    if v.sha(directory/'score-crop.npy')!=receipt['score_crop_sha256']:raise ValueError('Completed crop score corrupted')
    score=np.load(directory/'score-crop.npy',allow_pickle=False,mmap_mode='r')
    if score.dtype!=np.float32 or list(score.shape)!=receipt['score_crop_hw'] or not np.isfinite(score).all():raise ValueError('Malformed completed score crop')
    return record,receipt


def execute_image(plan,row,context,group_dir,branch,seed,detector,restorer,threshold):
    directory=group_dir/row['sample_id'];directory.mkdir(parents=True,exist_ok=True)
    identity=expected_image_identity(plan,row,context,branch,seed)
    if detector.fitted_identity!=identity['bank_identity'] or detector.contract_sha256!=identity['detector_contract_sha256'] or (restorer.identity if restorer else None)!=identity['restorer']:
        raise ValueError('Loaded model/bank differs from frozen expected image identity')
    cached=completed_image(directory,identity)
    if cached is not None:return cached[0]
    sample,checked=v.load_compiled_sample(plan['prepared'],row['sample_id'])
    annotation=v.read_original_annotation(plan['prepared'],row['source_image_id'],plan['data_root'],row['source_mapping'])
    record,_,_,crop=v.evaluate_one(sample,checked,annotation,branch=branch,detector=detector,restorer=restorer,
        image_threshold=threshold,return_crop_score=True)
    record['group']=identity['group'];record['restorer']=identity['restorer']
    record_provenance(record,identity)
    if crop.dtype!=np.float32 or list(crop.shape)!=row['source_mapping']['crop_size_wh'][::-1]:raise ValueError('Permanent score is not the native crop grid')
    atomic_npy(directory/'score-crop.npy',crop);atomic_json(directory/'record.json',record)
    receipt={'schema':'industrial-evaluated-image-v1','identity':identity,'record_sha256':v.sha(directory/'record.json'),
             'score_crop_sha256':v.sha(directory/'score-crop.npy'),'score_crop_hw':list(crop.shape)}
    atomic_json(directory/'complete.json',receipt)
    return record


def group_aggregate(plan,rows,group_dir,records,threshold,max_extra_bytes):
    if len(records)!=len(rows) or [r['sample_id'] for r in records]!=[r['sample_id'] for r in rows]:raise ValueError('Strict per-group source counts/order required')
    total=sum(r['pixels']['observed_pixels'] for r in records)
    if total*128>max_extra_bytes:raise MemoryError('Per-group pixel resource cap exceeded')
    scores=np.empty(total,np.float64);truth=np.empty(total,bool);position=0
    for row,record in zip(rows,records):
        crop=np.load(group_dir/row['sample_id']/'score-crop.npy',allow_pickle=False)
        score=v.geometry.remap_score_to_original_roi(crop,row['source_mapping'])
        annotation=v.read_original_annotation(plan['prepared'],row['source_image_id'],plan['data_root'],row['source_mapping'])
        mask=annotation['roi_mask_original_grid'];n=score.size
        if n!=record['pixels']['observed_pixels'] or int((mask!=0).sum())!=record['pixels']['observed_positive_pixels']:raise ValueError('Reconstructed original ROI counts differ')
        scores[position:position+n]=score.ravel();truth[position:position+n]=(mask!=0).ravel();position+=n
    pixels=native_rankings(scores,truth,max_extra_bytes);del scores,truth
    image=v.metrics.ranking_metrics(np.array([r['image_score'] for r in records],np.float64),np.array([r['original_image_label']=='anomaly' for r in records],bool))
    sums={key:sum(r['image_threshold_statistics'][key] for r in records) for key in ('tp','fp','fn','tn','coverage_aware_tp','coverage_aware_fn')}
    p=sums['tp']+sums['fn'];normal=sums['fp']+sums['tn']
    sums['ordinary_recall']=sums['tp']/p if p else None
    sums['ordinary_fnr']=sums['fn']/p if p else None
    sums['normal_fpr']=sums['fp']/normal if normal else None
    sums['coverage_aware_recall']=sums['coverage_aware_tp']/p if p else None
    sums['coverage_aware_fnr']=sums['coverage_aware_fn']/p if p else None
    def quality(chosen):
        elements=sum(r['restoration']['valid_rgb_elements'] for r in chosen);sse=sum(r['restoration']['sum_squared_error'] for r in chosen)
        undefined=sum(r['restoration']['ssim'] is None for r in chosen)
        return {'images':len(chosen),'sum_squared_error':sse,'valid_rgb_elements':elements,
                'pooled_psnr_db':float(-10*np.log10(sse/elements)) if sse>0 and elements else None,
                'perfect_pooled':bool(elements and sse==0),'ssim_mean':float(np.mean([r['restoration']['ssim'] for r in chosen])) if chosen and not undefined else None,
                'ssim_undefined_images':undefined,'perfect_psnr_images':sum(r['restoration']['perfect'] for r in chosen)}
    coverage={'original_defect_pixels':sum(r['annotation']['original_defect_pixels'] for r in records),
              'retained_defect_pixels':sum(r['annotation']['retained_defect_pixels'] for r in records),
              'invisible_positive_sources':sum(r['annotation']['all_defects_outside_roi'] for r in records),
              'original_positive_sources':p,'outside_roi_policy':'unobserved; no fabricated pixel score'}
    coverage['retained_area_fraction']=coverage['retained_defect_pixels']/coverage['original_defect_pixels'] if coverage['original_defect_pixels'] else None
    coverage['invisible_positive_fraction']=coverage['invisible_positive_sources']/p if p else None
    clipping={key:sum(r['clipping'][key] for r in records) for key in ('clipped_rgb_components','total_rgb_components','clipped_pixels','total_pixels')}
    clipping['rgb_component_fraction']=clipping['clipped_rgb_components']/clipping['total_rgb_components']
    clipping['pixel_fraction']=clipping['clipped_pixels']/clipping['total_pixels']
    return {'image_ranking':image,'pixel_ranking_original_roi':pixels,'image_threshold':threshold,'image_threshold_statistics':sums,
            'restoration':{'all':quality(records),'normal':quality([r for r in records if r['original_image_label']=='normal']),
                           'anomaly':quality([r for r in records if r['original_image_label']=='anomaly'])},
            'coverage':coverage,'clipping':clipping}


def expected_groups(plan):
    return {group_key(c,l,b,s) for c in plan['categories'] for l in plan['prepared']['protocol']['angle_levels_deg'] for b,s in BRANCHES}


def load_thresholds(plan,path,expected_sha256,engineering):
    value=v.strict_json(v.checked_bytes(path,expected_sha256))
    if value.get('schema')!='industrial-normal-valid-thresholds-v1' or value['context']['plan_sha256']!=plan['plan_sha256'] or value['context']['code_sha256']!=code_identity() or value.get('calibration_stage')!='valid normal only' or set(value['groups'])!=expected_groups(plan) or set(value['group_receipts_sha256'])!=expected_groups(plan):raise ValueError('Complete matching all-branch normal-VALID thresholds required')
    if value['context']['engineering_only'] and not engineering:raise ValueError('Engineering thresholds cannot authorize formal TEST')
    for c in plan['categories']:
        for l in plan['prepared']['protocol']['angle_levels_deg']:
            rows=group_rows(plan,c,l,'valid',value['context']['engineering_limit_per_group'])
            for b,s in BRANCHES:
                key=group_key(c,l,b,s);threshold=value['groups'][key]
                calibration_root=Path(path).resolve().parent
                if read_json(calibration_root/'execution.json')!=value['context']:raise ValueError('Calibration execution identity differs')
                receipt=v.strict_json(v.checked_bytes(calibration_root/key/'group-receipt.json',value['group_receipts_sha256'][key]))
                if receipt['context']!=value['context'] or receipt['threshold']!=threshold or receipt['source_ids']!=[r['source_image_id'] for r in rows]:raise ValueError('Calibration group receipt differs')
                original_rows=[]
                for row in rows:
                    directory=calibration_root/key/row['sample_id']
                    image=v.strict_json(v.checked_bytes(directory/'complete.json',receipt['completed_image_receipt_sha256'][row['sample_id']]))
                    identity=image['identity']
                    if identity!=expected_image_identity(plan,row,value['context'],b,s):raise ValueError('Calibration completion differs from expected pinned identities')
                    if identity['execution']!=value['context'] or identity['sample_id']!=row['sample_id'] or identity['npz_sha256']!=row['npz_sha256'] or identity['group']!=key:raise ValueError('Calibration image identity differs')
                    bank=plan['banks'][c]
                    if identity['bank_identity']['metadata_sha256']!=bank['metadata_sha256'] or identity['bank_identity']['training_manifest_sha256']!=bank['selection_sha256']:raise ValueError('Calibration bank identity differs')
                    record=v.strict_json(v.checked_bytes(directory/'record.json',image['record_sha256']))
                    record_provenance(record,identity)
                    if record['source_image_id']!=row['source_image_id'] or record['category']!=c or record['angle_deg']!=l or record['branch']!=b or record['seed']!=s or record['split_name']!='valid' or record['original_image_label']!='normal':raise ValueError('Mixed calibration group')
                    original_rows.append(record)
                if v.metrics.freeze_normal_threshold(original_rows)!=threshold:raise ValueError('Threshold does not reproduce original normal-VALID records')
                if threshold['valid_normal_source_ids']!=[r['source_image_id'] for r in rows] or threshold['quantile']!=.95 or threshold['method']!='empirical numpy.quantile higher; no conformal coverage guarantee' or not np.isfinite(threshold['image_threshold']):raise ValueError('Threshold calibration group/source provenance differs')
    return value


def run_stage(plan_path,plan_sha256,output,*,stage,write=False,resume=False,thresholds_path=None,
              thresholds_sha256=None,branch=None,seed=None,engineering_limit=None,max_extra_bytes=MAX_EXTRA_DEFAULT):
    if not write:raise ValueError('Inference requires explicit --write')
    if stage not in ('valid','test'):raise ValueError('Only valid calibration or TEST stage')
    plan=read_plan(plan_path,plan_sha256)
    engineering=plan['engineering_only'] or engineering_limit is not None
    thresholds=None
    if stage=='test':
        if not thresholds_path or not thresholds_sha256:raise ValueError('TEST forbidden before complete frozen normal-VALID thresholds')
        thresholds=load_thresholds(plan,thresholds_path,thresholds_sha256,engineering)
    elif branch is not None:raise ValueError('Normal threshold freeze runs all eleven branches before global completion')
    chosen=BRANCHES if branch is None else [(branch,seed)]
    for b,s in chosen:branch_key(b,s)
    estimate=resources(plan,stage,engineering_limit,max_extra_bytes,output)
    context=execution_context(plan,stage,thresholds_sha256 if stage=='test' else None,engineering_limit)
    root=open_execution(output,context,resume)
    atomic_json(root/'resource-estimate.json',estimate)
    threshold_groups={};threshold_group_hashes={}
    for b,s in chosen:
        restorer=None
        if b in v.MODES:
            job=plan['runs'][b,s]
            restorer=v.load_best_restorer(job['checkpoint_path'],job['run_report_path'],
                expected_checkpoint_sha256=job['checkpoint_sha256'],expected_run_report_sha256=job['run_report_sha256'],
                expected_allocation_sha256=plan['prepared']['summary']['allocation_sha256'],input_mode=b,seed=s,
                device=plan['factory_kwargs'].get('device','cpu'),trainer_path=job['trainer_path'])
        branch_receipts=[]
        for category in plan['categories']:
            item=plan['banks'][category]
            detector=v.load_shared_bank(item['directory'],expected_metadata_sha256=item['metadata_sha256'],
                expected_training_manifest_sha256=item['selection_sha256'],factory_kwargs=plan['factory_kwargs'])
            for level in plan['prepared']['protocol']['angle_levels_deg']:
                key=group_key(category,level,b,s);group_dir=root/key;group_dir.mkdir(exist_ok=True)
                rows=group_rows(plan,category,level,stage,engineering_limit)
                threshold=thresholds['groups'][key]['image_threshold'] if thresholds else None
                records=[execute_image(plan,row,context,group_dir,b,s,detector,restorer,threshold) for row in rows]
                hashes={r['sample_id']:v.sha(group_dir/r['sample_id']/'complete.json') for r in rows}
                receipt={'schema':'industrial-evaluation-group-v1','context':context,'category':category,'angle_deg':level,
                         'branch':b,'seed':s,'sample_ids':[r['sample_id'] for r in rows],
                         'source_ids':[r['source_image_id'] for r in rows],'completed_image_receipt_sha256':hashes,'count':len(rows)}
                if stage=='valid':
                    frozen=v.metrics.freeze_normal_threshold(records);receipt['threshold']=frozen;threshold_groups[key]=frozen
                else:receipt['metrics']=group_aggregate(plan,rows,group_dir,records,threshold,max_extra_bytes)
                atomic_json(group_dir/'group-receipt.json',receipt);branch_receipts.append((key,v.sha(group_dir/'group-receipt.json')))
                if stage=='valid':threshold_group_hashes[key]=v.sha(group_dir/'group-receipt.json')
            del detector;gc.collect()
            if plan['factory_kwargs'].get('device','cpu').startswith('cuda'):torch.cuda.empty_cache()
        atomic_json(root/(branch_key(b,s)+'-complete.json'),{'schema':'industrial-evaluation-branch-v1','context':context,'branch':b,'seed':s,'group_receipts':dict(branch_receipts)})
        del restorer;gc.collect()
        if plan['factory_kwargs'].get('device','cpu').startswith('cuda'):torch.cuda.empty_cache()
    if stage=='valid':
        if set(threshold_groups)!=expected_groups(plan):raise ValueError('Cannot publish partial threshold set')
        result={'schema':'industrial-normal-valid-thresholds-v1','context':context,'calibration_stage':'valid normal only','groups':threshold_groups,'group_receipts_sha256':threshold_group_hashes}
        atomic_json(root/'thresholds.json',result);return result
    return {'stage':'test','branches_completed':[branch_key(b,s) for b,s in chosen],
            'engineering_only':context['engineering_only'],'formal_experiment_complete':False,
            'next':'aggregate checks all eleven branches and strict source counts'}


MACRO_PATHS = {
    'image_auroc':('image_ranking','auroc'),
    'image_average_precision':('image_ranking','average_precision'),
    'pixel_auroc':('pixel_ranking_original_roi','auroc'),
    'pixel_average_precision':('pixel_ranking_original_roi','average_precision'),
    'restoration_psnr_db':('restoration','all','pooled_psnr_db'),
    'restoration_ssim':('restoration','all','ssim_mean'),
    'ordinary_threshold_recall':('image_threshold_statistics','ordinary_recall'),
    'normal_threshold_fpr':('image_threshold_statistics','normal_fpr'),
    'coverage_aware_threshold_recall':('image_threshold_statistics','coverage_aware_recall'),
    'coverage_retained_area_fraction':('coverage','retained_area_fraction'),
    'coverage_invisible_positive_fraction':('coverage','invisible_positive_fraction'),
    'clipped_rgb_component_fraction':('clipping','rgb_component_fraction'),
    'clipped_pixel_fraction':('clipping','pixel_fraction'),
}


def class_macro(selected,categories,branch,seed,level):
    """Equal class weight, not a pooled cross-bank score ranking or pixel mean.

    Undefined/perfect-infinite values are never silently dropped from a mean.
    Per-class PSNR here is derived from its pooled masked SSE/scalar count.
    """
    if len(selected)!=len(categories) or not categories:raise ValueError('Complete ordered class groups required')
    record={'branch':branch,'seed':seed,'angle_deg':level,'categories':categories,
            'averaging':'equal class arithmetic mean within level/branch/seed; no cross-bank score pooling'}
    for name,path in MACRO_PATHS.items():
        values=[]
        for group in selected:
            value=group['metrics']
            for key in path:value=value[key]
            if value is not None and not np.isfinite(value):raise ValueError('Finite/null class summaries required')
            values.append(value)
        record[name]=None if any(value is None for value in values) else float(np.mean(values))
        record[name+'_undefined_categories']=[c for c,value in zip(categories,values) if value is None]
    record['restoration_psnr_perfect_categories']=[c for c,g in zip(categories,selected) if g['metrics']['restoration']['all']['perfect_pooled']]
    return record


def aggregate(plan_path,plan_sha256,execution_root,thresholds_path,thresholds_sha256,*,write=False):
    if not write:raise ValueError('Aggregate receipt requires explicit --write')
    plan=read_plan(plan_path,plan_sha256);root=Path(execution_root);context=read_json(root/'execution.json')
    if context!=execution_context(plan,'test',thresholds_sha256,context['engineering_limit_per_group']):raise ValueError('Execution identity differs')
    load_thresholds(plan,thresholds_path,thresholds_sha256,context['engineering_only'])
    groups={};branch_hashes={}
    for b,s in BRANCHES:
        path=root/(branch_key(b,s)+'-complete.json');receipt=read_json(path)
        if receipt.get('schema')!='industrial-evaluation-branch-v1' or receipt['context']!=context or receipt['branch']!=b or receipt['seed']!=s:raise ValueError('Branch completion differs')
        expected={group_key(c,l,b,s):(c,l) for c in plan['categories'] for l in plan['prepared']['protocol']['angle_levels_deg']}
        if set(receipt['group_receipts'])!=set(expected):raise ValueError('Missing/extra per-class/level group')
        for key,expected_sha in receipt['group_receipts'].items():
            group=v.strict_json(v.checked_bytes(root/key/'group-receipt.json',expected_sha))
            category,level=expected[key]
            if group.get('schema')!='industrial-evaluation-group-v1' or group['category']!=category or group['angle_deg']!=level:raise ValueError('Group key/source category/level differs')
            rows=group_rows(plan,category,level,'test',context['engineering_limit_per_group'])
            if group['context']!=context or group['branch']!=b or group['seed']!=s or group['sample_ids']!=[r['sample_id'] for r in rows] or group['source_ids']!=[r['source_image_id'] for r in rows] or group['count']!=len(rows):raise ValueError('Strict source/count group completion mismatch')
            for row in rows:
                image_dir=root/key/row['sample_id'];image_receipt=read_json(image_dir/'complete.json')
                if v.sha(image_dir/'complete.json')!=group['completed_image_receipt_sha256'][row['sample_id']] or image_receipt['identity']['execution']!=context:raise ValueError('Completed image receipt identity differs')
                authoritative=expected_image_identity(plan,row,context,b,s)
                if completed_image(image_dir,authoritative) is None:raise ValueError('Image completion missing')
            groups[key]=group
        branch_hashes[branch_key(b,s)]=v.sha(path)
    if set(groups)!=expected_groups(plan):raise ValueError('Incomplete eleven-branch experiment')
    macro=[]
    for b,s in BRANCHES:
        for level in plan['prepared']['protocol']['angle_levels_deg']:
            selected=[groups[group_key(c,level,b,s)] for c in plan['categories']]
            record=class_macro(selected,plan['categories'],b,s,level)
            macro.append(record)
    seed_summary=[]
    for mode in v.MODES:
        for level in plan['prepared']['protocol']['angle_levels_deg']:
            selected=[m for m in macro if m['branch']==mode and m['angle_deg']==level]
            item={'mode':mode,'angle_deg':level,'seeds':list(v.SEEDS),'unit':'three paired training runs; never pooled as independent images'}
            for metric in MACRO_PATHS:
                values=[m[metric] for m in selected]
                item[metric]={'mean':None if any(x is None for x in values) else float(np.mean(values)),
                              'sample_std':None if any(x is None for x in values) else float(np.std(values,ddof=1))}
            seed_summary.append(item)
    result={'schema':'industrial-evaluation-summary-v1','context':context,'branch_receipts_sha256':branch_hashes,
            'group_metrics':{key:g['metrics'] for key,g in groups.items()},'class_macro_by_level_branch_seed':macro,
            'three_seed_descriptive_summary':seed_summary,'source_cluster_unit':'source_image_id shared across levels/branches/seeds',
            'paired_bootstrap_implemented':False,'official_test_source_coverage_complete':not context['engineering_only'],
            'scientific_accept':False,'image_rows_not_pooled_across_seeds':True}
    atomic_json(root/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit-bank')
    for key in ('compiled','source-manifest','summary-sha256','selection','selection-sha256','detector-spec','detector-spec-sha256'):fit.add_argument('--'+key,required=True)
    fit.add_argument('--write',required=True)
    for name in ('freeze-thresholds','test','aggregate'):
        p=sub.add_parser(name);p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--write',required=True)
        if name!='freeze-thresholds':p.add_argument('--thresholds',required=True);p.add_argument('--thresholds-sha256',required=True)
        if name!='aggregate':
            p.add_argument('--resume',action='store_true');p.add_argument('--engineering-limit',type=int);p.add_argument('--max-extra-gib',type=int,default=32)
        if name=='test':p.add_argument('--branch',choices=['sharp','blur',*v.MODES]);p.add_argument('--seed',type=int)
    a=parser.parse_args()
    if a.command=='fit-bank':result=fit_bank(a.compiled,a.source_manifest,a.summary_sha256,a.selection,a.selection_sha256,a.detector_spec,a.detector_spec_sha256,a.write,write=True)
    elif a.command=='aggregate':result=aggregate(a.plan,a.plan_sha256,a.write,a.thresholds,a.thresholds_sha256,write=True)
    else:result=run_stage(a.plan,a.plan_sha256,a.write,stage='valid' if a.command=='freeze-thresholds' else 'test',write=True,resume=a.resume,
        thresholds_path=getattr(a,'thresholds',None),thresholds_sha256=getattr(a,'thresholds_sha256',None),branch=getattr(a,'branch',None),seed=getattr(a,'seed',None),
        engineering_limit=a.engineering_limit,max_extra_bytes=a.max_extra_gib*1024**3)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
