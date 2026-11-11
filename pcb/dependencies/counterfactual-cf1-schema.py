#!/usr/bin/env python3
"""Frozen Rz90 motion-map counterfactual. No blur resynthesis or new training.

Math helper recomputes SO(3) forward geometry, 64 quadrature midpoints + 5 knots,
then timed quadratic descriptors on the original full canvas and common crop.
Durable runner is a separate three-seed substudy; main eleven branches stay read-only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
ADDENDUM_SHA256='f247bd244041c451e20389b2e6359568f7ee8041242eeee6e039faa753e44420'
E3_MANIFEST_SHA256='fea51917d95e34e10bed849e4bc85dd903eb40cab9e42086a5e68dcfdd5831cc'
RZ90=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=np.float64)
BRANCH='full_maps_cf_rz90'
_EXECS={}


def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def array_sha(value):
 value=np.ascontiguousarray(value)
 return hashlib.sha256((str(value.dtype)+'|'+str(value.shape)+'|').encode()+value.tobytes()).hexdigest()


def strict_json(text):
 def pairs(items):
  result={}
  for k,v in items:
   if k in result:raise ValueError('Duplicate JSON key '+k)
   result[k]=v
  return result
 def bad(v):raise ValueError('Nonfinite JSON '+v)
 return json.loads(text,object_pairs_hook=pairs,parse_constant=bad)


def executor(root=None):
 if root is None:
  root=ROOT/'dependencies/execution-e3'
  if not root.exists():root=ROOT/'audit/candidates/industrial-execution-e3-20261105'
 root=Path(root).resolve()
 if str(root) not in _EXECS:
  manifest=root/'manifest.json'
  if sha_file(manifest)!=E3_MANIFEST_SHA256:raise ValueError('Counterfactual requires pinned read-only E3 executor')
  table=strict_json(manifest.read_text())
  for row in table['files']:
   rel=Path(row['path']);p=(root/rel).resolve()
   if rel.is_absolute() or '..' in rel.parts or not p.is_relative_to(root) or p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:raise ValueError('E3 dependency bytes differ')
  spec=importlib.util.spec_from_file_location('counterfactual_readonly_e3',root/'code/revision/industrial_execution.py')
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  _EXECS[str(root)]=module
 return _EXECS[str(root)]


def read_addendum(path):
 if sha_file(path)!=ADDENDUM_SHA256:raise ValueError('Wrong frozen evaluation addendum')
 value=strict_json(Path(path).read_text());cf=value['counterfactual']
 if value['schema']!='industrial-evaluation-addendum-v1' or value['status']!='frozen-prior-to-formal-TEST-effects' or not np.array_equal(np.asarray(cf['rates_transform_camera_body_axes']),RZ90):raise ValueError('Unsupported counterfactual definition')
 return value


class CounterfactualIncomplete(ValueError):
 def __init__(self,reason,diagnostics=None):
  super().__init__(reason);self.reason=reason;self.diagnostics=diagnostics or {}


def _base(sample,engineering):
 required=('blur_linear_rgb','sharp_linear_rgb','cp_map','ep_map','network_input_9','valid_mask','metadata')
 if not isinstance(sample,dict) or any(k not in sample for k in required):raise ValueError('Explicit compiled float sample and original availability mask required')
 blur,target,network,mask=(sample[k] for k in ('blur_linear_rgb','sharp_linear_rgb','network_input_9','valid_mask'))
 if blur.dtype!=np.float32 or target.dtype!=np.float32 or network.dtype!=np.float32 or blur.shape!=target.shape or blur.ndim!=3 or blur.shape[-1]!=3 or network.shape!=blur.shape[:2]+(9,) or mask.shape!=blur.shape[:2] or mask.dtype!=bool or not np.isfinite(network).all() or not np.isfinite(target).all() or not np.array_equal(blur,network[...,:3]):raise ValueError('Original RGB/GT/mask float contract differs')
 for name,section in (('cp_map',slice(3,6)),('ep_map',slice(6,9))):
  value=sample[name]
  if value.dtype!=np.float32 or value.shape!=blur.shape or not np.isfinite(value).all() or np.any(value[...,2]!=0) or not np.array_equal(value,network[...,section]):raise ValueError('Original signed xy0 maps disagree with input')
 if np.any(blur<0) or np.any(blur>1) or np.any(target<0) or np.any(target>1):raise ValueError('Original linear RGB must be [0,1]')
 if 'fit_valid' in sample and 'full_exposure_valid' in sample and not np.array_equal(mask,sample['fit_valid']&sample['full_exposure_valid']):raise ValueError('Original availability mask is not its declared intersection')
 md=sample['metadata'];source=md['input_source'];mapping=source['mapping'];rates=np.asarray(source['trajectory']['rates_rad_s'],dtype=np.float64)
 if rates.shape!=(4,3) or not np.isfinite(rates).all() or md['map_scale_px']!=16.:raise ValueError('Four finite body-rate intervals and frozen map scale16 required')
 if not engineering and (source.get('protocol_sha256') is None or source.get('compiler_identity',{}).get('source_scope')!='visa-pcb1-4-official'):raise ValueError('Formal counterfactual requires official compiled provenance')
 return mapping,rates,source


def recompute_maps(sample,addendum,*,engineering=False,max_geometry_bytes=8*1024**3,executor_root=None):
 """Pure geometry; never calls image sampling or edits any original sample array.

 Wrong fit must be finite and valid over the WHOLE frozen ROI, even if original
 availability has holes. Failure is incomplete, never correct-map fallback.
 """
 if digest(addendum)!=digest(read_addendum_payload(addendum)):
  raise ValueError('Addendum payload differs')
 x=executor(executor_root);g=x.v.geometry._renderer()
 mapping,rates,source=_base(sample,engineering)
 if not engineering and source['protocol_sha256']!=addendum['base_protocol_sha256']:raise ValueError('Compiled protocol not frozen q16 base')
 width,height=map(int,mapping['full_size_wh']);left,top,right,bottom=map(int,mapping['crop_box_full_xyxy'])
 if width<1 or height<1 or not 0<=left<right<=width or not 0<=top<bottom<=height or [right-left,bottom-top]!=mapping['crop_size_wh'] or sample['blur_linear_rgb'].shape[:2]!=(bottom-top,right-left):raise ValueError('Original full/common crop geometry mismatch')
 estimate=69*height*width*192
 if type(max_geometry_bytes) is not int or estimate>max_geometry_bytes:raise CounterfactualIncomplete('geometry memory estimate exceeds explicit cap',{'estimated_bytes':estimate,'cap_bytes':max_geometry_bytes})
 k=g.validate_k(mapping['k_full']);times=np.linspace(0.,.02,5);wrong=rates@RZ90.T
 if not np.isclose(np.linalg.det(RZ90),1.,rtol=0,atol=1e-14) or not np.array_equal(RZ90.T@RZ90,np.eye(3)):raise ValueError('Rz90 must remain a proper rotation')
 norms=np.linalg.norm(rates,axis=1);wrong_norms=np.linalg.norm(wrong,axis=1)
 if not np.allclose(norms,wrong_norms,rtol=1e-14,atol=1e-14):raise ValueError('Body-rate norm was not preserved')
 midpoint,weights=g.exposure_quadrature(times,16);fit_times=np.unique(np.r_[times,midpoint])
 if len(midpoint)!=64 or len(fit_times)!=69:raise ValueError('Frozen 64 midpoint+5 knot construction differs')
 poses=g.orientations_at(times,wrong,fit_times)
 positions,trajectory_valid=g.dense_forward_trajectories(k,poses,height,width)
 displacements=positions-g.pixel_grid(height,width)[None]
 fit=g.fit_dense_quadratic(displacements,fit_times,trajectory_valid)
 ys,xs=slice(top,bottom),slice(left,right)
 valid=fit['fit_valid'][ys,xs]
 if not valid.all():raise CounterfactualIncomplete('wrong-motion descriptor invalid in frozen ROI',{'invalid_fit_pixels':int((~valid).sum()),'roi_pixels':valid.size})
 cp_xy=g.encode_displacements(fit['control_px'],16.)
 ep_xy=g.encode_displacements(fit['endpoint_px'],16.)
 rms,maximum=g.quadratic_errors(displacements,fit_times,cp_xy.astype(np.float64)*16.,ep_xy.astype(np.float64)*16.,fit['fit_valid'])
 cp=np.concatenate([cp_xy[ys,xs],np.zeros(valid.shape+(1,),np.float32)],axis=-1)
 ep=np.concatenate([ep_xy[ys,xs],np.zeros(valid.shape+(1,),np.float32)],axis=-1)
 if not np.isfinite(cp).all() or not np.isfinite(ep).all() or not np.isfinite(rms[ys,xs]).all() or not np.isfinite(maximum[ys,xs]).all():raise CounterfactualIncomplete('nonfinite wrong-motion descriptor/errors in frozen ROI')
 proof={'schema':'industrial-Rz90-descriptor-v1','addendum_sha256':ADDENDUM_SHA256,
  'base_sample_id':sample['metadata'].get('provenance',{}).get('sample_id','explicit-engineering-math-fixture'),
  'engineering_only':engineering,'rates_correct_rad_s':rates.tolist(),'rates_wrong_rad_s':wrong.tolist(),
  'rates_transform':RZ90.astype(int).tolist(),'times_seconds':times.tolist(),'midpoint_count':64,'fit_node_count':69,
  'fit_times_sha256':array_sha(fit_times),'exposure_weights_sha256':array_sha(weights),'k_full':k.tolist(),
  'full_size_wh':[width,height],'crop_box_full_xyxy':[left,top,right,bottom],'map_scale_px':16.,
  'angular_path_length_deg':float(np.rad2deg(np.dot(np.diff(times),norms))),
  'wrong_angular_path_length_deg':float(np.rad2deg(np.dot(np.diff(times),wrong_norms))),
  'original_rgb_sha256':array_sha(sample['blur_linear_rgb']),'original_target_sha256':array_sha(sample['sharp_linear_rgb']),
  'original_availability_sha256':array_sha(sample['valid_mask']),'original_network_sha256':array_sha(sample['network_input_9']),
  'cp_map_sha256':array_sha(cp),'ep_map_sha256':array_sha(ep),'fit_valid_sha256':array_sha(valid),
  'cp_mean_absolute_change':float(np.mean(np.abs(cp.astype(np.float64)-sample['cp_map']))),
  'ep_mean_absolute_change':float(np.mean(np.abs(ep.astype(np.float64)-sample['ep_map']))),
  'wrong_fit_error_rms_max_px':float(rms[ys,xs].max()),'wrong_fit_error_max_px':float(maximum[ys,xs].max()),
  'validity_policy':'all wrong fit nodes positive-depth/finite over whole original ROI; original availability unchanged; no image warp',
  'memory_estimate_bytes':estimate,'geometry_cap_bytes':max_geometry_bytes}
 return {'cp_map':cp,'ep_map':ep,'fit_valid':valid.copy(),'fit_times_seconds':fit_times,'identity':proof,'identity_sha256':digest(proof)}


def read_addendum_payload(value):
 # Pin canonical content as well as the external byte-level file. The payload
 # is normally obtained from read_addendum(); no mutable transform API exists.
 path=ROOT/'experiments/industrial-v1/evaluation-addendum-final-v1.json'
 if not path.exists():path=ROOT/'dependencies/evaluation-addendum-final-v1.json'
 expected=read_addendum(path)
 if value!=expected:raise ValueError('Counterfactual payload is not the frozen addendum')
 return expected


def network_input(sample,cf):
 identity=cf['identity']
 if digest(identity)!=cf['identity_sha256'] or array_sha(sample['network_input_9'])!=identity['original_network_sha256'] or array_sha(sample['blur_linear_rgb'])!=identity['original_rgb_sha256'] or array_sha(sample['sharp_linear_rgb'])!=identity['original_target_sha256'] or array_sha(sample['valid_mask'])!=identity['original_availability_sha256']:raise ValueError('Counterfactual descriptor does not bind this original sample')
 if array_sha(cf['cp_map'])!=identity['cp_map_sha256'] or array_sha(cf['ep_map'])!=identity['ep_map_sha256'] or not cf['fit_valid'].all() or array_sha(cf['fit_valid'])!=identity['fit_valid_sha256']:raise ValueError('Counterfactual maps/mask differ from checked geometry')
 result=sample['network_input_9'].copy();result[...,3:6]=cf['cp_map'];result[...,6:9]=cf['ep_map']
 if not np.array_equal(result[...,:3],sample['blur_linear_rgb']):raise ValueError('RGB changed in counterfactual construction')
 return result


class CounterfactualRestorer:
 """Explicit wrong-map input adapter around the SAME best full model."""
 def __init__(self,restorer,cf,executor_root=None):
  if restorer.contract['input_mode']!='full_maps':raise ValueError('Only the three original full checkpoints are counterfactually evaluated')
  self.base,self.cf=restorer,cf;self.contract,self.identity=restorer.contract,restorer.identity;self.executor=executor(executor_root)
 def predict_linear(self,sample):
  value=network_input(sample,self.cf)
  tensor=torch.from_numpy(value.transpose(2,0,1).copy())[None].to(self.base.device)
  self.base.model.eval()
  with torch.no_grad(),self.executor.v.fp32_inference(),torch.autocast(device_type=self.base.device.type,enabled=False):
   prediction=self.base.model(tensor)[-1]
  if prediction.dtype!=torch.float32 or prediction.shape!=(1,3,*sample['blur_linear_rgb'].shape[:2]) or not torch.isfinite(prediction).all():raise CounterfactualIncomplete('nonfinite/incompatible fixed-model counterfactual prediction')
  return prediction[0].detach().cpu().permute(1,2,0).contiguous().numpy()


def cf_group(category,level,seed):
 if type(seed) is not int or seed not in (0,1,2):raise ValueError('Original full seed0/1/2 required')
 return category+'__a'+format(level,'.12g').replace('.','p')+'__'+BRANCH+'-seed'+str(seed)


def main_bindings(x,plan,summary_path,summary_sha,threshold_path,threshold_sha,engineering):
 summary=x.v.strict_json(x.v.checked_bytes(summary_path,summary_sha));ctx=summary['context']
 if ctx!=x.execution_context(plan,'test',threshold_sha,ctx['engineering_limit_per_group']):raise ValueError('Main study summary context is not pinned E3')
 if ctx['engineering_only'] and not engineering:raise ValueError('Engineering primary result cannot authorize formal CF')
 thresholds=x.load_thresholds(plan,threshold_path,threshold_sha,engineering)
 if set(summary['branch_receipts_sha256'])!={x.branch_key(b,s) for b,s in x.BRANCHES}:raise ValueError('Complete main eleven-branch receipt required before CF')
 if not summary['official_test_source_coverage_complete'] and not engineering:raise ValueError('Formal primary source coverage is incomplete')
 return summary,thresholds


def correct_full_record(x,plan,row,seed,summary,summary_path):
 root=Path(summary_path).resolve().parent;branch='full_maps';key=x.group_key(row['category'],row['angle_deg'],branch,seed)
 bpath=root/(x.branch_key(branch,seed)+'-complete.json')
 branch_receipt=x.v.strict_json(x.v.checked_bytes(bpath,summary['branch_receipts_sha256'][x.branch_key(branch,seed)]))
 group=x.v.strict_json(x.v.checked_bytes(root/key/'group-receipt.json',branch_receipt['group_receipts'][key]))
 directory=root/key/row['sample_id']
 if sha_file(directory/'complete.json')!=group['completed_image_receipt_sha256'][row['sample_id']]:raise ValueError('Correct full per-image proof differs')
 expected=x.expected_image_identity(plan,row,summary['context'],branch,seed)
 loaded=x.completed_image(directory,expected)
 if loaded is None:raise ValueError('Correct full base image missing')
 return loaded[0],{'correct_full_record_sha256':sha_file(directory/'record.json'),'correct_full_complete_sha256':sha_file(directory/'complete.json')}


def counterfactual_context(x,plan,threshold_sha,main_summary_sha,limit,max_geometry_bytes):
 ctx=x.execution_context(plan,'test',threshold_sha,limit)
 ctx.update(schema='industrial-counterfactual-execution-v1',substudy=BRANCH,addendum_sha256=ADDENDUM_SHA256,
  base_main_summary_sha256=main_summary_sha,correct_full_thresholds_sha256=threshold_sha,
  execution_e3_manifest_sha256=E3_MANIFEST_SHA256,counterfactual_code_sha256=sha_file(__file__),geometry_cap_bytes=max_geometry_bytes)
 return ctx


def run(plan_path,plan_sha,threshold_path,threshold_sha,main_summary_path,main_summary_sha,output,*,addendum_path,
        write=False,resume=False,seed=None,engineering_limit=None,max_extra_bytes=32*1024**3,max_geometry_bytes=8*1024**3,executor_root=None):
 if not write:raise ValueError('Counterfactual inference requires explicit --write')
 x=executor(executor_root);addendum=read_addendum(addendum_path);plan=x.read_plan(plan_path,plan_sha)
 engineering=bool(plan['engineering_only'] or engineering_limit is not None)
 if not engineering and plan['prepared']['summary']['protocol_sha256']!=addendum['base_protocol_sha256']:raise ValueError('Wrong formal q16 source protocol')
 summary,thresholds=main_bindings(x,plan,main_summary_path,main_summary_sha,threshold_path,threshold_sha,engineering)
 if summary['context']['engineering_limit_per_group']!=engineering_limit:raise ValueError('CF and correct full must cover exactly the same source scope')
 seeds=(0,1,2) if seed is None else (seed,)
 for value in seeds:cf_group(plan['categories'][0],plan['prepared']['protocol']['angle_levels_deg'][0],value)
 x.resources(plan,'test',engineering_limit,max_extra_bytes,output)
 ctx=counterfactual_context(x,plan,threshold_sha,main_summary_sha,engineering_limit,max_geometry_bytes)
 root=x.open_execution(output,ctx,resume)
 for selected_seed in seeds:
  job=plan['runs']['full_maps',selected_seed]
  restorer=x.v.load_best_restorer(job['checkpoint_path'],job['run_report_path'],expected_checkpoint_sha256=job['checkpoint_sha256'],
    expected_run_report_sha256=job['run_report_sha256'],expected_allocation_sha256=plan['prepared']['summary']['allocation_sha256'],
    input_mode='full_maps',seed=selected_seed,device=plan['factory_kwargs'].get('device','cpu'),trainer_path=job['trainer_path'])
  group_receipts={}
  for category in plan['categories']:
   bank=plan['banks'][category];detector=x.v.load_shared_bank(bank['directory'],expected_metadata_sha256=bank['metadata_sha256'],
    expected_training_manifest_sha256=bank['selection_sha256'],factory_kwargs=plan['factory_kwargs'])
   for level in plan['prepared']['protocol']['angle_levels_deg']:
    key=cf_group(category,level,selected_seed);folder=root/key;folder.mkdir(exist_ok=True)
    rows=x.group_rows(plan,category,level,'test',engineering_limit);records=[]
    threshold=thresholds['groups'][x.group_key(category,level,'full_maps',selected_seed)]['image_threshold']
    for row in rows:
     directory=folder/row['sample_id'];directory.mkdir(exist_ok=True)
     correct,proof=correct_full_record(x,plan,row,selected_seed,summary,main_summary_path)
     sample,checked=x.v.load_compiled_sample(plan['prepared'],row['sample_id'])
     try:
      maps=recompute_maps(sample,addendum,engineering=engineering,max_geometry_bytes=max_geometry_bytes,executor_root=executor_root)
     except (CounterfactualIncomplete,ValueError) as error:
      x.atomic_json(directory/'incomplete.json',{'status':'incomplete','reason':getattr(error,'reason',str(error)),'diagnostics':getattr(error,'diagnostics',{}),'context':ctx,'sample_id':row['sample_id'],'npz_sha256':row['npz_sha256'],'required_scope':'all source-levels; no fallback/no dropped rows'})
      raise
     cfproof=dict(proof,base_sample_id=row['sample_id'],base_npz_sha256=row['npz_sha256'],addendum_sha256=ADDENDUM_SHA256,
        base_main_summary_sha256=main_summary_sha,correct_full_thresholds_sha256=threshold_sha,
        correct_full_threshold=threshold,descriptor_identity=maps['identity'],descriptor_identity_sha256=maps['identity_sha256'])
     expected=x.expected_image_identity(plan,row,ctx,'full_maps',selected_seed)
     expected.update(group=key,branch=BRANCH,counterfactual_sha256=digest(cfproof))
     cached=x.completed_image(directory,expected)
     if cached is not None:
      record=cached[0]
      if record.get('counterfactual')!=cfproof:raise ValueError('Completed CF record proof differs from fresh geometry/main binding')
     else:
      if restorer.identity!=correct['restorer'] or detector.fitted_identity!=correct['bank_identity']:raise ValueError('CF changed fixed model/bank')
      annotation=x.v.read_original_annotation(plan['prepared'],row['source_image_id'],plan['data_root'],row['source_mapping'])
      adapted=CounterfactualRestorer(restorer,maps,executor_root)
      record,_,_,crop=x.v.evaluate_one(sample,checked,annotation,branch='full_maps',detector=detector,restorer=adapted,image_threshold=threshold,return_crop_score=True)
      record.update(branch=BRANCH,group=key,counterfactual=cfproof)
      x.record_provenance(record,expected)
      x.atomic_npy(directory/'score-crop.npy',crop);x.atomic_json(directory/'record.json',record)
      x.atomic_json(directory/'complete.json',{'schema':'industrial-evaluated-image-v1','identity':expected,'record_sha256':sha_file(directory/'record.json'),
        'score_crop_sha256':sha_file(directory/'score-crop.npy'),'score_crop_hw':list(crop.shape)})
      del adapted
     records.append(record)
    metrics=x.group_aggregate(plan,rows,folder,records,threshold,max_extra_bytes)
    receipt={'schema':'industrial-counterfactual-group-v1','context':ctx,'category':category,'angle_deg':level,'branch':BRANCH,'seed':selected_seed,
      'sample_ids':[r['sample_id'] for r in rows],'source_ids':[r['source_image_id'] for r in rows],'count':len(rows),
      'completed_image_receipt_sha256':{r['sample_id']:sha_file(folder/r['sample_id']/'complete.json') for r in rows},'metrics':metrics}
    x.atomic_json(folder/'group-receipt.json',receipt);group_receipts[key]=sha_file(folder/'group-receipt.json')
   del detector
   if str(plan['factory_kwargs'].get('device','cpu')).startswith('cuda'):torch.cuda.empty_cache()
  x.atomic_json(root/(BRANCH+'-seed'+str(selected_seed)+'-complete.json'),{'schema':'industrial-counterfactual-branch-v1','context':ctx,'branch':BRANCH,'seed':selected_seed,'group_receipts':group_receipts})
  del restorer
  if str(plan['factory_kwargs'].get('device','cpu')).startswith('cuda'):torch.cuda.empty_cache()
 return {'status':'counterfactual branches complete; run aggregate for all-three-seed scope','seeds':list(seeds),'engineering_only':engineering}


def aggregate(plan_path,plan_sha,threshold_path,threshold_sha,main_summary_path,main_summary_sha,output,*,addendum_path,write=False,executor_root=None):
 if not write:raise ValueError('Counterfactual aggregate requires explicit --write')
 x=executor(executor_root);addendum=read_addendum(addendum_path);plan=x.read_plan(plan_path,plan_sha);root=Path(output);ctx=x.read_json(root/'execution.json')
 summary,thresholds=main_bindings(x,plan,main_summary_path,main_summary_sha,threshold_path,threshold_sha,ctx['engineering_only'])
 if ctx!=counterfactual_context(x,plan,threshold_sha,main_summary_sha,ctx['engineering_limit_per_group'],ctx['geometry_cap_bytes']) or ctx['engineering_limit_per_group']!=summary['context']['engineering_limit_per_group']:raise ValueError('CF execution/environment/scope identity differs')
 if ctx['addendum_sha256']!=ADDENDUM_SHA256 or ctx['plan_sha256']!=plan_sha or ctx['base_main_summary_sha256']!=main_summary_sha or ctx['correct_full_thresholds_sha256']!=threshold_sha or ctx['counterfactual_code_sha256']!=sha_file(__file__) or ctx['execution_e3_manifest_sha256']!=E3_MANIFEST_SHA256:raise ValueError('CF aggregate frozen context differs')
 groups={};branches={};descriptor_expectations={}
 for seed in (0,1,2):
  path=root/(BRANCH+'-seed'+str(seed)+'-complete.json');branch=x.read_json(path)
  expected_keys={cf_group(c,l,seed) for c in plan['categories'] for l in plan['prepared']['protocol']['angle_levels_deg']}
  if branch.get('schema')!='industrial-counterfactual-branch-v1' or branch['context']!=ctx or branch['seed']!=seed or branch['branch']!=BRANCH or set(branch['group_receipts'])!=expected_keys:raise ValueError('Incomplete CF seed/group scope')
  for category in plan['categories']:
   for level in plan['prepared']['protocol']['angle_levels_deg']:
    key=cf_group(category,level,seed);group=x.v.strict_json(x.v.checked_bytes(root/key/'group-receipt.json',branch['group_receipts'][key]))
    rows=x.group_rows(plan,category,level,'test',ctx['engineering_limit_per_group'])
    if group.get('schema')!='industrial-counterfactual-group-v1' or group['context']!=ctx or group['category']!=category or group['angle_deg']!=level or group['branch']!=BRANCH or group['seed']!=seed or group['sample_ids']!=[r['sample_id'] for r in rows] or group['source_ids']!=[r['source_image_id'] for r in rows] or group['count']!=len(rows):raise ValueError('CF strict source/level count differs')
    fixed_threshold=thresholds['groups'][x.group_key(category,level,'full_maps',seed)]['image_threshold']
    if group['metrics']['image_threshold']!=fixed_threshold:raise ValueError('CF group changed fixed correct-full threshold')
    for row in rows:
     directory=root/key/row['sample_id'];correct,proof=correct_full_record(x,plan,row,seed,summary,main_summary_path)
     receipt=x.v.strict_json(x.v.checked_bytes(directory/'complete.json',group['completed_image_receipt_sha256'][row['sample_id']]))
     record=x.v.strict_json(x.v.checked_bytes(directory/'record.json',receipt['record_sha256']))
     cf=record['counterfactual']
     if row['sample_id'] not in descriptor_expectations:
      original,_=x.v.load_compiled_sample(plan['prepared'],row['sample_id'])
      computed=recompute_maps(original,addendum,engineering=ctx['engineering_only'],max_geometry_bytes=ctx['geometry_cap_bytes'],executor_root=executor_root)
      descriptor_expectations[row['sample_id']]=(computed['identity'],computed['identity_sha256'])
     expected_descriptor,expected_descriptor_sha=descriptor_expectations[row['sample_id']]
     if cf['descriptor_identity']!=expected_descriptor or cf['descriptor_identity_sha256']!=expected_descriptor_sha or cf['correct_full_threshold']!=fixed_threshold or record['annotation']!=correct['annotation']:raise ValueError('CF geometry/original annotation/fixed threshold differs')
     a=correct['annotation']
     expected_stats=x.v.metrics.threshold_image_statistics(record['image_score'],row['source']['original_image_label'],fixed_threshold,
       original_defect_pixels=a['original_defect_pixels'],retained_defect_pixels=a['retained_defect_pixels'])
     if record['image_threshold_statistics']!=expected_stats:raise ValueError('CF operating rule does not reproduce the fixed correct-full threshold')
     if any(cf[k]!=v for k,v in proof.items()) or cf['base_sample_id']!=row['sample_id'] or cf['base_npz_sha256']!=row['npz_sha256'] or cf['addendum_sha256']!=ADDENDUM_SHA256 or cf['base_main_summary_sha256']!=main_summary_sha or cf['correct_full_thresholds_sha256']!=threshold_sha or digest(cf['descriptor_identity'])!=cf['descriptor_identity_sha256']:raise ValueError('CF proof/main linkage differs')
     expected=x.expected_image_identity(plan,row,ctx,'full_maps',seed);expected.update(group=key,branch=BRANCH,counterfactual_sha256=digest(cf))
     if x.completed_image(directory,expected) is None or record['restorer']!=correct['restorer'] or record['bank_identity']!=correct['bank_identity']:raise ValueError('CF model/bank identity differs')
    groups[key]=group
  branches[BRANCH+'-seed'+str(seed)]=sha_file(path)
 macro=[x.class_macro([groups[cf_group(c,l,seed)] for c in plan['categories']],plan['categories'],BRANCH,seed,l) for seed in (0,1,2) for l in plan['prepared']['protocol']['angle_levels_deg']]
 result={'schema':'industrial-counterfactual-summary-v1','context':ctx,'addendum_sha256':ADDENDUM_SHA256,
  'base_main_summary_sha256':main_summary_sha,'correct_full_thresholds_sha256':threshold_sha,
  'branch_receipts_sha256':branches,'group_metrics':{k:g['metrics'] for k,g in groups.items()},'class_macro_by_level_branch_seed':macro,
  'official_counterfactual_source_coverage_complete':not ctx['engineering_only'],'condition':BRANCH,
  'scope':'three original full seeds; all original sources/levels; correct-full fixed thresholds; no new bank/model selection',
  'statistics':'separate registered paired-statistics implementation; no required effect direction','descriptor_recomputed_unique_samples':len(descriptor_expectations),'scientific_accept':False}
 x.atomic_json(root/'summary.json',result);return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','aggregate'])
 for key in ('plan','plan-sha','thresholds','threshold-sha','main-summary','main-summary-sha','addendum','write'):p.add_argument('--'+key,required=True)
 p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--engineering-limit',type=int)
 p.add_argument('--max-extra-gib',type=int,default=32);p.add_argument('--max-geometry-gib',type=int,default=8);p.add_argument('--executor-root')
 a=p.parse_args();args=(a.plan,a.plan_sha,a.thresholds,a.threshold_sha,a.main_summary,a.main_summary_sha,a.write)
 if a.command=='run':result=run(*args,addendum_path=a.addendum,write=True,resume=a.resume,seed=a.seed,engineering_limit=a.engineering_limit,max_extra_bytes=a.max_extra_gib*1024**3,max_geometry_bytes=a.max_geometry_gib*1024**3,executor_root=a.executor_root)
 else:result=aggregate(*args,addendum_path=a.addendum,write=True,executor_root=a.executor_root)
 print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
