#!/usr/bin/env python3
"""Source-cluster paired statistics; no fitting, inference, pixel bootstrap or p-values.

Arithmetic fixture APIs are separate from the strict future E3/CF receipt adapter.
Three trained seeds are conditioned on, never resampled as independent subjects.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ADDENDUM_SHA256='f247bd244041c451e20389b2e6359568f7ee8041242eeee6e039faa753e44420'
SOURCE_SHA256='f72803e8fbde0ada85389f63808b581e03d927c7782d869ab53d55a53282dfd6'
SEEDS=(0,1,2)
CONDITIONS=('full_maps','rgb_zero_maps','endpoint_only','full_maps_cf_rz90')
CATEGORIES=('pcb1','pcb2','pcb3','pcb4')
LABEL={'normal':0,'anomaly':1}
BOOTSTRAP_SEED=20261105
REPLICATES=5000

E3_MANIFEST_SHA256='fea51917d95e34e10bed849e4bc85dd903eb40cab9e42086a5e68dcfdd5831cc'
CF1_MANIFEST_SHA256='0237c9951924a27039c407ab0e535e0966abbd041a7699bb4157f5ca4af8213f'
CF1_CODE_SHA256='7488448c969f98c69d64a77b3e125d734f366d5bd8d4caaeb655d78fa0cfe61a'
E3_CODE_SHA256={'float_dataset.py': 'cde8c7549760ae0323c74ca0e16c427ada101363791385324d4a5ef26b94100f', 'industrial_evaluate.py': 'fe5201571f01fab7a94ce70da4418eff6caaca8b1501858307d0c6326280bdc7', 'industrial_execution.py': '1fd7ad1399ea1374c49cf4d1bd4b7b27582ca0833c18aa84a16deb134fd6ad0e', 'industrial_geometry.py': 'cf7d9a295b45f8ff0c57af8ee5580e7583944076090376c3f2024d5c3b5d57f8', 'industrial_metrics.py': '39486df61cbcf0d5344f904968a4adf28dcc5981288042cce9cb18ab912e2053'}
s=importlib.util.spec_from_file_location('statistics_frozen_ranking_validation',HERE/'industrial_metrics.py')
metrics=importlib.util.module_from_spec(s);s.loader.exec_module(metrics)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


LOADED_CODE_SHA256={name:sha(HERE/name) for name in ('industrial_statistics.py','industrial_metrics.py')}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def read_json(path):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def checked_json(path,expected):
    if sha(path)!=expected:raise ValueError('Artifact SHA differs: '+str(path))
    value=read_json(path)
    if sha(path)!=expected:raise ValueError('Artifact changed during read')
    return value


def _finite_scalar(value,name,nonnegative=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):
        raise ValueError('Finite numeric '+name+' required')
    if isinstance(value,(int,np.integer)) and not -(2**53)<=int(value)<=2**53:
        raise ValueError('Integer outside exact float64 domain')
    if isinstance(value,np.floating) and np.asarray(value).dtype.itemsize>8:raise ValueError('Floating precision wider than float64 unsupported')
    value=float(value)
    if not math.isfinite(value) or nonnegative and value<0:raise ValueError('Invalid '+name)
    return value


def _weights(weights,n):
    w=np.asarray(weights)
    if w.shape!=(n,) or w.dtype.kind not in 'biu' or np.any(w<0):raise ValueError('Matching nonnegative integer source multiplicities required')
    if not 0<sum(int(x) for x in w)<=2**53:raise ValueError('Nonempty exact-count weight domain required')
    return w.astype(np.int64)


def _rank_rows(scores,labels,weights):
    """Each row of weights is a paired source resample, ties merged before metrics."""
    order=np.argsort(-scores,kind='stable');ordered=scores[order];truth=labels[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    w=weights[:,order]
    p=np.add.reduceat(w*truth[None,:],starts,axis=1).astype(np.float64)
    n=np.add.reduceat(w*(~truth)[None,:],starts,axis=1).astype(np.float64)
    P=p.sum(axis=1);N=n.sum(axis=1);tp=np.cumsum(p,axis=1);fp=np.cumsum(n,axis=1)
    oldtp=np.c_[np.zeros(len(w)),tp[:,:-1]]
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    auroc=np.full(len(w),np.nan);ap=np.full(len(w),np.nan);valid=(P>0)&(N>0)
    auroc[valid]=np.sum(n[valid]/N[valid,None]*(tp[valid]+oldtp[valid])/(2*P[valid,None]),axis=1)
    ap[valid]=np.sum(p[valid]/P[valid,None]*precision[valid],axis=1)
    return auroc,ap,P,N


def weighted_ranking(scores,labels,weights):
    """Exact multiplicity AUROC/AP, equal to explicit np.repeat expansion.

    Score-domain validation is shared with frozen M3, including raw mixed-int guard.
    Both class masses are required, matching the reviewed M3 undefined policy.
    """
    x,y=metrics._scores_labels(scores,labels);w=_weights(weights,len(x))
    auc,ap,p,n=_rank_rows(x,y,w[None,:])
    return {'auroc':None if not np.isfinite(auc[0]) else float(auc[0]),
            'average_precision':None if not np.isfinite(ap[0]) else float(ap[0]),
            'positive_count':int(p[0]),'negative_count':int(n[0]),
            'tie_policy':'Exact equal score groups with integer source multiplicities; noninterpolated AP'}


def ordered_sources(sources):
    output=[];seen=set()
    for source in sources:
        if set(source)!={'source_image_id','category','original_image_label'}:raise ValueError('Exact source metadata keys required')
        sid=source['source_image_id'];category=source['category'];label=source['original_image_label']
        if not isinstance(sid,str) or not sid or sid in seen or category not in CATEGORIES or label not in LABEL:
            raise ValueError('Unique source IDs, PCB categories and original binary labels required')
        seen.add(sid);output.append(dict(source))
    if not output:raise ValueError('Nonempty source clusters required')
    return sorted(output,key=lambda r:(r['category'],LABEL[r['original_image_label']],r['source_image_id']))


def draw_source_weights(sources,replicates=REPLICATES,seed=BOOTSTRAP_SEED):
    """One multiplicity per source; its all levels/seeds/conditions share this column."""
    if type(replicates) is not int or replicates<1 or type(seed) is not int:raise ValueError('Explicit positive replicate count/integer seed required')
    sources=ordered_sources(sources);strata={}
    for i,r in enumerate(sources):strata.setdefault((r['category'],LABEL[r['original_image_label']]),[]).append(i)
    rng=np.random.Generator(np.random.PCG64(seed));weights=np.zeros((replicates,len(sources)),dtype=np.int64)
    for rep in range(replicates):
        for key,indices in sorted(strata.items()):
            weights[rep,indices]=np.bincount(rng.integers(0,len(indices),size=len(indices),dtype=np.int64),minlength=len(indices))
    return weights,{'strata':[{'category':key[0],'original_binary_label':key[1],'source_count':len(indices),
                              'source_ids':[sources[i]['source_image_id'] for i in indices]} for key,indices in sorted(strata.items())],
                    'source_ids':[r['source_image_id'] for r in sources], 'source_order':'category, normal=0/anomaly=1, lexicographic source_image_id',
                    'multiplicities_sha256':hashlib.sha256((str(weights.dtype)+'|'+str(weights.shape)+'|').encode()+weights.tobytes()).hexdigest(),
                    'generator':'NumPy PCG64','seed':seed,'replicates':replicates,
                    'monte_carlo_order':'replicate outer loop; strata category then 0 normal then 1 anomaly; one rng.integers(0,n,size=n) per stratum; bincount into sorted source columns'}


def normalize_e3_record(record,*,proof=None):
    """Adapt actual E3/common CF record fields; no inference or invented runner API."""
    required={'sample_id','npz_sha256','source_image_id','category','original_image_label','angle_deg','branch','seed',
              'split_name','restoration','image_score','restorer','bank_identity','detector_contract_sha256'}
    if not isinstance(record,dict) or not required<=record.keys() or record['split_name']!='test':raise ValueError('Explicit E3 TEST record fields required')
    condition=record['branch'];seed=record['seed'];restorer=record['restorer']
    if condition not in CONDITIONS or type(seed) is not int or seed not in SEEDS or not isinstance(restorer,dict):raise ValueError('Matched restored condition/seed required')
    required_model={'checkpoint_sha256','run_report_sha256','best_model_sha256','selected_update','mode','seed','allocation_sha256','num_res','total_updates',
                    'inference_torch','inference_numpy','inference_device','inference_precision'}
    if not required_model<=restorer.keys():raise ValueError('Complete E3 restoration identity required')
    def valid_digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for key in ('checkpoint_sha256','run_report_sha256','best_model_sha256','allocation_sha256'):
        if not valid_digest(restorer[key]):raise ValueError('Restoration identity SHA required')
    if any(type(restorer[k]) is not int or restorer[k]<1 for k in ('selected_update','num_res','total_updates')) or restorer['selected_update']>restorer['total_updates']:
        raise ValueError('Restoration update/architecture identity required')
    bank=record['bank_identity']
    if not isinstance(bank,dict) or not {'training_manifest_sha256','bank_sha256','metadata_sha256'}<=bank.keys() or not all(valid_digest(bank[k]) for k in ('training_manifest_sha256','bank_sha256','metadata_sha256')):
        raise ValueError('Class shared bank identity required')
    if not valid_digest(record['npz_sha256']) or not valid_digest(record['detector_contract_sha256']):raise ValueError('Observation/detector digest required')
    if not isinstance(record['sample_id'],str) or not record['sample_id']:raise ValueError('Explicit sample ID required')
    mode='full_maps' if condition=='full_maps_cf_rz90' else condition
    if restorer.get('mode')!=mode or restorer.get('seed')!=seed:raise ValueError('Record/restorer mode or seed differs')
    if condition=='full_maps_cf_rz90':
        cf=record.get('counterfactual',{})
        if not isinstance(cf.get('descriptor_identity'),dict) or hashlib.sha256(canonical(cf['descriptor_identity']).encode()).hexdigest()!=cf.get('descriptor_identity_sha256'):raise ValueError('CF descriptor proof digest differs')
        if cf.get('addendum_sha256')!=ADDENDUM_SHA256 or cf.get('base_sample_id')!=record['sample_id'] or cf.get('base_npz_sha256')!=record['npz_sha256']:
            raise ValueError('CF addendum/base sample/NPZ join differs')
    elif 'counterfactual' in record:raise ValueError('Main branch must not masquerade as CF')
    return {'source_image_id':record['source_image_id'],'category':record['category'],'original_image_label':record['original_image_label'],
            'angle_deg':_finite_scalar(record['angle_deg'],'angle',True),'seed':seed,'condition':condition,
            'linear_mse':_finite_scalar(record['restoration']['mse'],'linear MSE',True),
            'image_score':_finite_scalar(record['image_score'],'image score'),
            'identity':{key:copy.deepcopy(record[key]) for key in ('sample_id','npz_sha256','restorer','bank_identity','detector_contract_sha256')},
            'counterfactual':copy.deepcopy(record.get('counterfactual')),'proof':copy.deepcopy(proof)}


def prepare_panel(rows,sources,levels):
    """Strict paired join. Missing/extra/duplicate cells or differing metadata fail."""
    sources=ordered_sources(sources);levels=tuple(_finite_scalar(v,'angle',True) for v in levels)
    if len(levels)!=3 or len(set(levels))!=3 or min(levels)<=0:raise ValueError('Three fixed positive unique severity levels required')
    if set(r['category'] for r in sources)!=set(CATEGORIES) or any(not any(r['category']==c and r['original_image_label']==label for r in sources) for c in CATEGORIES for label in LABEL):
        raise ValueError('All four categories x both original-label strata required')
    by_id={r['source_image_id']:r for r in sources};seen={};source_index={r['source_image_id']:i for i,r in enumerate(sources)}
    mse=np.empty((4,3,3,len(sources)),np.float64);score=np.empty_like(mse)
    sample_identity={};banks={};models={};allocation=None
    for row in rows:
        required={'source_image_id','category','original_image_label','angle_deg','seed','condition','linear_mse','image_score','identity','counterfactual','proof'}
        if set(row)!=required:raise ValueError('Exact normalized record keys required')
        sid=row['source_image_id'];source=by_id.get(sid)
        if source is None or any(row[k]!=source[k] for k in ('category','original_image_label')):raise ValueError('Result/source key or original label mismatch')
        if type(row['seed']) is not int or row['seed'] not in SEEDS or row['condition'] not in CONDITIONS or row['angle_deg'] not in levels:raise ValueError('Unexpected seed/condition/severity')
        key=(sid,row['angle_deg'],row['seed'],row['condition'])
        if key in seen:raise ValueError('Duplicate paired source/level/seed/condition cell')
        seen[key]=row;ident=row['identity'];obs=(ident['sample_id'],ident['npz_sha256'])
        observation_key=(sid,row['angle_deg'])
        if sample_identity.setdefault(observation_key,obs)!=obs:raise ValueError('Paired observation sample/NPZ differs')
        bank=canonical([ident['bank_identity'],ident['detector_contract_sha256']])
        if banks.setdefault(row['category'],bank)!=bank:raise ValueError('Class-specific shared bank/detector identity differs')
        restorer=ident['restorer'];modelkey=(row['condition'],row['seed'])
        if models.setdefault(modelkey,canonical(restorer))!=canonical(restorer):raise ValueError('Condition/seed model changes across sources/levels')
        current=restorer['allocation_sha256']
        if allocation is None:allocation=current
        if current!=allocation:raise ValueError('Matched data allocation differs')
        a,b,c,d=CONDITIONS.index(row['condition']),SEEDS.index(row['seed']),levels.index(row['angle_deg']),source_index[sid]
        mse[a,b,c,d]=_finite_scalar(row['linear_mse'],'linear MSE',True);score[a,b,c,d]=_finite_scalar(row['image_score'],'image score')
    if len(seen)!=len(sources)*3*3*4:raise ValueError('Missing paired source/level/seed/condition cell')
    core=('checkpoint_sha256','run_report_sha256','best_model_sha256','selected_update','mode','seed','allocation_sha256','num_res','total_updates')
    for sid,level,seed,condition in seen:
        if condition!='full_maps_cf_rz90':continue
        full=seen[sid,level,seed,'full_maps'];cf=seen[sid,level,seed,condition]
        if any(cf['identity']['restorer'].get(k)!=full['identity']['restorer'].get(k) for k in core):raise ValueError('CF must use the same best full checkpoint')
        if full['proof'] is not None and 'image_threshold' in full['proof'] and (cf['proof'] is None or cf['proof'].get('image_threshold')!=full['proof']['image_threshold'] or cf['counterfactual'].get('correct_full_threshold')!=full['proof']['image_threshold']):raise ValueError('CF changed correct-full frozen threshold')
        if full['proof'] is not None and (cf['counterfactual'].get('correct_full_record_sha256')!=full['proof']['record_sha256'] or cf['counterfactual'].get('correct_full_complete_sha256')!=full['proof']['complete_sha256']):
            raise ValueError('CF does not join exact base full result receipts')
    return {'sources':sources,'levels':levels,'mse':mse,'scores':score,'row_count':len(seen),'allocation_sha256':allocation,'normalized_rows':[seen[k] for k in sorted(seen,key=lambda k:(source_index[k[0]],levels.index(k[1]),k[2],CONDITIONS.index(k[3])))]}


def paired_statistics(panel,*,replicates=REPLICATES,return_replicates=False):
    """Fixture-safe arithmetic. Production adapter separately gates 802/all receipts."""
    sources=panel['sources'];weights,design=draw_source_weights(sources,replicates=replicates)
    allweights=np.vstack([np.ones((1,len(sources)),np.int64),weights])
    labels=np.array([LABEL[r['original_image_label']] for r in sources],bool)
    contrasts={'full_vs_zero':1,'full_vs_endpoint':2,'correct_full_vs_rz90':3}
    pieces={name:{metric:[] for metric in ('mse_reduction','image_auroc_delta','image_ap_delta')} for name in contrasts}
    detail=[]
    for category in CATEGORIES:
        indices=np.array([i for i,r in enumerate(sources) if r['category']==category]);w=allweights[:,indices];total=w.sum(axis=1)
        for li,level in enumerate(panel['levels']):
            for si,seed in enumerate(SEEDS):
                ranked={}
                for ci,condition in enumerate(CONDITIONS):
                    ranked[ci]=_rank_rows(panel['scores'][ci,si,li,indices],labels[indices],w)[:2]
                for name,baseline in contrasts.items():
                    # Positive means lower MSE/higher task score with correct full.
                    reduction=(w@(panel['mse'][baseline,si,li,indices]-panel['mse'][0,si,li,indices]))/total
                    auc=ranked[0][0]-ranked[baseline][0];ap=ranked[0][1]-ranked[baseline][1]
                    if not all(np.isfinite(v).all() for v in (reduction,auc,ap)):raise ValueError('Nonfinite resampled endpoint; no silent dropping')
                    for metric,value in zip(pieces[name],(reduction,auc,ap)):pieces[name][metric].append(value)
                    detail.append({'contrast':name,'category':category,'angle_deg':level,'seed':seed,
                                   'mse_reduction':float(reduction[0]),'image_auroc_delta':float(auc[0]),'image_ap_delta':float(ap[0])})
    results={};replicate_values={}
    for name,metrics_by_name in pieces.items():
        results[name]={}
        for metric,values in metrics_by_name.items():
            macro=np.stack(values).mean(axis=0)
            replicate_values[name+'__'+metric]=macro[1:].copy()
            primary=name=='full_vs_zero' and metric in ('mse_reduction','image_auroc_delta')
            quantiles=[.0125,.9875] if primary else [.025,.975]
            seed_points={str(seed):float(np.mean([r[metric] for r in detail if r['contrast']==name and r['seed']==seed])) for seed in SEEDS}
            point_rows=[r for r in detail if r['contrast']==name]
            severity={}
            for level in panel['levels']:
                level_values=[v for v,row in zip(values,point_rows) if row['angle_deg']==level]
                level_macro=np.stack(level_values).mean(axis=0)
                severity[str(level)]={'point':float(level_macro[0]),
                    'interval_95_exploratory':np.quantile(level_macro[1:],[.025,.975],method='linear').tolist()}
            results[name][metric]={'point':float(macro[0]),'interval':np.quantile(macro[1:],quantiles,method='linear').tolist(),
                'interval_coverage':.975 if primary else .95,'quantiles':quantiles,'interval_method':'percentile, numpy.quantile method=linear',
                'status':'co-primary Bonferroni alpha=.05/2' if primary else 'exploratory',
                'full_condition':'full_maps','comparator_condition':CONDITIONS[contrasts[name]],
                'effect_definition':'comparator MSE - correct-full MSE; positive favors correct full' if metric=='mse_reduction' else 'correct-full image ranking metric - comparator; positive favors correct full',
                'seed_point_effects':seed_points,'seed_point_range':[min(seed_points.values()),max(seed_points.values())],
                'severity_specific_exploratory':severity}
    result={'schema':'industrial-source-cluster-paired-statistics-v1','arithmetic_only':True,'formal_result_coverage_verified':False,'design':design,'contrasts':results,
            'point_by_category_level_seed':detail,'source_count':len(sources),'conditioned_seeds':list(SEEDS),
            'cluster_repeated_cells':9,'pairing':'one source multiplicity shared across all conditions/three levels/three seeds',
            'interpretation_limit':'Conditional on these three trained seeds and fixed class/label prevalence; source-image clusters, physical board/session independence unknown; no training-seed population inference',
            'pixel_bootstrap':False,'p_values':False,'must_win_rule':False,'numpy_version':np.__version__,
            'claim_rule':'Any joint supported-benefit claim must assess both co-primary endpoints; no sign-based selection or required benefit'}
    return (result,replicate_values) if return_replicates else result


def _path(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or '\x00' in relative:raise ValueError('Unsafe receipt path')
    root=Path(root).resolve();rel=Path(relative);path=root/rel
    if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root):raise ValueError('Receipt traversal')
    cursor=root
    for part in rel.parts:
        cursor/=part
        if cursor.is_symlink():raise ValueError('Receipt symlink refused')
    return path


def _branch_key(condition,seed):return condition if seed is None else condition+'-seed'+str(seed)
def _group_key(category,level,condition,seed):return category+'__a'+format(level,'.12g').replace('.','p')+'__'+_branch_key(condition,seed)


def _read_result_tree_schema_only(summary_path,expected_sha,*,kind,levels,anchor=None):
    """Read actual E3/CF JSON receipt chain, not models or native pixel arrays.

    Image statistics use record.json scalars; native score crops remain separately
    audited by the producer and are not independent resampling units here.
    Coverage is NOT established by this lower-level reader alone.
    """
    root=Path(summary_path).resolve().parent;summary=checked_json(summary_path,expected_sha)
    if kind=='main':
        if summary.get('schema')!='industrial-evaluation-summary-v1' or (summary.get('official_test_source_coverage_complete') is not True and not (anchor and anchor['engineering'])):raise ValueError('Complete E3 main summary required')
        specs=[('sharp',None),('blur',None)]+[(condition,seed) for condition in CONDITIONS[:3] for seed in SEEDS]
        branch_schema='industrial-evaluation-branch-v1';group_schema='industrial-evaluation-group-v1'
    elif kind=='cf':
        if summary.get('schema')!='industrial-counterfactual-summary-v1' or summary.get('addendum_sha256')!=ADDENDUM_SHA256 or (summary.get('official_counterfactual_source_coverage_complete') is not True and not (anchor and anchor['engineering'])):raise ValueError('Explicit frozen CF summary required')
        specs=[('full_maps_cf_rz90',seed) for seed in SEEDS]
        branch_schema='industrial-counterfactual-branch-v1';group_schema='industrial-counterfactual-group-v1'
    else:raise ValueError('Explicit main or cf adapter required')
    context=summary['context']
    if anchor is None:
        if context.get('stage')!='test' or context.get('engineering_only') is not False or context.get('engineering_limit_per_group') is not None:raise ValueError('Schema-only fixtures cannot establish engineering/production anchors')
    else:_check_execution_context(context,anchor,kind=kind)
    heads=summary['branch_receipts_sha256']
    if set(heads)!={_branch_key(c,s) for c,s in specs}:raise ValueError('Missing/extra completed result branches')
    output=[];proofs=[]
    common=('sample_id','npz_sha256','group','source_image_id','category','angle_deg','split_name','original_image_label','branch','seed','bank_identity','detector_contract_sha256','restorer')
    for condition,seed in specs:
        branchkey=_branch_key(condition,seed);branch=checked_json(_path(root,branchkey+'-complete.json'),heads[branchkey])
        if branch.get('schema')!=branch_schema or branch.get('context')!=context or branch.get('branch')!=condition or branch.get('seed')!=seed:raise ValueError('Branch receipt identity mismatch')
        categories=anchor['categories'] if anchor is not None else CATEGORIES
        expected_groups={_group_key(c,l,condition,seed):(c,float(l)) for c in categories for l in levels}
        if set(branch['group_receipts'])!=set(expected_groups):raise ValueError('Missing/extra class/severity groups')
        for groupkey,(category,level) in expected_groups.items():
            group=checked_json(_path(root,groupkey+'/group-receipt.json'),branch['group_receipts'][groupkey])
            if group.get('schema')!=group_schema or group.get('context')!=context or group.get('category')!=category or group.get('angle_deg')!=level or group.get('branch')!=condition or group.get('seed')!=seed:raise ValueError('Group receipt identity mismatch')
            ids=group['sample_ids'];sourceids=group['source_ids'];imageheads=group['completed_image_receipt_sha256']
            if len(ids)!=group['count'] or len(sourceids)!=len(ids) or len(set(ids))!=len(ids) or len(set(sourceids))!=len(sourceids) or set(imageheads)!=set(ids):raise ValueError('Duplicate/missing group sample/source keys')
            if anchor is not None:
                expected_rows=sorted((r for r in anchor['compiled']['index'].values() if r['category']==category and r['angle_deg']==level),key=lambda r:r['source_image_id'])
                limit=context['engineering_limit_per_group'];expected_rows=expected_rows if limit is None else expected_rows[:limit]
                if ids!=[r['sample_id'] for r in expected_rows] or sourceids!=[r['source_image_id'] for r in expected_rows]:raise ValueError('Result group differs from authoritative compiled source/level index')
            threshold=_finite_scalar(group['metrics']['image_threshold'],'frozen image threshold')
            for sampleid,sourceid in zip(ids,sourceids):
                directory=_path(root,groupkey+'/'+sampleid)
                complete_sha=imageheads[sampleid];complete=checked_json(directory/'complete.json',complete_sha)
                identity=complete['identity']
                if complete.get('schema')!='industrial-evaluated-image-v1' or identity.get('execution')!=context:raise ValueError('Image completion context mismatch')
                record=checked_json(directory/'record.json',complete['record_sha256'])
                if any(key not in record or record[key]!=identity.get(key) for key in common):raise ValueError('Record/completion identity mismatch')
                if kind=='cf' and identity.get('counterfactual_sha256')!=hashlib.sha256(canonical(record.get('counterfactual')).encode()).hexdigest():raise ValueError('CF completion does not bind counterfactual proof')
                expected={'sample_id':sampleid,'source_image_id':sourceid,'category':category,'angle_deg':level,'split_name':'test','group':groupkey,'branch':condition,'seed':seed}
                if any(record[key]!=value for key,value in expected.items()):raise ValueError('Result/group/source key mismatch')
                if complete['score_crop_hw']!=identity['score_crop_hw'] or not isinstance(complete['score_crop_sha256'],str) or len(complete['score_crop_sha256'])!=64:raise ValueError('Native score receipt metadata mismatch')
                if anchor is not None:_bind_record_to_anchor(record,complete,anchor,context)
                proof={'source_image_id':sourceid,'category':category,'original_image_label':record['original_image_label'],
                       'angle_deg':level,'condition':condition,'seed':seed,'sample_id':sampleid,
                       'summary_sha256':expected_sha,'branch_sha256':heads[branchkey],
                       'group_sha256':branch['group_receipts'][groupkey],'complete_sha256':complete_sha,
                       'record_sha256':complete['record_sha256'],'image_threshold':threshold}
                proofs.append(proof)
                if condition in CONDITIONS:output.append(normalize_e3_record(record,proof=proof))
    return output,summary,proofs


def check_source_coverage(proofs,sources,levels,*,kind):
    """The non-statistical sharp/blur baselines also retain every official source."""
    by_id={r['source_image_id']:r for r in ordered_sources(sources)}
    specs=[('sharp',None),('blur',None)]+[(c,s) for c in CONDITIONS[:3] for s in SEEDS] if kind=='main' else [('full_maps_cf_rz90',s) for s in SEEDS]
    expected={(category,float(level),condition,seed):{r['source_image_id'] for r in sources if r['category']==category}
              for category in CATEGORIES for level in levels for condition,seed in specs}
    actual={}
    for proof in proofs:
        sid=proof['source_image_id'];source=by_id.get(sid)
        if source is None or any(proof[k]!=source[k] for k in ('category','original_image_label')):raise ValueError('Receipt source/original label differs from official source metadata')
        key=(proof['category'],proof['angle_deg'],proof['condition'],proof['seed'])
        if sid in actual.setdefault(key,set()):raise ValueError('Duplicate source in complete result group')
        actual[key].add(sid)
    if actual!=expected:raise ValueError('Missing/extra official source coverage in result group')


C3_CODE_SHA256={'industrial_compile.py':'a16fed03c20b622a193718bec471e50ffc0673f1911d4cde7035ef758974aeb2','industrial_geometry.py':'cf7d9a295b45f8ff0c57af8ee5580e7583944076090376c3f2024d5c3b5d57f8','imu_synthesis.py':'18b50dbf8bd2d317c75c9ed557628d831ee22b483d33d8e33181fbf01228fd37','float_dataset.py':'cde8c7549760ae0323c74ca0e16c427ada101363791385324d4a5ef26b94100f'}
TRAINING_PLAN_SHA256='d61f62c0f80cd4b6e2b14e559142d9e9960b35b69c49446c1ca0cc9ca1648758'
CONTEXT_FIELDS={'schema','stage','plan_sha256','threshold_sha256','engineering_limit_per_group','engineering_only','code_sha256','torch','numpy','scipy','device','cuda_runtime','gpu','cpu_threads','cpu_interop_threads'}


def _json_lines(path,expected):
    before=sha(path)
    if before!=expected:raise ValueError('Metadata JSONL hash differs')
    values=[read_json_line(line) for line in Path(path).read_text().splitlines()]
    if sha(path)!=before:raise ValueError('Metadata changed during read')
    return values


def _resolve(base,value):
    if not isinstance(value,str) or not value:raise ValueError('Explicit metadata path required')
    return (Path(base)/value).resolve()


def _compiler_metadata():
    if any(sha(HERE/name)!=expected for name,expected in C3_CODE_SHA256.items()):raise ValueError('Metadata lookup helpers differ from frozen C3')
    spec=importlib.util.spec_from_file_location('statistics_c3_metadata',HERE/'industrial_compile.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def _compiled_anchor(plan,base,engineering):
    comp=plan['compiled']
    if set(comp)!={'directory','source_manifest','data_root','summary_sha256'}:raise ValueError('Exact E3 compiled binding required')
    root=_resolve(base,comp['directory']);summary=checked_json(root/'compile-summary.json',comp['summary_sha256'])
    if summary.get('complete') is not True or summary.get('wrote_files') is not True or summary['code_sha256']!=C3_CODE_SHA256:raise ValueError('Complete reviewed C3 compilation required')
    protocol=checked_json(root/'protocol.json',summary['protocol_sha256']);allocation=checked_json(root/'allocation.json',summary['allocation_sha256'])
    records=_json_lines(root/'compile-records.jsonl',summary['compile_records_sha256'])
    sources=_json_lines(_resolve(base,comp['source_manifest']),summary['source']['source_manifest_sha256'])
    inspection=checked_json(summary['source']['inspection_summary_resolved'],summary['source']['summary_sha256'])
    if (inspection.get('inspection_complete') is not True or inspection['source_manifest_sha256']!=summary['source']['source_manifest_sha256']
        or inspection['source_inspector_sha256']!='aefe277ea601b811fe8fa4ce538bfb6b0573c5f5d7a561e205a4f6609eadf99d'
        or any(inspection.get(k)!=0 for k in ('annotation_label_mismatch_count','annotation_mask_path_mismatch_count','annotation_label_token_mask_class_mismatch_count'))):raise ValueError('Complete S2 source identity required')
    if not engineering and (summary['source']['source_manifest_sha256']!=SOURCE_SHA256 or summary['protocol_sha256']!='86778bc7cf0b77074e1e30846124cfd6ffd64417085b81a8bbdfb6f7944f3e86' or protocol['source_scope']!='visa-pcb1-4-official'):raise ValueError('Registered q16/f828 compilation required')
    if allocation.get('allocation_schema')!='imda-float-allocation-v2' or len({r['source_image_id'] for r in sources})!=len(sources):raise ValueError('Allocation/source schema or uniqueness differs')
    c=_compiler_metadata();source_by_id={r['source_image_id']:r for r in sources}
    assignments={r['source_image_id']:'test' for r in sources if r['split']=='test'}
    for category in sorted({r['object'] for r in sources}):
        normal=sorted((r for r in sources if r['object']==category and r['split']=='train' and r['label']=='normal'),key=lambda r:c.digest([protocol['split_seed'],category,r['image_info']['decoded_pixel_sha256']]))
        if len(normal)<2:raise ValueError('Original normal train/valid allocation cannot be reconstructed')
        for i,r in enumerate(normal):assignments[r['source_image_id']]='valid' if i<math.ceil(.2*len(normal)) else 'train'
    expected={}
    for source in sources:
        split=assignments[source['source_image_id']];family,_,_,_,levels=c._choices(source,split,protocol,summary['protocol_sha256'])
        mapping=c.geometry.roi_geometry(c.geometry.resize_geometry((source['image_info']['width'],source['image_info']['height']),protocol['long_side']),protocol['margin'])
        k=c.geometry.simulated_k(mapping['full_size_wh'],protocol['horizontal_fov_degrees']);margin=protocol['margin']
        mapping.update(camera_definition={'simulated_full_canvas_horizontal_fov_degrees':float(protocol['horizontal_fov_degrees'])},k_full=k.tolist(),k_crop=(np.array([[1,0,-margin],[0,1,-margin],[0,0,1]])@k).tolist())
        for level in levels:expected['industrial-'+c.digest([family,level])]=(source,split,family,level,mapping)
    entries={r['sample_id']:r for r in allocation['samples']}
    if (len(entries)!=len(allocation['samples']) or set(entries)!=set(expected) or len(records)!=len(entries)
        or [r['sample_id'] for r in records]!=[r['sample_id'] for r in allocation['samples']]
        or allocation['allocation_id']!='industrial-'+c.digest([summary['source'],summary['protocol_sha256']])):raise ValueError('C3 authoritative allocation/record set or order differs')
    index={};npz_records=[];paths=set()
    for record in records:
        entry=entries[record['sample_id']];source,split,family,level,mapping=expected[record['sample_id']]
        expected_source={'image':source['image'],'image_file_sha256':source['image_info']['file_sha256'],'image_decoded_pixel_sha256':source['image_info']['decoded_pixel_sha256'],'mask':source['mask'] or None,'mask_file_sha256':source['mask_info']['file_sha256'] if source['mask_info'] else None,'original_image_label':source['label'],'original_split':source['split'],'category':source['object']}
        provenance={'sample_id':entry['sample_id'],'source_dataset_id':protocol['dataset_id'],'source_dataset_version':protocol['dataset_version'],'source_image_id':source['source_image_id'],'group_kind':'source_image','group_id':source['source_image_id'],'trajectory_family_id':family,'trajectory_id':'simulated:'+c.digest([family,level]),'original_split_id':inspection['official_split_sha256']+':'+source['split'],'split_name':split,'physical_scene_id':None,'physical_session_id':None}
        if {k:v for k,v in entry.items() if k!='npz'}!=provenance or record['source']!=expected_source or record['source_mapping']!=mapping or record['trajectory_family_id']!=family or record['npz']!=entry['npz']:raise ValueError('C3 source/level/label/ROI/provenance differs')
        _path(root,entry['npz'])  # Safety only: do NOT open or rehash NPZ payloads.
        if entry['npz'] in paths:raise ValueError('Duplicate compiled NPZ path')
        paths.add(entry['npz']);annotation=record['annotation'];original=source['mask_info']['positive_pixels'] if source['mask_info'] else 0
        if annotation['image_label']!=source['label'] or annotation['original_defect_pixels']!=original or not 0<=annotation['retained_defect_pixels']<=original:raise ValueError('Compiled original annotation differs from S2 source')
        provenance['split_manifest_sha256']=summary['allocation_sha256'];width,height=mapping['crop_size_wh']
        npz_records.append({'sample_id':entry['sample_id'],'split_name':split,'npz':entry['npz'],'sha256':record['npz_sha256'],'height':height,'width':width,'valid_pixels':height*width,'provenance':provenance})
        item=dict(record,source_image_id=source['source_image_id'],category=source['object'],angle_deg=float(level),split_name=split)
        if split=='test':index[source['source_image_id'],float(level)]=item
    if summary['npz_split_counts']!=dict(Counter(r['split_name'] for r in npz_records)) or summary['split_source_counts']!=dict(Counter(assignments.values())) or summary['compiled_samples']!=len(npz_records):raise ValueError('Actual compile counts differ')
    test_sources=[{'source_image_id':r['source_image_id'],'category':r['object'],'original_image_label':r['label']} for r in sources if r['split']=='test']
    if len(index)!=len(test_sources)*len(protocol['angle_levels_deg']) or not engineering and len(test_sources)!=802:raise ValueError('Complete authoritative source/level grid required')
    return {'root':root,'summary':summary,'protocol':protocol,'allocation':allocation,'records':records,'entries':entries,'npz_records':npz_records,'test_sources':test_sources,'index':index,'sources':source_by_id,'compile_summary_sha256':comp['summary_sha256'],'source_manifest_sha256':summary['source']['source_manifest_sha256']}


def _run_anchors(plan,base,compiled,engineering):
    budget=checked_json(HERE.parents[1]/'experiments/industrial-v1/training-plan-final.json',TRAINING_PLAN_SHA256)
    jobs=plan['runs'];grid={(mode,seed) for mode in CONDITIONS[:3] for seed in SEEDS}
    if len(jobs)!=9 or {(r['mode'],r['seed']) for r in jobs}!=grid:raise ValueError('All nine distinct E3 run bindings required')
    runs={};common=None;initial={};sampling={};hashed_sources={}
    for job in jobs:
        if set(job)!={'mode','seed','checkpoint_path','checkpoint_sha256','run_report_path','run_report_sha256','trainer_path'}:raise ValueError('Exact E3 run binding keys required')
        mode,seed=job['mode'],job['seed'];report=checked_json(_resolve(base,job['run_report_path']),job['run_report_sha256']);contract=report['contract'];total=contract['total_updates']
        if (type(seed) is not int or contract['seed']!=seed or contract['input_mode']!=mode or contract['allocation_sha256']!=compiled['summary']['allocation_sha256'] or contract['npz_records']!=compiled['npz_records'] or contract.get('map_scale_px')!=compiled['protocol']['map_scale_px']
            or contract['code_sha256']!=budget['expected_training_code_sha256'] or report.get('complete_budget') is not True or report.get('completed_update')!=total or report.get('requested_total_updates')!=total or report.get('resumed_from') is not None):raise ValueError('Actual run source/mode/seed/from-scratch/full-budget contract differs')
        trainer=_resolve(base,job['trainer_path'])
        for name,value in budget['expected_training_code_sha256'].items():
            path=trainer.parent/name
            if path not in hashed_sources:hashed_sources[path]=sha(path)
            if hashed_sources[path]!=value:raise ValueError('Pinned trainer/model code differs')
        if not engineering:
            cfg=budget['formal_gpu']
            fields={'total_updates':cfg['updates'],**{k:cfg[k] for k in ('num_res','batch_size','patch_size','validation_interval','checkpoint_interval','threads','device')}}
            if any(contract.get(k)!=v for k,v in fields.items()) or contract['optimizer']!={'name':'Adam','lr':cfg['learning_rate'],'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'foreach':False,'fused':False,'capturable':False,'amsgrad':False,'maximize':False} or report['environment'].get('cuda_executed') is not True:raise ValueError('Actual formal run differs from d61 fixed GPU budget')
        if (contract.get('float_schema')!='imda-float-xy0-v2' or contract.get('architecture')!='GAMDUNetPlus 9-channel from scratch' or contract.get('precision')!='float32'
            or contract.get('loss')!='final-resolution masked linear-RGB spatial L1 only; no clamp, FFT or multiscale loss' or contract.get('deterministic_algorithms') is not True or contract.get('TF32') is not False or contract.get('cudnn_benchmark') is not False or contract.get('sampling')!='NumPy PCG64 SeedSequence([seed,update,0x1ADA]), with replacement; stateless crops; workers=0'):raise ValueError('Frozen E3 numerical/training policy differs')
        normalized={k:v for k,v in contract.items() if k not in ('input_mode','seed')}
        if common is None:common=normalized
        elif normalized!=common:raise ValueError('Nine actual contracts do not share data/budget/model/optimizer')
        history=report['history']
        if type(total) is not int or total<1 or len(history)!=total or [r['update'] for r in history]!=list(range(1,total+1)):raise ValueError('Actual run history is not complete')
        valid=[r for r in history if 'valid' in r]
        if not valid:raise ValueError('Actual validation selection evidence missing')
        best=min(valid,key=lambda r:r['valid']['mae']);selected=report['best_model']
        if selected['state_key']!='best_model_state' or selected['selected_update']!=best['update'] or report['best_update']!=best['update'] or report['best_valid_mae']!=best['valid']['mae'] or selected['sha256']!=best['model_sha256']:raise ValueError('Actual best-state selection differs')
        filename=Path(job['checkpoint_path']).name
        if filename!=f'update-{total:08d}.pt' or report['checkpoint_files'].get(filename)!=job['checkpoint_sha256'] or Path(selected['checkpoint']).name!=filename:raise ValueError('Plan does not select the final full-budget embedded-best checkpoint')
        sh=[r['sampling_plan_sha256'] for r in history];ih=report['initial_parameter_sha256']
        if seed in initial and (initial[seed]!=ih or sampling[seed]!=sh):raise ValueError('Same-seed actual initialization/sampling differs')
        initial[seed]=ih;sampling[seed]=sh
        runs[mode,seed]={'checkpoint_sha256':job['checkpoint_sha256'],'run_report_sha256':job['run_report_sha256'],'best_model_sha256':selected['sha256'],'selected_update':selected['selected_update'],'mode':mode,'seed':seed,'allocation_sha256':contract['allocation_sha256'],'num_res':contract['num_res'],'total_updates':total}
    return runs


def _bank_anchors(plan,base,compiled):
    banks={};entries=compiled['entries'];categories=sorted({r['source']['category'] for r in compiled['records']})
    if set(plan['banks'])!=set(categories):raise ValueError('Exactly one plan bank per source category required')
    for category,item in plan['banks'].items():
        if set(item)!={'directory','selection_path','selection_sha256','fit_receipt_sha256','metadata_sha256'}:raise ValueError('Exact E3 bank bindings required')
        selection=checked_json(_resolve(base,item['selection_path']),item['selection_sha256']);directory=_resolve(base,item['directory'])
        receipt=checked_json(directory/'fit-receipt.json',item['fit_receipt_sha256']);metadata=checked_json(directory/'metadata.json',item['metadata_sha256'])
        if (receipt['schema']!='industrial-sharp-normal-bank-fit-v1' or receipt['category']!=category or receipt['compile_summary_sha256']!=compiled['compile_summary_sha256']
            or receipt['selection_sha256']!=item['selection_sha256'] or receipt['metadata_sha256']!=item['metadata_sha256'] or receipt['bank_sha256']!=metadata['bank_sha256']
            or receipt['detector_contract_sha256']!=metadata['contract_sha256'] or metadata['training_manifest_sha256']!=item['selection_sha256'] or receipt['industrial_results'] is not False):raise ValueError('Actual bank fit/metadata/source identities differ')
        if receipt['engineering_only'] and not plan['engineering_only']:raise ValueError('Engineering bank cannot enter formal plan')
        rows=[r for r in compiled['records'] if entries[r['sample_id']]['split_name']=='train' and r['source']['category']==category]
        rows.sort(key=lambda r:hashlib.sha256(canonical([selection['selection_seed'],category,entries[r['sample_id']]['source_image_id'],r['source']['image_decoded_pixel_sha256']]).encode()).hexdigest())
        if selection['source_limit'] not in (64,128) or not rows or any(r['source']['original_image_label']!='normal' for r in rows):raise ValueError('Known clean-normal source selection required')
        chosen=rows[:selection['source_limit']]
        expected=[{'source_image_id':entries[r['sample_id']]['source_image_id'],'sample_id':r['sample_id'],'npz_sha256':r['npz_sha256'],'source_decoded_sha256':r['source']['image_decoded_pixel_sha256']} for r in chosen]
        if selection['sources']!=expected or selection['allocation_sha256']!=compiled['summary']['allocation_sha256'] or selection['compile_summary_sha256']!=compiled['compile_summary_sha256']:raise ValueError('Bank selection is not the authoritative clean-normal train subset')
        if [{k:r[k] for k in expected[0]} for r in receipt['training_sources']]!=expected or [r['srgb_sha256'] for r in receipt['training_sources']]!=[r['srgb_sha256'] for r in metadata['training_records']]:raise ValueError('Bank actual source ledger differs')
        if hashlib.sha256(canonical(metadata['contract']).encode()).hexdigest()!=metadata['contract_sha256'] or any(metadata['contract']['config'].get(k)!=v for k,v in plan['detector'].get('config',{}).items()):raise ValueError('Bank detector configuration differs from base plan')
        banks[category]={'bank_identity':{'training_manifest_sha256':item['selection_sha256'],'bank_sha256':metadata['bank_sha256'],'metadata_sha256':item['metadata_sha256']},'detector_contract_sha256':metadata['contract_sha256']}
    return banks


def load_base_plan_anchor(plan_path,expected_sha,*,engineering=False):
    """External E3 plan + small JSONs only. Never open NPZ/model/native score payloads."""
    if plan_path is None or expected_sha is None:raise ValueError('External base E3 plan path and SHA are required')
    path=Path(plan_path).resolve();plan=checked_json(path,expected_sha)
    if set(plan)!={'schema','compiled','detector','banks','runs','engineering_only'} or plan['schema']!='industrial-evaluation-plan-v1' or type(plan['engineering_only']) is not bool:raise ValueError('Exact external E3 base plan required')
    if plan['engineering_only'] and not engineering:raise ValueError('Engineering plan requires explicit engineering-only boundary')
    compiled=_compiled_anchor(plan,path.parent,engineering);runs=_run_anchors(plan,path.parent,compiled,engineering);banks=_bank_anchors(plan,path.parent,compiled)
    return {'plan':plan,'plan_sha256':expected_sha,'plan_path':str(path),'compiled':compiled,'runs':runs,'banks':banks,'categories':sorted(banks),'engineering':engineering,
            'identity':{'base_plan_sha256':expected_sha,'compile_summary_sha256':compiled['compile_summary_sha256'],'allocation_sha256':compiled['summary']['allocation_sha256'],
                        'compile_records_sha256':compiled['summary']['compile_records_sha256'],'actual_protocol_sha256':compiled['summary']['protocol_sha256'],
                        'actual_source_manifest_sha256':compiled['source_manifest_sha256'],'training_plan_sha256':None if engineering else TRAINING_PLAN_SHA256,
                        'compiled_index_sha256':hashlib.sha256(canonical([compiled['index'][k] for k in sorted(compiled['index'])]).encode()).hexdigest(),
                        'scope':'metadata identities relative to externally pinned base plan; no NPZ payload/model weights/native scores re-read',
                        'engineering_only':engineering}}


def _check_execution_context(context,anchor,*,kind):
    if not CONTEXT_FIELDS<=context.keys():raise ValueError('Missing required E3 execution context fields')
    schema='industrial-durable-execution-v1' if kind=='main' else 'industrial-counterfactual-execution-v1'
    if context['schema']!=schema or context['stage']!='test' or context['plan_sha256']!=anchor['plan_sha256'] or context['code_sha256']!=E3_CODE_SHA256:raise ValueError('Execution context differs from externally pinned E3 plan/code')
    if context['device']!=anchor['plan']['detector'].get('device','cpu'):raise ValueError('Inference device differs from base plan')
    if any(not isinstance(context[k],str) or not context[k] for k in ('torch','numpy','scipy','device')) or any(type(context[k]) is not int or context[k]<1 for k in ('cpu_threads','cpu_interop_threads')):raise ValueError('Explicit inference environment identity required')
    limited=context['engineering_limit_per_group']
    if limited is not None and (type(limited) is not int or limited<1):raise ValueError('Invalid engineering limit')
    expected=bool(anchor['plan']['engineering_only'] or limited is not None)
    if type(context['engineering_only']) is not bool or context['engineering_only']!=expected or expected and not anchor['engineering']:raise ValueError('Engineering context cannot assert formal coverage')


def _bind_record_to_anchor(record,complete,anchor,context):
    key=record['source_image_id'],float(record['angle_deg']);expected=anchor['compiled']['index'].get(key)
    if expected is None:raise ValueError('Result source/level not in authoritative compiled index')
    if (record['sample_id']!=expected['sample_id'] or record['npz_sha256']!=expected['npz_sha256'] or record['category']!=expected['category']
        or record['original_image_label']!=expected['source']['original_image_label'] or record['split_name']!='test'
        or record.get('annotation')!=expected['annotation']):raise ValueError('Result sample/NPZ/label/annotation differs from compiled observation')
    width,height=expected['source_mapping']['crop_size_wh'];roi=expected['source_mapping']['original_roi_index_xyxy'];pixels=(roi[2]-roi[0])*(roi[3]-roi[1])
    if complete['score_crop_hw']!=[height,width] or record['pixels']['observed_pixels']!=pixels or record['pixels']['observed_positive_pixels']!=expected['annotation']['retained_defect_pixels']:raise ValueError('Result native crop/original ROI differs from authoritative mapping')
    restoration=record['restoration'];elements=width*height*3
    if restoration['valid_rgb_elements']!=elements or not math.isclose(restoration['mse'],restoration['sum_squared_error']/elements,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('Result MSE/common availability scope differs')
    bank=anchor['banks'][record['category']]
    if record['bank_identity']!=bank['bank_identity'] or record['detector_contract_sha256']!=bank['detector_contract_sha256']:raise ValueError('Result bank/detector differs from base plan')
    if record['branch'] in ('sharp','blur'):
        if record['seed'] is not None or record['restorer'] is not None:raise ValueError('Baseline cannot claim a restored run')
    else:
        mode='full_maps' if record['branch']=='full_maps_cf_rz90' else record['branch'];model=anchor['runs'][mode,record['seed']]
        expected_model=dict(model,inference_torch=context['torch'],inference_numpy=context['numpy'],inference_device=context['device'],inference_precision='FP32/TF32off/deterministic')
        if record['restorer']!=expected_model:raise ValueError('Result restorer differs from actual pinned run report')


def check_producer_contexts(main,cf,main_sha,anchor=None):
    """Bind actual reviewed E3 and CF1 context fields; no producer modules imported."""
    if anchor is None:raise ValueError('External base E3 plan anchor required')
    a,b=main['context'],cf['context']
    _check_execution_context(a,anchor,kind='main');_check_execution_context(b,anchor,kind='cf')
    if a.get('schema')!='industrial-durable-execution-v1' or b.get('schema')!='industrial-counterfactual-execution-v1':raise ValueError('Frozen E3/CF1 context schemas required')
    if a.get('code_sha256')!=E3_CODE_SHA256 or b.get('code_sha256')!=E3_CODE_SHA256:raise ValueError('E3 execution code identity differs')
    if any(b.get(k)!=v for k,v in a.items() if k!='schema'):raise ValueError('CF changed E3 plan/environment/threshold context')
    if (b.get('counterfactual_code_sha256')!=CF1_CODE_SHA256 or b.get('execution_e3_manifest_sha256')!=E3_MANIFEST_SHA256
            or b.get('addendum_sha256')!=ADDENDUM_SHA256 or b.get('base_main_summary_sha256')!=main_sha
            or b.get('correct_full_thresholds_sha256')!=a['threshold_sha256']):raise ValueError('CF1 producer/addendum/main identity differs')


def read_result_tree(summary_path,expected_sha,*,kind,base_plan,base_plan_sha,engineering=False):
    anchor=load_base_plan_anchor(base_plan,base_plan_sha,engineering=engineering)
    return _read_result_tree_schema_only(summary_path,expected_sha,kind=kind,levels=anchor['compiled']['protocol']['angle_levels_deg'],anchor=anchor)


def normalize_completed_results(main_summary,main_sha,cf_summary,cf_sha,source_manifest=None,protocol_path=None,addendum_path=None,*,base_plan=None,base_plan_sha=None,engineering=False):
    """Externally anchored E3/C3/run/bank metadata, then result records. No large payload read."""
    anchor=load_base_plan_anchor(base_plan,base_plan_sha,engineering=engineering)
    addendum=checked_json(addendum_path,ADDENDUM_SHA256)
    compiled=anchor['compiled'];protocol=compiled['protocol'];sources=compiled['test_sources'];levels=protocol['angle_levels_deg']
    if source_manifest is not None and sha(source_manifest)!=compiled['source_manifest_sha256']:raise ValueError('Caller source file differs from base-plan anchor')
    if protocol_path is not None and sha(protocol_path)!=compiled['summary']['protocol_sha256']:raise ValueError('Caller protocol file differs from base-plan anchor')
    if not engineering and (compiled['summary']['protocol_sha256']!=addendum['base_protocol_sha256'] or TRAINING_PLAN_SHA256!=addendum['training_plan_sha256']):raise ValueError('Registered protocol/training-plan identities were not verified')
    main,main_receipt,main_proofs=_read_result_tree_schema_only(main_summary,main_sha,kind='main',levels=levels,anchor=anchor)
    cf,cf_receipt,cf_proofs=_read_result_tree_schema_only(cf_summary,cf_sha,kind='cf',levels=levels,anchor=anchor)
    check_producer_contexts(main_receipt,cf_receipt,main_sha,anchor)
    threshold=main_receipt['context']['threshold_sha256']
    if cf_receipt['base_main_summary_sha256']!=main_sha or cf_receipt['correct_full_thresholds_sha256']!=threshold:raise ValueError('CF main/normal-valid thresholds differ')
    for row in cf:
        value=row['counterfactual']
        if value.get('base_main_summary_sha256')!=main_sha or value.get('correct_full_thresholds_sha256')!=threshold:raise ValueError('CF record main/threshold differs')
    identity=dict(anchor['identity'],main_summary_sha256=main_sha,cf_summary_sha256=cf_sha,addendum_sha256=ADDENDUM_SHA256,
        producer_manifests={'E3':E3_MANIFEST_SHA256,'CF1':CF1_MANIFEST_SHA256},
        main_record_proof_count=len(main_proofs),cf_record_proof_count=len(cf_proofs),normalized_restored_rows=len(main)+len(cf),source_count=len(sources),
        record_proofs_sha256=hashlib.sha256(canonical(main_proofs+cf_proofs).encode()).hexdigest(),
        native_score_arrays='not read or bootstrapped; upstream actual E3/CF execution remains separately audited')
    if engineering:return None,identity
    check_source_coverage(main_proofs,sources,levels,kind='main');check_source_coverage(cf_proofs,sources,levels,kind='cf')
    panel=prepare_panel(main+cf,sources,levels)
    return panel,identity


def read_json_line(line):
    # Same strict duplicate/nonfinite policy without creating temporary files.
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON field')
            result[key]=value
        return result
    return json.loads(line,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('main-summary','cf-summary','source-manifest','protocol','addendum'):p.add_argument('--'+name,type=Path)
    p.add_argument('--main-summary-sha256');p.add_argument('--cf-summary-sha256');p.add_argument('--base-plan',type=Path);p.add_argument('--base-plan-sha256');p.add_argument('--engineering',action='store_true');p.add_argument('--write',type=Path)
    a=p.parse_args(argv)
    required=(a.main_summary,a.cf_summary,a.addendum,a.main_summary_sha256,a.cf_summary_sha256,a.base_plan,a.base_plan_sha256)
    if not any(required) and a.write is None:
        print(json.dumps({'status':'awaiting_complete_E3_and_CF_results','statistics_performed':False,
                          'formal_experiment_complete':False,'replicates':REPLICATES,'source_clusters':802,
                          'addendum_sha256':ADDENDUM_SHA256},indent=2));return
    if not all(required):raise ValueError('All main/CF/source/protocol/addendum artifacts and external summary hashes required')
    if a.engineering and a.write is not None:raise ValueError('Engineering metadata audit cannot publish registered statistics')
    if a.write is not None and a.write.exists():raise FileExistsError('New statistics output directory required')
    panel,identity=normalize_completed_results(a.main_summary,a.main_summary_sha256,a.cf_summary,a.cf_summary_sha256,a.source_manifest,a.protocol,a.addendum,base_plan=a.base_plan,base_plan_sha=a.base_plan_sha256,engineering=a.engineering)
    if a.write is None:
        print(json.dumps({'status':'engineering_anchored_metadata_audit_only' if a.engineering else 'complete_anchored_input_preflight_only','statistics_performed':False,'formal_experiment_complete':False,'inputs':identity},indent=2));return
    result,replicates=paired_statistics(panel,return_replicates=True)
    if any(sha(HERE/name)!=value for name,value in LOADED_CODE_SHA256.items()):raise ValueError('Statistics implementation changed during execution')
    result.update(arithmetic_only=False,formal_result_coverage_verified=True,statistics_performed=True,
                  inputs=identity,code_sha256=LOADED_CODE_SHA256,
                  scientific_accept=False,formal_training_completion_claim='refer to separately verified E3 run/CF artifacts; no inference performed here')
    a.write.mkdir(parents=True,exist_ok=False)
    normalized=a.write/'normalized-records.jsonl'
    with normalized.open('x') as f:
        for row in panel['normalized_rows']:f.write(canonical(row)+'\n')
    identity['normalized_records_sha256']=sha(normalized)
    replicate_path=a.write/'bootstrap-replicates.npz'
    with replicate_path.open('xb') as f:np.savez_compressed(f,**replicates)
    identity['bootstrap_replicates_sha256']=sha(replicate_path)
    with (a.write/'statistics.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'statistics_written','path':str((a.write/'statistics.json').resolve()),'statistics_sha256':sha(a.write/'statistics.json')},indent=2))


if __name__=='__main__':main()
