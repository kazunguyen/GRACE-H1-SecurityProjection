"""Isolated V2 training/selection and full GRACE workflow over shared caches."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
from .io import read_json, write_json, resolve_root, fingerprint, file_hash
from .projection_v2 import ProjectionV2Config, train_projection_v2, project_embeddings_v2
from .pipeline import PipelineConfig, prepare_run, verify_frozen, evaluate_run, compare_runs
from .data import load_training_data, embedding_recipe
from .inference import plan_inference, run_inference, retry_exhausted

TRAIN_SELECTION = Path('artifacts/projection_v2/training_selection.json')
RUN_SELECTION = Path('runs/projection_v2/run_selection.json')


def load_config(root, path=None):
    return read_json(path or resolve_root(root)/'configs/projection_v2.json')


def validate_config(config):
    if not config['datasets'] or any(d not in ('devign','reveal') for d in config['datasets']):
        raise ValueError('Choose devign/reveal datasets')
    if not config['splits'] or any(s not in ('validation','test') for s in config['splits']):
        raise ValueError('Choose validation/test inference splits')
    if 'test' in config['splits'] and not config.get('test_protocol_frozen',False):
        raise ValueError('Freeze the protocol using validation before enabling test_protocol_frozen')
    if len(set(config['objectives'])) != len(config['objectives']) or not config['objectives']:
        raise ValueError('Unique training objectives required')
    for objective in config['objectives']:
        ProjectionV2Config(objective=objective,**config['training']).validate()
    valid_arms={'baseline_whitening'}|{'h1_'+o for o in config['objectives']}
    if len(set(config['llm_arms']))!=len(config['llm_arms']) or not set(config['llm_arms'])<=valid_arms:
        raise ValueError('Untrained/duplicate V2 LLM arm')
    if 'baseline_whitening' not in config['llm_arms']:
        raise ValueError('Keep the paired whitening control')
    if config['pipeline']['whitening_dimension']!=config['training']['dimension']:
        raise ValueError('Use identical teacher/baseline dimension')


def training_identity(config):
    # Inference split/workers must not affect trained head reuse.
    return fingerprint({k:config[k] for k in ('datasets','objectives','training')})


def train_all(root, config):
    root=resolve_root(root);validate_config(config)
    heads={}
    for dataset in config['datasets']:
        heads[dataset]={}
        for objective in config['objectives']:
            heads[dataset][objective]=train_projection_v2(root,dataset,ProjectionV2Config(objective=objective,**config['training']))
    selected={'schema':'grace_projection_v2_selection','training_config_hash':training_identity(config),'heads':heads}
    write_json(root/TRAIN_SELECTION,selected)
    return selected


def require_heads(root,config):
    root=resolve_root(root);validate_config(config)
    selected=read_json(root/TRAIN_SELECTION)
    if selected['training_config_hash']!=training_identity(config):
        raise ValueError('V2 training config changed; train/export matching heads')
    for dataset in config['datasets']:
        _,_,x,_,tm,vm,_=load_training_data(root,dataset)
        data=read_json(root/'data/manifest.json')['datasets'][dataset]
        supervised_rows={s:data[s]['rows_sha256'] for s in ('train','validation')}
        for objective in config['objectives']:
            h=selected['heads'][dataset][objective]
            expected=asdict(ProjectionV2Config(objective=objective,**config['training']))
            if h['config']!=expected or h['train_identity']!=tm['rows_identity'] or h['validation_identity']!=vm['rows_identity'] or h['encoder_recipe']!=embedding_recipe(tm):
                raise ValueError('V2 head dataset/recipe/config differs')
            if h['supervised_rows_sha256']!=supervised_rows:
                raise ValueError('V2 train/validation code or labels changed')
            for name,digest in h['source'].items():
                if file_hash(root/'grace_h1'/name)!=digest:
                    raise ValueError('V2 training source differs; use the exported snapshot')
            project_embeddings_v2(x[:1],h,root)
    return selected


def require_runs(root,config):
    root=resolve_root(root);heads=require_heads(root,config)
    selected=read_json(root/RUN_SELECTION)
    if selected['experiment_config']!=config:
        raise ValueError('V2 preparation config differs')
    rows=selected['runs']
    expected={(d,s,a) for d in config['datasets'] for s in config['splits'] for a in config['llm_arms']}
    if len(rows)!=len(expected) or {(r['dataset'],r['split'],r['arm']) for r in rows}!=expected:
        raise ValueError('Missing V2 prepared arms/cohort')
    for row in rows:
        path=root/row['path'];path.resolve().relative_to(root/'runs')
        manifest=verify_frozen(root,path)
        if manifest['config']!={'dataset':row['dataset'],'split':row['split'],'arm':row['arm'],**config['pipeline']}:
            raise ValueError('V2 pipeline controls differ')
        inputs=manifest['inputs']
        if inputs['data_manifest']!=file_hash(root/'data/manifest.json'):
            raise ValueError('V2 prepared data differs')
        for name,digest in inputs['source'].items():
            if file_hash(root/'grace_h1'/name)!=digest:
                raise ValueError('V2 CPU preparation source differs')
        if inputs['template']!=file_hash(root/'configs/prompt_template.txt') or inputs['contract']!=file_hash(root/'configs/reference_contract.yaml'):
            raise ValueError('V2 prompt or contract changed')
        if row['arm']!='baseline_whitening':
            h=heads['heads'][row['dataset']][row['arm'].removeprefix('h1_')]
            if inputs['head']!=h['projection_sha256']:
                raise ValueError('V2 prepared head differs')
    return rows


def prepare_all(root,config,checkpoint_callback=None):
    root=resolve_root(root);heads=require_heads(root,config)
    for dataset in config['datasets']:
        for arm in config['llm_arms']:
            if arm!='baseline_whitening' and heads['heads'][dataset][arm.removeprefix('h1_')]['selection_status']=='baseline_fallback' and not config.get('allow_identity_fallback_run',False):
                raise ValueError(f'{dataset}/{arm}: no validation proxy gain; keep tuning offline or explicitly enable the identity control')
    runs=[]
    write_json(root/RUN_SELECTION,{'experiment_config':config,'runs':runs})
    for dataset in config['datasets']:
        for split in config['splits']:
            for arm in config['llm_arms']:
                h=None if arm=='baseline_whitening' else heads['heads'][dataset][arm.removeprefix('h1_')]
                if h and h['selection_status']=='baseline_fallback' and not config.get('allow_identity_fallback_run',False):
                    raise ValueError(f'{dataset}/{arm}: no validation proxy gain; keep tuning offline or explicitly enable the identity control')
                path=prepare_run(root,PipelineConfig(dataset=dataset,split=split,arm=arm,**config['pipeline']),h,checkpoint_callback)
                runs.append({'dataset':dataset,'split':split,'arm':arm,'path':path.relative_to(root).as_posix()})
                write_json(root/RUN_SELECTION,{'experiment_config':config,'runs':runs})
    write_json(root/RUN_SELECTION,{'experiment_config':config,'runs':runs})
    return runs


def preflight_all(root,config,*,force_fresh=False):
    root=resolve_root(root)
    return [row|plan_inference(root,root/row['path'],force_fresh=force_fresh) for row in require_runs(root,config)]


def infer_all(root,config,allow_api=False,*,force_fresh=False):
    root=resolve_root(root)
    reports=[row|run_inference(root,root/row['path'],allow_api=allow_api,workers=config['inference_workers'],force_fresh=force_fresh)
             for row in require_runs(root,config)]
    write_json(root/'runs/projection_v2/inference_experiment.json',reports)
    return reports


def retry_exhausted_all(root,config,*,workers=None):
    """Retry only exhausted requests for the currently selected frozen runs."""
    root=resolve_root(root)
    aggregate=root/'runs/projection_v2/inference_experiment.json'
    archive=root/'runs/projection_v2/inference_experiment_before_retry.json'
    if aggregate.exists() and not archive.exists(): write_json(archive,read_json(aggregate))
    reports=[]
    worker_count=workers or config['inference_workers']
    for row in require_runs(root,config):
        print(f"Retry exhausted only: {row['dataset']}/{row['split']}/{row['arm']} with {worker_count} workers",flush=True)
        report=retry_exhausted(root,root/row['path'],allow_api=True,workers=worker_count)
        reports.append(row|report)
    write_json(root/'runs/projection_v2/inference_experiment.json',reports)
    return reports


def evaluate_all(root,config):
    root=resolve_root(root);rows=require_runs(root,config);scores=[];comparisons=[]
    for row in rows:
        path=root/row['path']
        if not (path/'predictions.jsonl').exists(): continue
        scores.append(row|{'metrics':evaluate_run(root,path),'inference':read_json(path/'inference_summary.json')})
    baseline={(row['dataset'],row['split']):row for row in scores if row['arm']=='baseline_whitening'}
    for row in scores:
        if row['arm']!='baseline_whitening' and (row['dataset'],row['split']) in baseline:
            left=baseline[(row['dataset'],row['split'])]
            comparisons.append(compare_runs(root,root/left['path'],root/row['path'],config['bootstrap']))
    report={'status':'complete' if len(scores)==len(rows) and all(r['inference']['status']=='complete' for r in scores) else 'partial',
            'experiment':config,'scores':scores,'comparisons':comparisons,
            'note':'Validation retrieval proxy gain is not a full GRACE quality claim.'}
    write_json(root/'runs/projection_v2/experiment_summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','prepare','preflight','infer','evaluate'])
    parser.add_argument('--root',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args();root=resolve_root(args.root);config=load_config(root,args.config)
    if args.stage=='infer': result=infer_all(root,config,args.allow_api)
    else: result={'train':train_all,'prepare':prepare_all,'preflight':preflight_all,'evaluate':evaluate_all}[args.stage](root,config)
    print(result)


if __name__=='__main__': main()
