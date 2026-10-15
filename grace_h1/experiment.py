from __future__ import annotations
from pathlib import Path
import argparse,json
from .io import resolve_root,read_json,write_json,fingerprint
from .projection import TrainingConfig,train_projection
from .pipeline import PipelineConfig,prepare_run,evaluate_run,compare_runs
from .inference import plan_inference,run_inference

def load_config(root,path=None):
    root=resolve_root(root);return read_json(path or root/'configs'/'experiment.json')

def train_all(root,config):
    root=resolve_root(root);heads={}
    for dataset in config['datasets']:
        heads[dataset]={}
        for objective in config['objectives']:
            heads[dataset][objective]=train_projection(root,dataset,TrainingConfig(objective=objective,**config['training']))
    selected={'training_config_hash':fingerprint(config['training']),'heads':heads}
    write_json(root/'artifacts'/'training_selection.json',selected);return selected

def prepare_all(root,config,checkpoint_callback=None):
    root=resolve_root(root);selected=read_json(root/'artifacts'/'training_selection.json')
    if selected['training_config_hash']!=fingerprint(config['training']): raise ValueError('Training config changed; run train first')
    runs=[]
    for dataset in config['datasets']:
        for split in config['splits']:
            for arm in config['llm_arms']:
                head=None if arm=='baseline_whitening' else selected['heads'][dataset][arm.removeprefix('h1_')]
                path=prepare_run(root,PipelineConfig(dataset=dataset,split=split,arm=arm,**config['pipeline']),head,checkpoint_callback)
                runs.append({'dataset':dataset,'split':split,'arm':arm,'path':path.relative_to(root).as_posix()})
    write_json(root/'runs'/'run_selection.json',{'experiment_config':config,'runs':runs});return runs

def selected_runs(root,config):
    root=resolve_root(root);selection=read_json(root/'runs'/'run_selection.json')
    if selection['experiment_config']!=config: raise ValueError('Experiment config changed; prepare matching runs first')
    return selection['runs']

def preflight_all(root,config):
    root=resolve_root(root)
    return [r|plan_inference(root,root/r['path']) for r in selected_runs(root,config)]

def infer_all(root,config,allow_api=False):
    root=resolve_root(root);results=[]
    for row in selected_runs(root,config):
        report=run_inference(root,root/row['path'],allow_api=allow_api,workers=config['inference_workers'])
        results.append(row|report)
    write_json(root/'runs'/'inference_experiment.json',results);return results

def evaluate_all(root,config):
    root=resolve_root(root);runs=selected_runs(root,config);scores=[];comparisons=[]
    for row in runs:
        path=root/row['path']
        if not (path/'predictions.jsonl').exists(): continue
        metrics=evaluate_run(root,path);inference=read_json(path/'inference_summary.json')
        scores.append(row|{'metrics':metrics,'inference':inference})
    by_key={(r['dataset'],r['split'],r['arm']):r for r in scores}
    for row in scores:
        if row['arm']=='baseline_whitening': continue
        baseline=by_key.get((row['dataset'],row['split'],'baseline_whitening'))
        if baseline: comparisons.append(compare_runs(root,root/baseline['path'],root/row['path'],bootstrap=config['bootstrap']))
    complete=len(scores)==len(runs) and all(r['inference']['status']=='complete' for r in scores)
    report={'status':'complete' if complete else 'partial','experiment':config,'scores':scores,'comparisons':comparisons,
            'note':'No claim of improved GRACE until paired quality/cost is measured.'}
    write_json(root/'runs'/'experiment_summary.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','prepare','preflight','infer','evaluate','all'])
    p.add_argument('--root',type=Path);p.add_argument('--config',type=Path);p.add_argument('--allow-api',action='store_true')
    args=p.parse_args();root=resolve_root(args.root);config=load_config(root,args.config)
    result=None
    if args.stage in ('train','all'): result=train_all(root,config)
    if args.stage in ('prepare','all'): result=prepare_all(root,config)
    if args.stage in ('preflight','all'): result=preflight_all(root,config)
    if args.stage in ('infer','all'): result=infer_all(root,config,args.allow_api)
    if args.stage in ('evaluate','all'): result=evaluate_all(root,config)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
