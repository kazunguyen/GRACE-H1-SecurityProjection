"""Validate handoffs without starting training, graph extraction or an API call."""
from dataclasses import asdict
from pathlib import Path
from .io import resolve_root,read_json,file_hash,fingerprint,row_identity
from .data import load_rows,embedding_recipe
from .projection import TrainingConfig

def require_trained_heads(root,config):
    root=resolve_root(root)
    selected=read_json(root/'artifacts'/'training_selection.json')
    if selected['training_config_hash']!=fingerprint(config['training']): raise ValueError('Head training config differs; rerun Kaggle')
    sources={name:file_hash(root/'grace_h1'/name) for name in ('projection.py','data.py','metrics.py')}
    for dataset in config['datasets']:
        train=load_rows(root,dataset,'train',labels=True);valid=load_rows(root,dataset,'validation',labels=True)
        tm=read_json(root/'cache'/'embeddings'/dataset/'train'/'manifest.json')
        vm=read_json(root/'cache'/'embeddings'/dataset/'validation'/'manifest.json')
        groups={}
        for i,row in enumerate(train): groups.setdefault(row['code_sha256'],[]).append(i)
        ti=[ids[0] for ids in groups.values() if len({train[i]['label'] for i in ids})==1]
        hashes=set(groups);vi=[i for i,row in enumerate(valid) if row['code_sha256'] not in hashes]
        if tm['rows_identity']!=row_identity(train) or vm['rows_identity']!=row_identity(valid): raise ValueError('Head dataset identity differs')
        for objective in config['objectives']:
            head=selected['heads'][dataset][objective];cfg=TrainingConfig(objective=objective,**config['training']);cfg.validate()
            expected=fingerprint({'dataset':dataset,'config':asdict(cfg),'train':tm['rows_identity'],'validation':vm['rows_identity'],
                'recipe':embedding_recipe(tm),'data_manifest':file_hash(root/'data'/'manifest.json'),
                'training_indices':ti,'selection_indices':vi,'source':sources})
            if head['training_fingerprint']!=expected: raise ValueError('Head fingerprint differs; export matching Kaggle heads')
            path=root/head['path'];path.resolve().relative_to(root/'artifacts'/'heads')
            if file_hash(path)!=head['projection_sha256']: raise ValueError('Head weights checksum differs')
    return selected

def require_prepared_runs(root,config):
    from .experiment import selected_runs
    from .pipeline import verify_frozen
    root=resolve_root(root);runs=selected_runs(root,config)
    expected={(d,s,a) for d in config['datasets'] for s in config['splits'] for a in config['llm_arms']}
    if len(runs)!=len(expected) or {(r['dataset'],r['split'],r['arm']) for r in runs}!=expected: raise ValueError('Prepared cohort/arms incomplete')
    heads=require_trained_heads(root,config)
    for row in runs:
        path=root/row['path'];path.resolve().relative_to(root/'runs')
        manifest=verify_frozen(root,path);inputs=manifest['inputs']
        if inputs['data_manifest']!=file_hash(root/'data'/'manifest.json'): raise ValueError('Prepared dataset differs')
        for name,h in inputs['source'].items():
            if file_hash(root/'grace_h1'/name)!=h: raise ValueError('Prepared code differs; recreate Colab inputs')
        cfg={'dataset':row['dataset'],'split':row['split'],'arm':row['arm'],**config['pipeline']}
        if manifest['config']!=cfg: raise ValueError('Prepared pipeline config differs')
        if row['arm']!='baseline_whitening':
            head=heads['heads'][row['dataset']][row['arm'].removeprefix('h1_')]
            if inputs['head']!=head['projection_sha256']: raise ValueError('Prepared head differs')
    return runs
