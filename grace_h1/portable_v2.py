"""Checksummed V2 handoffs with isolated mutable selection pointers."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import json
import argparse
import os
import shutil
import tempfile
import zipfile
from .io import resolve_root, file_hash, read_json, write_json, fingerprint
from .experiment_v2 import load_config, validate_config, training_identity, require_heads, require_runs, RUN_SELECTION, TRAIN_SELECTION
from .pipeline import verify_frozen

SCHEMA='grace_projection_v2_bundle_v1'
KINDS=('training_inputs','cpu_inputs','heads','training_checkpoint','prepared','checkpoint')


def create_bundle(root,destination,kind):
    root=resolve_root(root);destination=Path(destination);cfg=load_config(root);validate_config(cfg)
    if kind not in KINDS: raise ValueError('Unknown V2 bundle kind')
    files=[]
    source={p.name:file_hash(p) for p in (root/'grace_h1').glob('*.py')}
    job={'workflow':'projection_v2','experiment':cfg,'source':source,
         'data_manifest_sha256':file_hash(root/'data/manifest.json'),'training_config_hash':training_identity(cfg)}
    if kind in ('training_inputs','cpu_inputs'):
        files+=list((root/'grace_h1').glob('*.py'))+list(root.glob('requirements*.txt'))
        files+=[root/'configs'/name for name in ('projection_v2.json','reference_contract.yaml','prompt_template.txt')]
        files+=[root/'data/manifest.json']
        splits=('train','validation') if kind=='training_inputs' else tuple(dict.fromkeys(['train','validation',*cfg['splits']]))
        for dataset in cfg['datasets']:
            for split in splits:
                files.append(root/'data'/dataset/(split+'.jsonl'))
                folder=root/'cache/embeddings'/dataset/split
                files += [folder/name for name in ('vectors.npy','index.jsonl','manifest.json')]
        if kind=='cpu_inputs':
            graphs=[p for p in (root/'cache/graphs').rglob('*') if p.name in ('features.json','features.json.gz')]
            for name in ('legacy_candidates','candidates','projected'):
                files += [p for p in (root/'cache'/name).rglob('*') if p.is_file() and '.tmp' not in p.name]
            inventory=root/'cache/projection_v2/cpu_inventory.json'
            write_json(inventory,{'job':job,'graph_paths':[p.relative_to(root).as_posix() for p in graphs]})
            files+=graphs+[inventory]
        else:
            # Reuse label-blind validation candidate caches; never ship test cohorts.
            for directory in ('candidates','legacy_candidates'):
                for p in (root/'cache'/directory).glob('*.jsonl'):
                    with p.open(encoding='utf-8') as f:
                        first=next((line for line in f if line.strip()),None)
                    if first and any(json.loads(first)['record_id'].startswith(d+':validation:') for d in cfg['datasets']):
                        files.append(p)
    elif kind in ('heads','training_checkpoint'):
        if kind=='heads': require_heads(root,cfg)
        if (root/TRAIN_SELECTION).exists(): files.append(root/TRAIN_SELECTION)
        for dataset in cfg['datasets']:
            for objective in cfg['objectives']:
                files += [p for p in (root/'artifacts/projection_v2'/dataset/objective).rglob('*') if p.is_file() and '.tmp' not in p.name]
        train_identities={read_json(root/'cache/embeddings'/d/'train/manifest.json')['rows_identity'] for d in cfg['datasets']}
        for p in (root/'cache/projection_v2').glob('*.json'):
            meta=read_json(p)
            if meta.get('train_identity') in train_identities:
                files += [p,p.with_suffix('.npz')]
        if (root/'artifacts/projection_v2/replications.json').exists(): files.append(root/'artifacts/projection_v2/replications.json')
    else:
        inventory=read_json(root/'cache/projection_v2/cpu_inventory.json');job=inventory['job']
        if job['experiment']!=cfg or job['data_manifest_sha256']!=file_hash(root/'data/manifest.json'):
            raise ValueError('CPU input config/data changed; recreate V2 inputs')
        for name,h in job['source'].items():
            if file_hash(root/'grace_h1'/name)!=h: raise ValueError('CPU input source changed')
        base=set(inventory['graph_paths'])
        files += [p for p in (root/'cache/graphs').rglob('*') if p.name in ('features.json','features.json.gz') and p.relative_to(root).as_posix() not in base]
        for directory in ('candidates','projected'):
            files += [p for p in (root/'cache'/directory).rglob('*') if p.is_file() and '.tmp' not in p.name]
        rows=require_runs(root,cfg) if kind=='prepared' else (read_json(root/RUN_SELECTION)['runs'] if (root/RUN_SELECTION).exists() else [])
        for row in rows:
            folder=root/row['path'];verify_frozen(root,folder)
            files+=[folder/'freeze.json']+[root/name for name in read_json(folder/'freeze.json')['files'] if name.startswith('runs/')]
        if (root/RUN_SELECTION).exists(): files.append(root/RUN_SELECTION)
    files=sorted(set(files));entries=[]
    for p in files:
        if not p.is_file(): raise FileNotFoundError(p)
        entries.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)})
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.tmp.zip')
    manifest={'schema':SCHEMA,'kind':kind,'entries':entries,'job':job,'no_credentials':True}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.writestr('bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2))
        for path,item in zip(files,entries): archive.write(path,item['path'])
    os.replace(temporary,destination)
    return {'path':str(destination),'kind':kind,'files':len(files),'bytes':destination.stat().st_size,'sha256':file_hash(destination)}


def import_bundle(root,archive,expected_kind):
    root=resolve_root(root);root.mkdir(parents=True,exist_ok=True);archive=Path(archive)
    pointers={TRAIN_SELECTION.as_posix(),RUN_SELECTION.as_posix()}
    def replaceable(name):
        if name in pointers: return True
        p=root/name
        return (expected_kind in ('heads','training_checkpoint') and name.startswith('artifacts/projection_v2/')
                and p.name in ('last.pt','history.json','best.json','projection.npz')
                and not (p.parent/'manifest.json').exists())
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('bundle_manifest.json'))
        if expected_kind not in KINDS or manifest.get('schema')!=SCHEMA or manifest.get('kind')!=expected_kind:
            raise ValueError('Wrong V2 bundle kind/schema')
        job=manifest['job']
        if expected_kind in ('heads','training_checkpoint','prepared','checkpoint'):
            cfg=load_config(root)
            if job['training_config_hash']!=training_identity(cfg) or job['data_manifest_sha256']!=file_hash(root/'data/manifest.json'):
                raise ValueError('Output training config/data differs')
            if expected_kind in ('prepared','checkpoint') and job['experiment']!=cfg:
                raise ValueError('CPU output experiment config differs')
            for name,digest in job['source'].items():
                if Path(name).name!=name or file_hash(root/'grace_h1'/name)!=digest:
                    raise ValueError('Output source differs; use identical branch snapshot')
        entries=manifest['entries'];names=[e['path'] for e in entries]
        if len({n.casefold() for n in names})!=len(names) or sorted(bundle.namelist())!=sorted(names+['bundle_manifest.json']):
            raise ValueError('Duplicate/unexpected archive members')
        for item in entries:
            name=item['path'];p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe V2 archive path')
            if expected_kind in ('heads','training_checkpoint') and not name.startswith(('artifacts/projection_v2/','cache/projection_v2/')):
                raise ValueError('Unexpected head output path')
            if expected_kind in ('training_inputs','cpu_inputs') and not name.startswith(('grace_h1/','configs/','data/','cache/','requirements')):
                raise ValueError('Unexpected input path')
            if expected_kind=='training_inputs' and ('/test/' in name or name.endswith('/test.jsonl') or name.startswith('cache/responses/')):
                raise ValueError('Training bundle may not import test data/responses')
            if expected_kind in ('prepared','checkpoint') and not name.startswith(('cache/graphs/','cache/candidates/','cache/projected/','runs/')):
                raise ValueError('Unexpected CPU output path')
            target=root/name;target.resolve().relative_to(root)
            if target.exists() and file_hash(target)!=item['sha256'] and not replaceable(name):
                raise ValueError(f'Existing immutable file differs: {name}')
        copied=reused=updated=0
        with tempfile.TemporaryDirectory(prefix='projection-v2-import-',dir=root) as scratch:
            staging=Path(scratch)
            # Validate every byte before updating even a mutable selection pointer.
            for item in entries:
                p=staging/item['path'];p.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(item['path']) as source,p.open('wb') as target:
                    shutil.copyfileobj(source,target,1024*1024)
                if p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:
                    raise ValueError('V2 archive checksum mismatch')
            # Publish immutable payload first, then selection pointers.
            for item in sorted(entries,key=lambda e:(2 if e['path'] in pointers else 1 if e['path'].endswith('/manifest.json') else 0)):
                target=root/item['path']
                if target.exists() and file_hash(target)==item['sha256']: reused+=1;continue
                existed=target.exists();target.parent.mkdir(parents=True,exist_ok=True)
                os.replace(staging/item['path'],target)
                if existed: updated+=1
                else: copied+=1
    return {'kind':expected_kind,'copied':copied,'reused_identical':reused,'updated_selection_pointers':updated,'archive_sha256':file_hash(archive)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['create','import'])
    parser.add_argument('kind',choices=KINDS)
    parser.add_argument('--path',required=True,type=Path)
    parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    operation=create_bundle if args.action=='create' else import_bundle
    print(operation(resolve_root(args.root),args.path,args.kind))


if __name__=='__main__': main()
