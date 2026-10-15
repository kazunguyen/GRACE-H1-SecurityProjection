"""Portable, checksummed bundles for Kaggle training and Colab preparation."""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import json,os,shutil,tempfile,zipfile
from .io import resolve_root,file_hash,read_json,write_json

SCHEMA='grace_h1_bundle_v1'

def create_bundle(root,destination,kind='training_inputs',joern_home=None):
    root=resolve_root(root);destination=Path(destination);files=[]
    job=None;external={}
    if kind in ('training_inputs','cpu_inputs'):
        files+=list((root/'grace_h1').glob('*.py'))+list(root.glob('requirements*.txt'))
        files+=list((root/'configs').glob('*.json'))+list((root/'configs').glob('*.yaml'))+list((root/'configs').glob('*.txt'))
        files+=[root/'data'/'manifest.json']
        # The GPU job receives train/validation only; CPU preparation needs queries.
        for dataset in ('devign','reveal'):
            for split in (('train','validation') if kind=='training_inputs' else ('train','validation','test')):
                files+=[root/'data'/dataset/(split+'.jsonl')]
                folder=root/'cache'/'embeddings'/dataset/split
                files += [folder/name for name in ('vectors.npy','index.jsonl','manifest.json')]
        if kind=='cpu_inputs':
            graphs=[p for p in (root/'cache'/'graphs').rglob('*') if p.name in ('features.json','features.json.gz')]
            for name in ('legacy_candidates','candidates','projected'):
                files += [p for p in (root/'cache'/name).rglob('*') if p.is_file() and '.tmp' not in p.name]
            inventory=root/'cache'/'cpu_input_inventory.json'
            job={'source':{p.name:file_hash(p) for p in sorted((root/'grace_h1').glob('*.py'))},
                 'data_manifest_sha256':file_hash(root/'data'/'manifest.json'),
                 'experiment':read_json(root/'configs'/'experiment.json')}
            write_json(inventory,{'job':job,'graph_paths':[p.relative_to(root).as_posix() for p in sorted(graphs)]})
            files+=graphs+[inventory]
    elif kind in ('cpu_prepared','cpu_checkpoint'):
        from .pipeline import verify_frozen
        inventory=read_json(root/'cache'/'cpu_input_inventory.json');job=inventory['job'];base=set(inventory['graph_paths'])
        if job['data_manifest_sha256']!=file_hash(root/'data'/'manifest.json'): raise ValueError('CPU input dataset changed')
        if read_json(root/'configs'/'experiment.json')!=job['experiment']: raise ValueError('CPU experiment changed; recreate inputs')
        for name,h in job['source'].items():
            if file_hash(root/'grace_h1'/name)!=h: raise ValueError('CPU input source changed')
        for name in ('candidates','projected','graphs'):
            files += [p for p in (root/'cache'/name).rglob('*') if p.is_file() and '.tmp' not in p.name
                      and (name!='graphs' or (p.name in ('features.json','features.json.gz') and p.relative_to(root).as_posix() not in base))]
        for frozen in sorted((root/'runs').glob('*/*/*/*/freeze.json')):
            verify_frozen(root,frozen.parent)
            files += [frozen]+[root/p for p in read_json(frozen)['files'] if p.startswith('runs/')]
        if kind=='cpu_prepared':
            from .offload import require_prepared_runs
            require_prepared_runs(root,job['experiment'])
            files.append(root/'runs'/'run_selection.json')
    elif kind=='joern_runtime':
        home=Path(joern_home).expanduser().resolve() if joern_home else None
        if home is None or not (home/'frontends'/'c2cpg').is_dir(): raise ValueError('Provide the installed Linux --joern-home')
        required=['joern','joern-parse','joern-export','c2cpg.sh','.installation_root']
        runtime=[home/p for p in required]
        for name in ('bin','lib','conf','frontends/c2cpg'):
            runtime += [p for p in (home/name).rglob('*') if p.is_file()]
        versions=sorted(p.name for p in (home/'lib').glob('io.joern.joern-cli-*.jar'))
        if len(versions)!=1: raise ValueError('No unique Joern version jar')
        job={'joern_version':versions[0].removeprefix('io.joern.joern-cli-').removesuffix('.jar'),
             'java_major':21,'scope':'Linux C/C++ parse/export runtime; other language frontends excluded',
             'source_home':str(home),'upstream':'https://github.com/joernio/joern'}
        external={p:'tools/joern/joern-cli/'+p.relative_to(home).as_posix() for p in runtime};files+=runtime
    elif kind=='trained_heads':
        files=[p for p in (root/'artifacts'/'heads').rglob('*') if p.is_file() and '.tmp' not in p.name]
        selection=root/'artifacts'/'training_selection.json'
        if selection.exists(): files.append(selection)
        if not files: raise FileNotFoundError('Train at least one head before export')
    else: raise ValueError('Unsupported bundle kind')
    files=sorted(set(files));entries=[]
    for p in files:
        if not p.is_file(): raise FileNotFoundError(p)
        entries.append({'path':external.get(p) or p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p),
                        'mode':p.stat().st_mode & 0o777})
    manifest={'schema':SCHEMA,'kind':kind,'entries':entries,'no_credentials':True,'job':job}
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
        bundle.writestr('bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2))
        for p,item in zip(files,entries): bundle.write(p,item['path'])
    os.replace(temp,destination)
    return {'path':str(destination),'kind':kind,'files':len(files),'bytes':destination.stat().st_size,'sha256':file_hash(destination),'job':job}

def import_bundle(root,archive,expected_kind='trained_heads'):
    root=resolve_root(root);root.mkdir(parents=True,exist_ok=True);archive=Path(archive)
    copied=reused=0
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('bundle_manifest.json'))
        if manifest.get('schema')!=SCHEMA or manifest.get('kind')!=expected_kind: raise ValueError('Wrong bundle schema/kind')
        if expected_kind not in ('training_inputs','trained_heads','cpu_inputs','cpu_prepared','cpu_checkpoint','joern_runtime'):
            raise ValueError('Unsupported import kind')
        if expected_kind in ('cpu_prepared','cpu_checkpoint'):
            job=manifest['job']
            if job['data_manifest_sha256']!=file_hash(root/'data'/'manifest.json'): raise ValueError('CPU bundle dataset differs')
            for name,h in job['source'].items():
                if file_hash(root/'grace_h1'/name)!=h: raise ValueError('CPU bundle code differs; use the same implementation snapshot')
        entries=manifest['entries'];names=[e['path'] for e in entries]
        if len({n.casefold() for n in names})!=len(names): raise ValueError('Duplicate bundle paths')
        if sorted(bundle.namelist())!=sorted(names+['bundle_manifest.json']): raise ValueError('Unexpected archive members')
        for item in entries:
            name=item['path'];p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name: raise ValueError('Nonportable bundle path')
            if expected_kind=='trained_heads' and not name.startswith('artifacts/'): raise ValueError('Head bundle may only import artifacts')
            if expected_kind in ('cpu_prepared','cpu_checkpoint') and not name.startswith(('cache/graphs/','cache/candidates/','cache/projected/','runs/')):
                raise ValueError('Unexpected CPU output path')
            if expected_kind=='joern_runtime' and not name.startswith('tools/joern/joern-cli/'): raise ValueError('Unexpected runtime path')
            target=root/name;target.resolve().relative_to(root)
            if target.exists() and file_hash(target)!=item['sha256']:
                raise ValueError(f'Existing file differs; import to a new folder instead of overwriting {name}')
        # Verify every member before publishing any of them.
        with tempfile.TemporaryDirectory(prefix='grace-h1-import-',dir=root) as scratch:
            staging=Path(scratch)
            for item in entries:
                p=staging/item['path'];p.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(item['path']) as source,p.open('wb') as out: shutil.copyfileobj(source,out,1024*1024)
                if p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']: raise ValueError('Bundle checksum mismatch')
                if os.name!='nt': p.chmod(item.get('mode',0o644) & 0o777)
            for item in entries:
                target=root/item['path']
                if target.exists(): reused+=1;continue
                target.parent.mkdir(parents=True,exist_ok=True);os.replace(staging/item['path'],target);copied+=1
    return {'kind':expected_kind,'copied':copied,'reused_identical':reused,'archive_sha256':file_hash(archive)}
