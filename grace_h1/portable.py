"""Portable, checksummed bundles for optional Kaggle training."""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import json,os,shutil,tempfile,zipfile
from .io import resolve_root,file_hash,read_json,sha256

SCHEMA='grace_h1_bundle_v1'

def create_bundle(root,destination,kind='training_inputs'):
    root=resolve_root(root);destination=Path(destination);files=[]
    if kind=='training_inputs':
        files+=list((root/'grace_h1').glob('*.py'))+[root/'requirements.txt']
        files+=list((root/'configs').glob('*.json'))+list((root/'configs').glob('*.yaml'))+list((root/'configs').glob('*.txt'))
        files+=[root/'data'/'manifest.json']
        # Train/validation only: test code/labels are not sent to the GPU job.
        for dataset in ('devign','reveal'):
            for split in ('train','validation'):
                files+=[root/'data'/dataset/(split+'.jsonl')]
                folder=root/'cache'/'embeddings'/dataset/split
                files += [folder/name for name in ('vectors.npy','index.jsonl','manifest.json')]
    elif kind=='trained_heads':
        files=[p for p in (root/'artifacts'/'heads').rglob('*') if p.is_file() and '.tmp' not in p.name]
        selection=root/'artifacts'/'training_selection.json'
        if selection.exists(): files.append(selection)
        if not files: raise FileNotFoundError('Train at least one head before export')
    else: raise ValueError('Unsupported bundle kind')
    files=sorted(set(files));entries=[]
    for p in files:
        if not p.is_file(): raise FileNotFoundError(p)
        entries.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)})
    manifest={'schema':SCHEMA,'kind':kind,'entries':entries,'no_credentials':True}
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
        bundle.writestr('bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2))
        for p,item in zip(files,entries): bundle.write(p,item['path'])
    os.replace(temp,destination)
    return {'path':str(destination),'kind':kind,'files':len(files),'bytes':destination.stat().st_size,'sha256':file_hash(destination)}

def import_bundle(root,archive,expected_kind='trained_heads'):
    root=resolve_root(root);root.mkdir(parents=True,exist_ok=True);archive=Path(archive)
    copied=reused=0
    with zipfile.ZipFile(archive) as bundle:
        manifest=json.loads(bundle.read('bundle_manifest.json'))
        if manifest.get('schema')!=SCHEMA or manifest.get('kind')!=expected_kind: raise ValueError('Wrong bundle schema/kind')
        entries=manifest['entries'];names=[e['path'] for e in entries]
        if len({n.casefold() for n in names})!=len(names): raise ValueError('Duplicate bundle paths')
        if sorted(bundle.namelist())!=sorted(names+['bundle_manifest.json']): raise ValueError('Unexpected archive members')
        for item in entries:
            name=item['path'];p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name: raise ValueError('Nonportable bundle path')
            if expected_kind=='trained_heads' and not name.startswith('artifacts/'): raise ValueError('Head bundle may only import artifacts')
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
            for item in entries:
                target=root/item['path']
                if target.exists(): reused+=1;continue
                target.parent.mkdir(parents=True,exist_ok=True);os.replace(staging/item['path'],target);copied+=1
    return {'kind':expected_kind,'copied':copied,'reused_identical':reused,'archive_sha256':file_hash(archive)}
