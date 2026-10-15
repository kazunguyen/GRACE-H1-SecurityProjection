"""Copy portable inputs from prior runs. Never mutate a source cache."""
from pathlib import Path
import argparse, json, shutil, sys, gzip
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grace_h1.io import file_hash, sha256, fingerprint, row_identity, write_json, write_jsonl, iter_jsonl, read_json
ROOT=Path(__file__).resolve().parents[1]

def copy_file(source,dest,entries):
    source=Path(source);dest=Path(dest);digest=file_hash(source)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() or file_hash(dest)!=digest: shutil.copy2(source,dest)
    if file_hash(dest)!=digest: raise ValueError(f'Copy mismatch {source}')
    entries.append({'source':str(source),'destination':dest.relative_to(ROOT).as_posix(),'bytes':dest.stat().st_size,'sha256':digest})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--pooling',choices=['masked_mean','cls'],default='masked_mean')
    args=ap.parse_args();old=args.workspace.resolve()/'Graduation-Project'
    dataset_root=old/'Grace-with-filters-v2'/'[CONTRACT]'/'datasets'
    shared=old/'Ablation_V2'/'Reproduced-Code-Based'/'shared_cache'
    original=read_json(dataset_root/'dataset_manifest.json');entries=[];manifest={'schema':'grace_h1_data_v1','datasets':{}}
    import pyarrow.parquet as pq
    import numpy as np
    all_hashes=set();embedding_specs=[]
    for dataset in ('devign','reveal'):
        manifest['datasets'][dataset]={}
        for split in ('train','validation','test'):
            item=next(e for e in original['files'] if e['dataset']==dataset and e['split']==split)
            source=dataset_root/item['path']
            if file_hash(source)!=item['sha256']: raise ValueError('Pinned source parquet changed')
            parquet=ROOT/'data'/dataset/(split+'.parquet');copy_file(source,parquet,entries)
            rows=[]
            for n,raw in enumerate(pq.read_table(parquet).to_pylist()):
                code=str(raw['func' if dataset=='devign' else 'functionSource'])
                label=int(raw['target' if dataset=='devign' else 'label'])
                if label not in (0,1): raise ValueError('Non-binary source label')
                sid=str(raw['id']) if dataset=='devign' else f"{raw['hash']}#{n}"
                rows.append({'record_id':f'{dataset}:{split}:{sid}','dataset':dataset,'split':split,'source_id':sid,
                             'code':code,'code_sha256':sha256(code),'label':label,
                             'project':raw.get('project') or 'unknown','commit_id':raw.get('commit_id')})
            if len(rows)!=item['rows']: raise ValueError('Source row count mismatch')
            out=ROOT/'data'/dataset/(split+'.jsonl');write_jsonl(out,rows)
            rid=row_identity(rows);all_hashes.update(r['code_sha256'] for r in rows)
            manifest['datasets'][dataset][split]={'count':len(rows),'identity':rid,'rows_path':out.relative_to(ROOT).as_posix(),
                'rows_sha256':file_hash(out),'source_sha256':item['sha256'],'parquet':parquet.relative_to(ROOT).as_posix()}
            choices=[]
            for path in sorted((shared/'codet5').glob('*.json')):
                m=read_json(path);s=m.get('spec',{})
                if s.get('rows')==rid and s.get('pooling')==args.pooling and isinstance(s.get('checkpoint'),dict):
                    index=path.with_suffix('.index.jsonl');array=path.with_suffix('.npy')
                    if index.exists() and array.exists() and row_identity(list(iter_jsonl(index)))==rid:
                        if file_hash(array)!=m['sha256']: raise ValueError('Prior embedding bytes changed')
                        choices.append((path,m,index,array))
            if not choices: raise FileNotFoundError(f'No verified {args.pooling} embeddings for {dataset}/{split}; use encoder notebook/job first')
            path,m,index,array=choices[0];dest=ROOT/'cache'/'embeddings'/dataset/split
            copy_file(array,dest/'vectors.npy',entries);copy_file(index,dest/'index.jsonl',entries)
            a=np.load(dest/'vectors.npy',mmap_mode='r',allow_pickle=False)
            if a.dtype!=np.float32 or list(a.shape)!=m['shape']: raise ValueError('Embedding shape/dtype mismatch')
            spec={k:m['spec'][k] for k in ('architecture','checkpoint','pooling','max_length','seed')}
            embedding_specs.append(spec)
            write_json(dest/'manifest.json',{'schema':'grace_h1_embedding_v1','rows_identity':rid,'shape':m['shape'],
                'encoder_spec':spec,'source_manifest':m,'source_manifest_sha256':file_hash(path),
                'files':{name:file_hash(dest/name) for name in ('vectors.npy','index.jsonl')}})
            print(f'copied {dataset}/{split}: {len(rows)} rows, {a.shape}',flush=True)
    if any(s!=embedding_specs[0] for s in embedding_specs): raise ValueError('Mixed encoder recipes')
    write_json(ROOT/'data'/'manifest.json',manifest)
    copy_file(old/'STANDARD_FOR_ALL_INFERENCES'/'evaluation_contract.yaml',ROOT/'configs'/'reference_contract.yaml',entries)
    copy_file(old/'Ablation_V2'/'Devign'/'Contract'/'prompt_template.txt',ROOT/'configs'/'prompt_template.txt',entries)
    copy_file(dataset_root/'dataset_manifest.json',ROOT/'provenance'/'source_dataset_manifest.json',entries)
    graph_count=0;graph_bytes=0
    for h in sorted(all_hashes):
        card=shared/'joern'/h
        source=card/'features.json.gz'
        if not source.exists(): source=card/'features.json'
        if source.exists():
            raw=gzip.decompress(source.read_bytes()) if source.suffix=='.gz' else source.read_bytes()
            g=json.loads(raw);expected=g.pop('feature_sha256',None)
            if g.get('code_sha256')!=h or fingerprint(g)!=expected: raise ValueError(f'Invalid graph feature {h}')
            dest=ROOT/'cache'/'graphs'/h/source.name;copy_file(source,dest,entries)
            graph_count+=1;graph_bytes+=source.stat().st_size
            if graph_count%2000==0: print(f'copied {graph_count} graph features',flush=True)
    for source in sorted((shared/'candidates').glob('*.jsonl')):
        copy_file(source,ROOT/'cache'/'legacy_candidates'/source.name,entries)
    # Keep raw response/prompt records as references; inference imports them only
    # after exact prompt + model + decoding identity checks, never just record ID.
    for folder in ('Reproduced-Code-Based','Reproduced-Paper-Based'):
        for dataset in ('devign','reveal'):
            for split in ('validation','test'):
                base=old/'Ablation_V2'/folder/dataset/split
                for name in ('run_manifest.json','inference/raw_responses.jsonl','inference/raw_responses.checkpoint.jsonl'):
                    source=base/name
                    if source.is_file(): copy_file(source,ROOT/'cache'/'prior_runs'/folder/dataset/split/name,entries)
    report={'schema':'grace_h1_import_v1','source_workspace':str(args.workspace.resolve()),'pooling':args.pooling,
            'graph_feature_count':graph_count,'graph_feature_bytes':graph_bytes,'required_code_hashes':len(all_hashes),
            'copied_files':len(entries),'copied_bytes':sum(x['bytes'] for x in entries),'entries':entries,
            'copied_raw_cpg_binaries':False,'note':'Reusing exported graph features; missing graphs are computed by the full pipeline.'}
    write_json(ROOT/'cache'/'import_manifest.json',report)
    summary={k:v for k,v in report.items() if k!='entries'}
    summary['file_index']='cache/import_manifest.json'
    summary['file_index_sha256']=file_hash(ROOT/'cache'/'import_manifest.json')
    write_json(ROOT/'provenance'/'cache_import.json',summary)
    print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
if __name__=='__main__': main()
