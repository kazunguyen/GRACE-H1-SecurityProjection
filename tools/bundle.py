from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grace_h1.portable import create_bundle,import_bundle
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('action',choices=['training-inputs','trained-heads','cpu-inputs','cpu-prepared','cpu-checkpoint','joern-runtime','import-heads','import-prepared','import-checkpoint']);p.add_argument('--path',type=Path);p.add_argument('--joern-home',type=Path)
a=p.parse_args()
if not a.action.startswith('import-'):
    result=create_bundle(ROOT,a.path or ROOT/'bundles'/('grace-h1-'+a.action+'.zip'),a.action.replace('-','_'),joern_home=a.joern_home)
else:
    if not a.path: p.error('--path is required for import-heads')
    result=import_bundle(ROOT,a.path,{'import-heads':'trained_heads','import-prepared':'cpu_prepared','import-checkpoint':'cpu_checkpoint'}[a.action])
print(json.dumps(result,indent=2))
