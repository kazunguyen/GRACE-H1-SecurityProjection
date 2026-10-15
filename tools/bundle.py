from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grace_h1.portable import create_bundle,import_bundle
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('action',choices=['training-inputs','trained-heads','import-heads']);p.add_argument('--path',type=Path)
a=p.parse_args()
if a.action=='training-inputs': result=create_bundle(ROOT,a.path or ROOT/'bundles'/'grace-h1-training-inputs.zip','training_inputs')
elif a.action=='trained-heads': result=create_bundle(ROOT,a.path or ROOT/'bundles'/'grace-h1-trained-heads.zip','trained_heads')
else:
    if not a.path: p.error('--path is required for import-heads')
    result=import_bundle(ROOT,a.path,'trained_heads')
print(json.dumps(result,indent=2))
