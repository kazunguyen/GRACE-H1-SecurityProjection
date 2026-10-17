"""Numerical, leakage-boundary, resume and shared-pipeline regression checks."""
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import tempfile
import unittest
import json
import zipfile
import numpy as np
import torch

from grace_h1.io import write_json, write_jsonl, read_json, file_hash, sha256, row_identity, fingerprint
from grace_h1.projection import whitening
from grace_h1.projection_v2 import (ProjectionV2Config, fit_teacher, build_model, normalize_np,
    project_arrays, mine_local_bank, balanced_batches, metric_losses, training_indices,
    train_projection_v2, project_embeddings_v2, acceptable)
from grace_h1 import experiment_v2
from grace_h1 import portable_v2

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    shutil.copytree(ROOT/'grace_h1',root/'grace_h1',ignore=shutil.ignore_patterns('__pycache__'))
    (root/'configs').mkdir()
    for name in ('reference_contract.yaml','prompt_template.txt'):
        shutil.copyfile(ROOT/'configs'/name,root/'configs'/name)
    rng=np.random.default_rng(123)
    manifest={'datasets':{'devign':{}}}
    all_rows=[]
    for split,n in [('train',60),('validation',12)]:
        rows=[]
        for i in range(n):
            code=f'int {split}_{i}(int x) {{ return x + {i}; }}'
            rows.append({'record_id':f'devign:{split}:{i}','code':code,'code_sha256':sha256(code),'label':int(i%3==0)})
        path=root/'data'/'devign'/(split+'.jsonl')
        write_jsonl(path,rows)
        manifest['datasets']['devign'][split]={'rows_path':path.relative_to(root).as_posix(),
            'rows_sha256':file_hash(path),'identity':row_identity(rows),'count':n}
        folder=root/'cache/embeddings/devign'/split;folder.mkdir(parents=True)
        x=rng.normal(size=(n,16)).astype(np.float32)
        np.save(folder/'vectors.npy',x)
        write_jsonl(folder/'index.jsonl',[{k:r[k] for k in ('record_id','code_sha256')} for r in rows])
        write_json(folder/'manifest.json',{'rows_identity':row_identity(rows),'shape':list(x.shape),
            'files':{p:file_hash(folder/p) for p in ('vectors.npy','index.jsonl')},
            'encoder_spec':{'architecture':'T5Encoder','checkpoint':{'test':'fixture'},'pooling':'masked_mean','max_length':512,'seed':42}})
        all_rows.extend(rows)
    write_json(root/'data/manifest.json',manifest)
    # No test split exists: training and model selection must still work.
    for row in all_rows:
        graph={'schema':'grace_h1_graph_v1','code_sha256':row['code_sha256'],
            'nodes':[{'id':'1','type':'METHOD','code':'f'},{'id':'2','type':'RETURN','code':'return x;'}],
            'edges':[{'src':'1','dst':'2','label':'AST'}], 'ast_edge_tokens':['1->2[AST]']}
        graph['feature_sha256']=fingerprint(graph)
        write_json(root/'cache/graphs'/row['code_sha256']/'features.json',graph)


def small_config(**updates):
    return replace(ProjectionV2Config(),dimension=8,hidden_dimension=12,batch_size=8,epochs=3,
                   positive_neighbors=2,negative_neighbors=4,negatives_per_anchor=2,patience=6,
                   device='cpu',**updates)


class ProjectionV2Tests(unittest.TestCase):
    def test_identity_numpy_parity_and_residual_bound(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(40,16)).astype(np.float32)
        cfg=small_config();mu,kernel,t=fit_teacher(x,cfg.dimension)
        expected=whitening(x,x,cfg.dimension)[0]
        np.testing.assert_allclose(t,expected,atol=1e-6)
        model=build_model(16,cfg).eval();u=torch.tensor(x)
        z,_=model(u,torch.tensor(t));np.testing.assert_allclose(z.detach().numpy(),t,atol=1e-7)
        with torch.no_grad(): model.fc2.weight.normal_(0,0.1)
        weights={'teacher_mean':mu,'teacher_kernel':kernel,'mean':np.zeros(16,dtype=np.float32),
                 'scale':np.ones(16,dtype=np.float32),'residual_radius':np.asarray(cfg.residual_radius,dtype=np.float32),
                 **{n.replace('.','_'):v.detach().numpy() for n,v in model.state_dict().items()}}
        z,_=model(u,torch.tensor(t));actual=project_arrays(x,weights,batch_size=7)
        np.testing.assert_allclose(z.detach().numpy(),actual,atol=2e-6)
        self.assertTrue(((actual*t).sum(1)>=np.sqrt(1-cfg.residual_radius**2)-1e-6).all())

    def test_local_mining_labels_self_and_balanced_coverage(self):
        labels=np.array([0]*11+[1]*4)
        vectors=normalize_np(np.random.default_rng(1).normal(size=(15,8)))
        positive,negative=mine_local_bank(vectors,labels,3,4,chunk=3)
        for i in range(15):
            self.assertNotIn(i,positive[i]);self.assertTrue((labels[positive[i]]==labels[i]).all())
            self.assertTrue((labels[negative[i]]!=labels[i]).all())
        batches=list(balanced_batches(labels,8,42))
        for b in batches: self.assertEqual(int((labels[b]==0).sum()),int((labels[b]==1).sum()))
        seen=np.concatenate(batches);np.testing.assert_array_equal(np.sort(seen[labels[seen]==0]),np.arange(11))
        np.testing.assert_array_equal(np.concatenate(list(balanced_batches(labels,8,42))),seen)

    def test_no_semihard_and_finite_gradients(self):
        a=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
        p=a.detach().clone();n=-p[:,None,:]
        local,semi,geometry,active=metric_losses(a,p,n,p,0.1,0.2)
        self.assertEqual(active,0.0);self.assertEqual(float(semi.detach()),0.0)
        (local+semi+geometry).backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_conflict_dedup_and_gate(self):
        rows=[{'code_sha256':'a','label':0},{'code_sha256':'a','label':0},
              {'code_sha256':'b','label':0},{'code_sha256':'b','label':1},{'code_sha256':'c','label':1}]
        np.testing.assert_array_equal(training_indices(rows),[0,4])
        base={'macro_label_precision_at_5':0.5,'per_class_label_precision_at_5':{'0':0.5,'1':0.5},'teacher_top5_overlap':1.0}
        improved=base|{'macro_label_precision_at_5':0.6,'teacher_top5_overlap':0.2}
        self.assertFalse(acceptable(improved,base,ProjectionV2Config()))
        self.assertTrue(acceptable(improved|{'teacher_top5_overlap':0.8},base,ProjectionV2Config()))

    def test_epoch_resume_and_identity_fallback(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';fixture(a);fixture(b)
            cfg=small_config(selection_min_gain=1.1)
            ma=train_projection_v2(a,'devign',cfg)
            pause=train_projection_v2(b,'devign',cfg,stop_after_epoch=1)
            self.assertEqual(pause['status'],'interrupted_at_epoch_boundary')
            mb=train_projection_v2(b,'devign',cfg)
            sa=torch.load((a/ma['path']).parent/'last.pt',weights_only=True)
            sb=torch.load((b/mb['path']).parent/'last.pt',weights_only=True)
            for key in sa['model']: torch.testing.assert_close(sa['model'][key],sb['model'][key],rtol=0,atol=0)
            self.assertEqual(sa['history'],sb['history'])
            self.assertEqual(ma['selection_status'],'baseline_fallback');self.assertEqual(ma['best']['epoch'],0)
            x=np.load(a/'cache/embeddings/devign/train/vectors.npy')
            np.testing.assert_allclose(project_embeddings_v2(x,ma,a),whitening(x,x,8)[0],atol=2e-6)
            with (a/ma['path']).open('ab') as f: f.write(b'corrupt')
            with self.assertRaises(ValueError): project_embeddings_v2(x,ma,a)

    def test_shared_pipeline_and_isolated_selections(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            root=Path(tmp);fixture(root)
            cfg=small_config(selection_min_gain=1.1)
            training=asdict(cfg);training.pop('objective')
            config={'datasets':['devign'],'objectives':['residual_local'],'splits':['validation'],
                    'training':training,'llm_arms':['baseline_whitening','h1_residual_local'],
                    'pipeline':{'whitening_dimension':8,'semantic_top_k':5,'lexical_weight':0.7,'graph_workers':2,'joern_home':None},
                    'allow_identity_fallback_run':True,'test_protocol_frozen':False,'inference_workers':1,'bootstrap':20}
            experiment_v2.train_all(root,config)
            runs=experiment_v2.prepare_all(root,config)
            self.assertEqual(len(experiment_v2.require_runs(root,config)),2)
            self.assertFalse((root/'artifacts/training_selection.json').exists())
            self.assertFalse((root/'runs/run_selection.json').exists())
            self.assertEqual(read_json(root/runs[1]['path']/'run_manifest.json')['target_count'],12)
            self.assertFalse(read_json(root/runs[1]['path']/'run_manifest.json')['query_labels_used_for_retrieval'])
            with self.assertRaises(ValueError): experiment_v2.validate_config(config|{'splits':['test']})

    def test_bundle_roundtrip_and_path_traversal(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            base=Path(tmp);root=base/'local';fixture(root)
            training=asdict(small_config(selection_min_gain=1.1));training.pop('objective')
            cfg={'datasets':['devign'],'objectives':['residual_local'],'splits':['validation'],'training':training,
                 'llm_arms':['baseline_whitening','h1_residual_local'],
                 'pipeline':{'whitening_dimension':8,'semantic_top_k':5,'lexical_weight':0.7,'graph_workers':2,'joern_home':None},
                 'allow_identity_fallback_run':True,'test_protocol_frozen':False,'inference_workers':1,'bootstrap':20}
            write_json(root/'configs/projection_v2.json',cfg)
            cpu=base/'cpu.zip';train_bundle=base/'train.zip';heads=base/'heads.zip';prepared=base/'prepared.zip'
            portable_v2.create_bundle(root,cpu,'cpu_inputs')
            portable_v2.create_bundle(root,train_bundle,'training_inputs')
            with zipfile.ZipFile(train_bundle) as z:
                self.assertFalse(any('/test/' in n or '/responses/' in n or n.endswith('/test.jsonl') for n in z.namelist()))
            experiment_v2.train_all(root,cfg)
            portable_v2.create_bundle(root,heads,'heads')
            remote=base/'remote';portable_v2.import_bundle(remote,cpu,'cpu_inputs')
            portable_v2.import_bundle(remote,heads,'heads');experiment_v2.prepare_all(remote,cfg)
            portable_v2.create_bundle(remote,prepared,'prepared')
            portable_v2.import_bundle(root,prepared,'prepared')
            self.assertEqual(len(experiment_v2.require_runs(root,cfg)),2)
            self.assertEqual(len(experiment_v2.preflight_all(root,cfg)),2)
            bad=base/'bad.zip'
            with zipfile.ZipFile(cpu) as z: m=json.loads(z.read('bundle_manifest.json'))
            m['entries']=[{'path':'../escape','bytes':0,'sha256':sha256(b'')}]
            with zipfile.ZipFile(bad,'w') as z:
                z.writestr('bundle_manifest.json',json.dumps(m));z.writestr('../escape','')
            with self.assertRaises(ValueError): portable_v2.import_bundle(root,bad,'cpu_inputs')

    def test_label_changes_invalidate_training_cache(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            root=Path(tmp);fixture(root);cfg=small_config(selection_min_gain=1.1)
            before=train_projection_v2(root,'devign',cfg)
            path=root/'data/devign/train.jsonl'
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            rows[0]['label']=1-rows[0]['label'];write_jsonl(path,rows)
            data=read_json(root/'data/manifest.json');data['datasets']['devign']['train']['rows_sha256']=file_hash(path)
            write_json(root/'data/manifest.json',data)
            after=train_projection_v2(root,'devign',cfg)
            self.assertNotEqual(before['training_fingerprint'],after['training_fingerprint'])
            self.assertEqual(len(list((root/'cache/projection_v2').glob('*.npz'))),2)


if __name__=='__main__': unittest.main()
