"""Generate the single branch entry notebook and its Kaggle/Colab handoffs."""
from pathlib import Path
import json
import hashlib
import textwrap

ROOT=Path(__file__).resolve().parents[1]

def md(source):
    text=textwrap.dedent(source).strip()
    return {'cell_type':'markdown','id':hashlib.sha256(('md'+text).encode()).hexdigest()[:12],'metadata':{},'source':text.splitlines(keepends=True)}

def code(source):
    text=textwrap.dedent(source).strip()+'\n'
    return {'cell_type':'code','id':hashlib.sha256(('code'+text).encode()).hexdigest()[:12],'metadata':{},'execution_count':None,'outputs':[],
            'source':text.splitlines(keepends=True)}

def save(name,cells):
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python'},'grace_workflow':'projection_v2'},'nbformat':4,'nbformat_minor':5}
    (ROOT/'notebooks'/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')

BOOTSTRAP='''
import hashlib, json, sys, zipfile
from pathlib import Path, PurePosixPath
def bootstrap_sources(root, archive_path, expected_kind):
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('bundle_manifest.json'))
        if manifest['schema']!='grace_projection_v2_bundle_v1' or manifest['kind']!=expected_kind:
            raise ValueError('Wrong V2 input bundle')
        for entry in manifest['entries']:
            name=entry['path'];p=PurePosixPath(name)
            source_file=name.startswith('grace_h1/') and p.suffix=='.py'
            dependency_file=name in ('requirements.txt','requirements-cpu.txt','requirements-training.txt')
            if not (source_file or dependency_file): continue
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\\\' in name:
                raise ValueError('Unsafe source path')
            data=archive.read(name)
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('Source checksum mismatch')
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('Source changed: use a fresh cloud ROOT')
            target.write_bytes(data)
    if str(root) not in sys.path: sys.path.insert(0,str(root))
'''

save('full-pipeline.ipynb',[
md('''
    # GRACE Projection V2 — bắt đầu tại đây

    Nhánh `projection-v2-local-metric` dùng residual MLP + local contrastive training. H1/ReVeal lịch sử đã lưu trên nhánh `reveal`; notebook này điều phối V2 và dùng cache chung.

    | Bước | Chạy ở đâu | Cần chuyển môi trường khi nào |
    |---|---|---|
    | 1. Train và chọn head bằng validation retrieval proxy | Kaggle GPU, hoặc local | Chưa có head: cell tạo `projection-v2-training-inputs.zip` rồi dừng |
    | 2. Retrieval/graph/prompt CPU | Colab CPU, hoặc local WSL | Chưa có frozen runs: cell tạo `projection-v2-cpu-inputs.zip` rồi dừng |
    | 3. Preflight → LLM → evaluation | Local có Vertex ADC | Quay lại đây sau khi tải `projection-v2-heads.zip` và `projection-v2-cpu-prepared.zip` vào `bundles/` |

    Mặc định chỉ chuẩn bị/đánh giá **validation ReVeal**, chưa gọi API mới. Đọc `docs/PROJECTION_V2_RESEARCH_AND_PLAN.md` và `docs/PROJECTION_V2_RUNBOOK.md` trước khi chuyển sang test.
'''),
code('''
    from pathlib import Path
    import os,sys,json
    candidates=([Path(os.environ['GRACE_H1_ROOT'])] if os.environ.get('GRACE_H1_ROOT') else [])
    candidates += [Path.cwd(),*Path.cwd().parents,Path('D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection'),Path('/mnt/d/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection')]
    ROOT=next((p.resolve() for p in candidates if (p/'grace_h1').is_dir()),None)
    if ROOT is None: raise RuntimeError('Set GRACE_H1_ROOT to the project folder')
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    TRAIN_ON_KAGGLE=True
    PREPARE_ON_COLAB=True  # False để prepare bằng Joern local WSL; dùng cùng graph cache.
    RUN_INFERENCE=False   # True sẽ gọi Vertex cho prompt chưa có response cache.
    from grace_h1.io import load_local_environment
    from grace_h1.experiment_v2 import load_config,validate_config,require_heads,require_runs,train_all,prepare_all,preflight_all,infer_all,evaluate_all
    from grace_h1.portable_v2 import create_bundle,import_bundle
    load_local_environment(ROOT)
    config=load_config(ROOT);validate_config(config)
    HEADS=ROOT/'bundles/projection-v2-heads.zip'
    PREPARED=ROOT/'bundles/projection-v2-cpu-prepared.zip'
    print('Project:',ROOT,'\u005cn',json.dumps(config,indent=2))
'''),
md('''
    ## Bước 1 — Kaggle train

    Khi cell dưới dừng, upload zip được tạo lên Kaggle cùng `notebooks/kaggle_train_projection_v2.ipynb`. Bật GPU T4 và Internet, Run All. Tải `projection-v2-heads.zip` về `bundles/` rồi chạy lại từ đầu notebook này. Có thể tái sử dụng checkpoint V2 trong notebook Kaggle. Không cần CodeT5 weights; train dùng embedding cache.
'''),
code('''
    if HEADS.exists(): print(import_bundle(ROOT,HEADS,'heads'))
    try:
        selection=require_heads(ROOT,config)
    except (FileNotFoundError,KeyError,ValueError) as exc:
        if TRAIN_ON_KAGGLE:
            print(create_bundle(ROOT,ROOT/'bundles/projection-v2-training-inputs.zip','training_inputs'))
            raise RuntimeError('DỪNG BƯỚC 1: chạy notebooks/kaggle_train_projection_v2.ipynb trên Kaggle GPU; tải projection-v2-heads.zip về bundles/. '+str(exc)) from exc
        selection=train_all(ROOT,config)
    for dataset,heads in selection['heads'].items():
        for objective,h in heads.items():
            print(dataset,objective,h['selection_status'],'best epoch',h['best']['epoch'],'macro P@5',h['best']['retrieval']['macro_label_precision_at_5'])
'''),
md('''
    ## Bước 2 — CPU preparation

    Nếu head chỉ có `baseline_fallback`, notebook dừng trước khi tạo graph: chưa có proxy gain để chạy LLM. Đọc history và điều chỉnh bằng validation. `allow_identity_fallback_run=true` chỉ dùng khi chủ động đo identity control.

    Chạy Colab: đặt `projection-v2-cpu-inputs.zip`, `projection-v2-heads.zip`, và Joern runtime có sẵn `grace-h1-joern-runtime.zip` vào **MyDrive/GRACE-projection-v2-transfer**. Chạy `notebooks/colab_cpu_prepare_v2.ipynb` với accelerator **None**. Tải `projection-v2-cpu-prepared.zip` về `bundles/` rồi quay lại đây. Muốn chạy CPU local WSL, đổi `PREPARE_ON_COLAB=False` và đặt `JOERN_HOME` hợp lệ.
'''),
code('''
    blocked=[f'{d}/{a}' for d in config['datasets'] for a in config['llm_arms'] if a!='baseline_whitening' and selection['heads'][d][a.removeprefix('h1_')]['selection_status']=='baseline_fallback']
    if blocked and not config.get('allow_identity_fallback_run',False):
        raise RuntimeError('Chưa có validation retrieval proxy gain: '+', '.join(blocked)+'. Tiếp tục nghiên cứu offline trước khi chạy LLM.')
    if PREPARED.exists(): print(import_bundle(ROOT,PREPARED,'prepared'))
    try:
        run_rows=require_runs(ROOT,config)
        print('Dùng lại frozen V2 preparation đã có.')
    except (FileNotFoundError,KeyError,ValueError) as exc:
        if PREPARE_ON_COLAB:
            print(create_bundle(ROOT,ROOT/'bundles/projection-v2-cpu-inputs.zip','cpu_inputs'))
            raise RuntimeError('DỪNG BƯỚC 2: chạy notebooks/colab_cpu_prepare_v2.ipynb; tải projection-v2-cpu-prepared.zip về bundles/. '+str(exc)) from exc
        run_rows=prepare_all(ROOT,config)
    print(run_rows)
'''),
md('''
    ## Bước 3 — Preflight và inference local

    Cell preflight không gọi API. Kiểm tra số request thiếu, rồi đặt `RUN_INFERENCE=True` để chạy Vertex bằng ADC local. Hai arm dùng cùng prompt template/graph/query cohort/generation settings. Response cache chỉ được tái sử dụng khi exact prompt và settings khớp.
'''),
code('''
    plans=preflight_all(ROOT,config)
    for plan in plans: print(plan)
'''),
code('''
    if RUN_INFERENCE:
        inference=infer_all(ROOT,config,allow_api=True)
        for item in inference: print(item)
    else:
        print('Chưa gọi API. Đặt RUN_INFERENCE=True sau khi xem preflight nếu muốn chạy inference.')
'''),
md('''
    ## Bước 4 — Paired evaluation và chuyển sang test

    Báo cáo V2 nằm trong `runs/projection_v2/experiment_summary.json`; bảng retrieval proxy không thay thế F1/MCC của GRACE. Chỉ sau khi đánh giá validation, lưu cấu hình và checkpoint đã chọn, rồi đổi `splits` thành `["test"]` và `test_protocol_frozen=true` trong `configs/projection_v2.json`. Đây là xác nhận protocol của người chạy, không phải bằng chứng đã chạy validation. Training identity không đổi khi đổi split; Bước 1 dùng lại head, Bước 2 tạo input CPU mới. Không dùng test để chọn hyperparameter/seed.
'''),
code('''
    if all((ROOT/r['path']/'predictions.jsonl').exists() for r in run_rows):
        report=evaluate_all(ROOT,config)
        print('Status:',report['status'])
        for row in report['scores']: print(row['dataset'],row['arm'],row['metrics'])
        for comparison in report['comparisons']: print(comparison)
    else:
        print('Chưa có đủ predictions; tiếp tục Bước 3 trước khi evaluation.')
''')])

save('kaggle_train_projection_v2.ipynb',[
md('''
    # Projection V2 — Kaggle GPU training

    Upload `projection-v2-training-inputs.zip` từ Bước 1 của notebook full-pipeline. Bật GPU T4 và Internet. Bundle chỉ chứa train/validation cho dataset trong config; không có test vectors hoặc response/credential. Mạng residual nhỏ đọc embedding cache, không tải CodeT5 lại. Xuất `projection-v2-heads.zip` về `bundles/` local.
'''),
code('''
    from pathlib import Path
    import os,sys,subprocess
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    ROOT=Path('/kaggle/working/grace-projection-v2')
    matches=sorted(Path('/kaggle/input').rglob('projection-v2-training-inputs.zip'))
    if len(matches)!=1: raise RuntimeError('Upload đúng một projection-v2-training-inputs.zip hoặc gán INPUT_ZIP bằng đường dẫn cụ thể')
    INPUT_ZIP=matches[0]
    RESUME_ZIP=None  # Có thể gán đường dẫn projection-v2-training-checkpoint.zip của lượt trước.
    EXTRA_SEEDS=[]  # [43,44] để kiểm tra replication offline; không chọn seed tốt nhất bằng test.
'''),
code(BOOTSTRAP+'''
bootstrap_sources(ROOT,INPUT_ZIP,'training_inputs')
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-training.txt')])
from grace_h1.portable_v2 import import_bundle,create_bundle
print(import_bundle(ROOT,INPUT_ZIP,'training_inputs'))
if RESUME_ZIP:
    with zipfile.ZipFile(RESUME_ZIP) as z: resume_kind=json.loads(z.read('bundle_manifest.json'))['kind']
    if resume_kind not in ('heads','training_checkpoint'): raise ValueError('Resume archive must be a V2 training output')
    print(import_bundle(ROOT,RESUME_ZIP,resume_kind))
from grace_h1.experiment_v2 import load_config,train_all
from grace_h1.io import write_json
config=load_config(ROOT)
print(json.dumps(config,indent=2))
'''),
md('''
    ## Train và chọn checkpoint

    Log mỗi epoch có macro label P@5 trên validation và overlap với teacher. Checkpoint `last.pt` được ghi mỗi epoch; interrupt chạy phần finally để đóng gói phần đã xong. `baseline_fallback` nghĩa là chưa tìm được learned checkpoint đạt các gate. CE control cùng kiến trúc giúp tách hiệu quả của loss local contrastive. Các số này chưa là benchmark GRACE + LLM.
'''),
code('''
    import copy
    try:
        selected=train_all(ROOT,config)
        replications={str(config['training']['seed']):selected}
        for seed in EXTRA_SEEDS:
            replica=copy.deepcopy(config);replica['training']['seed']=int(seed)
            replications[str(seed)]=train_all(ROOT,replica)
        if EXTRA_SEEDS:
            write_json(ROOT/'artifacts/projection_v2/replications.json',replications)
            selected=train_all(ROOT,config)  # Khôi phục selection seed chính.
    finally:
        if (ROOT/'artifacts/projection_v2').exists():
            print(create_bundle(ROOT,Path('/kaggle/working/projection-v2-training-checkpoint.zip'),'training_checkpoint'))
    for d,heads in selected['heads'].items():
        for name,h in heads.items(): print(d,name,h['selection_status'],h['best'])
'''),
code('''
    print(create_bundle(ROOT,Path('/kaggle/working/projection-v2-heads.zip'),'heads'))
    print('Tải projection-v2-heads.zip vào bundles/ local và quay lại full-pipeline.ipynb, Bước 1.')
''')])

save('colab_cpu_prepare_v2.ipynb',[
md('''
    # Projection V2 — Colab CPU preparation

    Accelerator **None**. Đặt ba zip `projection-v2-cpu-inputs.zip`, `projection-v2-heads.zip`, `grace-h1-joern-runtime.zip` vào MyDrive/GRACE-projection-v2-transfer. Không cần Vertex credential hoặc GPU. I/O parse chạy ở `/content`, Drive chỉ dùng cho zip đầu vào/checkpoint/output. Không chạy hai notebook cùng lúc vào cùng thư mục transfer.
'''),
code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import os,sys,subprocess,shutil
    ROOT=Path('/content/grace-projection-v2')
    TRANSFER=Path('/content/drive/MyDrive/GRACE-projection-v2-transfer')
    INPUT_ZIP=TRANSFER/'projection-v2-cpu-inputs.zip'
    HEADS_ZIP=TRANSFER/'projection-v2-heads.zip'
    JOERN_ZIP=TRANSFER/'grace-h1-joern-runtime.zip'
    CHECKPOINT=TRANSFER/'projection-v2-cpu-checkpoint.zip'
    for p in (INPUT_ZIP,HEADS_ZIP,JOERN_ZIP):
        if not p.is_file(): raise FileNotFoundError(p)
'''),
code(BOOTSTRAP+'''
bootstrap_sources(ROOT,INPUT_ZIP,'cpu_inputs')
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-cpu.txt'),'tqdm>=4.66'])
from grace_h1.portable_v2 import import_bundle,create_bundle
from grace_h1.portable import import_bundle as import_joern_bundle
print(import_bundle(ROOT,INPUT_ZIP,'cpu_inputs'))
print(import_bundle(ROOT,HEADS_ZIP,'heads'))
print(import_joern_bundle(ROOT,JOERN_ZIP,'joern_runtime'))
subprocess.check_call(['apt-get','update','-qq'])
subprocess.check_call(['apt-get','install','-y','-qq','openjdk-21-jre-headless'])
os.environ['JAVA_HOME']='/usr/lib/jvm/java-21-openjdk-amd64'
os.environ['PATH']=os.environ['JAVA_HOME']+'/bin:'+os.environ['PATH']
os.environ['JOERN_HOME']=str(ROOT/'tools/joern/joern-cli')
if CHECKPOINT.exists(): print(import_bundle(ROOT,CHECKPOINT,'checkpoint'))
from grace_h1.experiment_v2 import load_config,prepare_all
config=load_config(ROOT)
print('workers:',config['pipeline']['graph_workers'],'datasets/splits:',config['datasets'],config['splits'])
'''),
md('''
    ## Chuẩn bị và theo dõi graph

    Thanh tiến độ cập nhật theo graph hoàn tất, có cache hit, parse mới và ID đang chạy. Giữ 3 workers như config hiện tại. Checkpoint theo mốc 500 graph có thay đổi, cuối arm và khi interrupt; không checkpoint sau từng mẫu. Nếu cần đổi workers/config/source, tạo lại CPU-inputs tương ứng trước khi resume.
'''),
code('''
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from tqdm.auto import tqdm
    from grace_h1 import pipeline as pipeline_module
    class ProgressPool(ThreadPoolExecutor):
        def map(self,fn,rows,timeout=None,chunksize=1):
            lock=Lock();state={'cache':0,'parsed':0};active={}
            bar=tqdm(total=len(rows),desc='Graph preparation',unit='graph',dynamic_ncols=True)
            def tracked(row):
                h=row['code_sha256'];folder=ROOT/'cache/graphs'/h
                cached=(folder/'features.json.gz').exists() or (folder/'features.json').exists()
                with lock:
                    active[h]=row['record_id']
                    bar.set_postfix_str(f"cache {state['cache']} | parsed {state['parsed']} | đang chạy {len(active)}: {str(list(active.values())[:2])[:100]}")
                try:
                    result=fn(row)
                    with lock:
                        state['cache' if cached else 'parsed']+=1
                        bar.update(1)
                    return result
                finally:
                    with lock:
                        active.pop(h,None)
                        bar.set_postfix_str(f"cache {state['cache']} | parsed {state['parsed']} | đang chạy {len(active)}: {str(list(active.values())[:2])[:100]}")
            def results():
                try: yield from super(ProgressPool,self).map(tracked,rows,timeout=timeout,chunksize=chunksize)
                finally: bar.close()
            return results()
    pipeline_module.ThreadPoolExecutor=ProgressPool
    def checkpoint():
        local=ROOT/'bundles/projection-v2-cpu-checkpoint.zip'
        info=create_bundle(ROOT,local,'checkpoint')
        target=CHECKPOINT.with_name(CHECKPOINT.name+'.tmp')
        shutil.copyfile(local,target);os.replace(target,CHECKPOINT)
        tqdm.write(f"Checkpoint saved: {CHECKPOINT} ({info['bytes']} bytes)")
    try:
        rows=prepare_all(ROOT,config,checkpoint_callback=checkpoint)
    finally:
        checkpoint()
    print(rows)
'''),
code('''
    output=ROOT/'bundles/projection-v2-cpu-prepared.zip'
    print(create_bundle(ROOT,output,'prepared'))
    shutil.copyfile(output,TRANSFER/output.name)
    from google.colab import files
    files.download(str(output))
    print('Đặt projection-v2-cpu-prepared.zip vào bundles/ local, mở lại full-pipeline.ipynb và chạy từ đầu.')
''')])
print('Generated V2 entry, Kaggle and Colab notebooks')
