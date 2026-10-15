# GRACE H1 — Security Projection

Triển khai **H1: projection bảo mật nhỏ trên CodeT5 đóng băng** để chọn demonstration, sau đó chạy đủ GRACE. Encoder không fine-tune và không cần tải lại weights vì đã có embedding cache.

## Bắt đầu

- [Notebook full pipeline local](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/full-pipeline.ipynb>)
- [Notebook Colab CPU preparation](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/colab_cpu_prepare.ipynb>)
- [Hướng dẫn chuyển local → Kaggle → Colab → local](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/CLOUD_HANDOFF.md>)
- [Notebook Kaggle train projection](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/kaggle_train_projection.ipynb>)
- [Cơ chế và thiết kế so sánh](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/METHOD.md>)
- [Cache, cấu hình và cách chạy](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/RUNBOOK.md>)

Mặc định notebook chạy **toàn bộ test Devign (2.732 query)**, so baseline whitening với H1 SupCon + hard negatives. Head CE cũng được train làm control; chỉ gọi LLM cho CE nếu bật thêm arm. Có thể thêm ReVeal trong configs/experiment.json; cả sáu split của hai dataset đã được copy.

## Pipeline

Cached CodeT5 768d → train-only projection 128d → common top50 → semantic top5 → lexical/AST rerank → một demo có nhãn → CPG + code + demo → Vertex LLM → parser → evaluation paired.

Baseline dùng whitening 256d fit trên train, cùng embedding recipe, candidate pool, reranking, graph, prompt và LLM. Đây là paired experiment mới, không ghép trực tiếp với số baseline CLS cũ. Các primitive baseline được snapshot trong grace_h1/reference.py; nguồn/hash ghi trong provenance/reference_sources.json.

## Tài nguyên đã copy

- Devign train/validation/test: 21.854 / 2.732 / 2.732 mẫu.
- ReVeal train/validation/test: 18.187 / 2.273 / 2.274 mẫu.
- Sáu embedding matrices masked_mean, cùng checkpoint fingerprint và thứ tự ID/code hash.
- **26.867 graph feature records**, candidate caches và response/checkpoint có sẵn.
- Tổng bytes nguồn đã copy: **598.157.163** (~570,45 MiB), chưa tính canonical JSONL sinh thêm.

provenance/cache_import.json chứa tổng hợp; cache/import_manifest.json ghi từng file, checksum và nguồn. Raw CPG binaries không copy: graph features đã export đủ cho retrieval/prompt; chỉ graph thiếu mới cần Joern. Số graph được copy không đồng nghĩa toàn bộ train/query của mọi arm đều đã có graph.

## Chạy local

Mở **notebooks/full-pipeline.ipynb** đầu tiên và Run All. Bước 1 hướng dẫn sang Kaggle để train head; Bước 2 hướng dẫn sang Colab CPU để prepare retrieval/graph/prompt. Tải output về bundles/ rồi Run All lại; khi đủ artifact khớp, notebook mới tiếp tục preflight → Vertex → evaluation ở local. Chi tiết ở docs/CLOUD_HANDOFF.md.

Luồng mặc định không train hoặc chạy Joern trên local. Dùng kernel đã có ADC cho inference. Nếu chủ động chạy tất cả local, đặt TRAIN_ON_KAGGLE=False và PREPARE_ON_COLAB=False; khi cần graph mới, đặt JOERN_HOME. Pipeline mặc định hai Joern workers, heap tối đa 2 GB mỗi process.

.env đã được tạo từ các cấu hình không chứa khóa bí mật tìm thấy ở workspace cũ (project/location/Joern). ADC dùng môi trường đã đăng nhập; credential files và API keys không được copy vào Git hay bundle. Có .env.example nếu chuyển máy.

CLI chạy các stage trên local, từ project root:

~~~bash
python -m pip install -r requirements.txt
python -m grace_h1.experiment train
python -m grace_h1.experiment prepare
python -m grace_h1.experiment preflight
python -m grace_h1.experiment infer --allow-api
python -m grace_h1.experiment evaluate
~~~

Nếu máy chưa có PyTorch và chỉ dùng CPU, cài PyTorch CPU trước requirements:

~~~bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
~~~

Chạy toàn bộ bằng CLI: python -m grace_h1.experiment all --allow-api. Không có --allow-api thì CLI chỉ dùng response cache, chưa gọi API còn thiếu. Notebook mặc định RUN_INFERENCE=True; chỉ đến inference sau khi các handoff Kaggle/Colab đã khớp. Cell preflight hiển thị lượng request thiếu trước đó.

## Bundle cloud đã chuẩn bị

- bundles/grace-h1-training-inputs.zip → Kaggle GPU, train head.
- bundles/grace-h1-cpu-inputs.zip → Colab CPU, dataset/embedding/candidate/graph cache.
- bundles/grace-h1-joern-runtime.zip → Colab, Joern 4.0.623 C/C++ runtime; notebook cài Java21.

Colab cần thêm grace-h1-trained-heads.zip lấy từ Kaggle, rồi xuất grace-h1-cpu-prepared.zip về local. CPU notebook có checkpoint zip về Drive; không cần credential hoặc gọi LLM. Kích thước/hash ghi tại provenance/cloud_bundles.json. Hướng dẫn từng lần chuyển môi trường ở docs/CLOUD_HANDOFF.md.

~~~bash
python tools/bundle.py training-inputs
python tools/bundle.py cpu-inputs
python tools/bundle.py import-heads --path bundles/grace-h1-trained-heads.zip
python tools/bundle.py import-prepared --path bundles/grace-h1-cpu-prepared.zip
~~~

## Artifact và resume

| Thư mục | Nội dung |
|---|---|
| data/ | Snapshot parquet + canonical JSONL + manifest |
| cache/embeddings/ | Sáu vector/index/manifest đã copy |
| cache/graphs/ | Exported graph features theo code hash |
| cache/candidates/, projected/ | Candidate pool và vector đã transform |
| cache/responses/ | Response dùng chung theo exact prompt + inference config |
| artifacts/heads/ | Weights NPZ, optimizer checkpoint, history và model manifest |
| runs/ | Retrieval, prompts, freeze, từng request, prediction, metrics và paired CI |
| provenance/ | Nguồn, hash và báo cáo copy cache |

Train resume theo epoch hoặc reuse head hoàn tất. Prepare reuse frozen run. Inference lưu mỗi request đã hoàn thành ngay và không gọi lại response khớp; request exhausted được thử lại ở lượt sau. API retry vẫn có thể phát sinh thêm request; báo cáo tách unique prompts, API attempts và cache hits.

## Trạng thái bàn giao

Đã tạo code, notebook điều phối local/Kaggle/Colab, cấu hình, copy cache và commit theo từng nhóm chức năng. **Chưa chạy notebook/training/benchmark/LLM và chưa thêm hoặc chạy test suite trong lượt authoring này.** Chỉ thực hiện copy dữ liệu với checksum và rà code/diff. Vì vậy chưa tuyên bố H1 làm tăng F1/MCC hoặc xác nhận runtime toàn pipeline.

Git được khởi tạo riêng, chưa có origin. Data/cache/artifacts/runs/bundles/.env được ignore; code/docs/notebook và provenance summary được version hóa. Mọi stage/commit kiểm tra kích thước file theo skill đã được chỉ định.
