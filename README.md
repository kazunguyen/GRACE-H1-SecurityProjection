# GRACE H1 — Security Projection

Triển khai **H1: projection bảo mật nhỏ trên CodeT5 đóng băng** để chọn demonstration, sau đó chạy đủ GRACE. Encoder không fine-tune và không cần tải lại weights vì đã có embedding cache.

## Bắt đầu

- [Notebook full pipeline local](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/full_pipeline.ipynb>)
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

Dùng kernel WSL hiện có có Joern/ADC; mở notebook local rồi Run All. Cell đầu cho phép đổi dataset, workers, bật/tắt inference và import head đã train trên Kaggle. Nếu cần graph mới, đặt JOERN_HOME; pipeline mặc định hai Joern workers, heap tối đa 2 GB mỗi process.

.env đã được tạo từ các cấu hình không chứa khóa bí mật tìm thấy ở workspace cũ (project/location/Joern). ADC dùng môi trường đã đăng nhập; credential files và API keys không được copy vào Git hay bundle. Có .env.example nếu chuyển máy.

CLI tương đương, từ project root:

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

Chạy toàn bộ bằng CLI: python -m grace_h1.experiment all --allow-api. Không có --allow-api thì CLI chỉ dùng response cache, chưa gọi API còn thiếu. Notebook mặc định RUN_INFERENCE=True để chạy đủ các stage khi người dùng Run All; cell preflight hiển thị lượng request thiếu trước đó.

## Kaggle tùy chọn

1. Đính kèm bundle đã tạo sẵn [grace-h1-training-inputs.zip](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/bundles/grace-h1-training-inputs.zip>) (158.284.487 bytes, ~151 MiB) làm Kaggle Dataset. Hash nằm ở provenance/training_bundle.json; tạo lại bundle nếu đổi code/config/input.
2. Upload notebook Kaggle, bật GPU và Internet, Run All. Chỉ train các head; không gọi LLM, không cần Joern và không có test code/labels trong input bundle.
3. Tải grace-h1-trained-heads.zip về bundles/ của project local.
4. Đặt TRAINED_BUNDLE trong notebook local rồi Run All. Model/config/data/source khớp sẽ reuse head, không train lại.

CLI tạo/import bundle:

~~~bash
python tools/bundle.py training-inputs
python tools/bundle.py import-heads --path bundles/grace-h1-trained-heads.zip
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

Đã tạo code, notebook, cấu hình, copy cache và commit theo từng nhóm chức năng. **Chưa chạy notebook/training/benchmark/LLM và chưa thêm hoặc chạy test suite trong lượt authoring này.** Chỉ thực hiện copy dữ liệu với checksum và rà code/diff. Vì vậy chưa tuyên bố H1 làm tăng F1/MCC hoặc xác nhận runtime toàn pipeline.

Git được khởi tạo riêng, chưa có origin. Data/cache/artifacts/runs/bundles/.env được ignore; code/docs/notebook và provenance summary được version hóa. Mọi stage/commit kiểm tra kích thước file theo skill đã được chỉ định.
