# GRACE H1 — Security Projection

Triển khai **H1: projection bảo mật nhỏ trên CodeT5 đóng băng** để chọn demonstration, sau đó chạy đủ GRACE. Encoder không fine-tune và không cần tải lại weights vì đã có embedding cache.

## Nhánh Projection V2 đang triển khai

Nhánh hiện tại `projection-v2-local-metric` triển khai residual MLP + local contrastive learning, balanced sampling, semi-hard warmup và giữ geometry của whitening teacher. Plan đã được commit trước code. [Nghiên cứu và plan](docs/PROJECTION_V2_RESEARCH_AND_PLAN.md) · [Cách chạy V2](docs/PROJECTION_V2_RUNBOOK.md).

Mở **[notebooks/full-pipeline.ipynb](notebooks/full-pipeline.ipynb)** đầu tiên. Notebook điều phối **Kaggle train → CPU local/Colab → local Vertex/evaluation**, có điểm dừng và chỉ dẫn chuyển notebook. Mặc định validation ReVeal và `RUN_INFERENCE=False`; chưa có benchmark V2. H1/Devign/ReVeal bên dưới là kết quả lịch sử, vẫn có thể quay lại nhánh `devign` hoặc `reveal`.

- [Kaggle train Projection V2](notebooks/kaggle_train_projection_v2.ipynb).
- [Colab CPU preparation V2](notebooks/colab_cpu_prepare_v2.ipynb).
- Input đã đóng gói: `bundles/projection-v2-training-inputs.zip`; metadata tại `provenance/projection_v2_bundles.json`.

## Bắt đầu với H1 lịch sử

- [Notebook full pipeline local](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/full-pipeline.ipynb>)
- [Notebook Colab CPU preparation](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/colab_cpu_prepare.ipynb>)
- [Hướng dẫn chuyển local → Kaggle → Colab → local](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/CLOUD_HANDOFF.md>)
- [Notebook Kaggle train projection](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/kaggle_train_projection.ipynb>)
- [Cơ chế và thiết kế so sánh](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/METHOD.md>)
- [Cache, cấu hình và cách chạy](<D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/docs/RUNBOOK.md>)

Nhánh reveal chạy **toàn bộ test ReVeal (2.274 query)**, so baseline whitening với H1 SupCon + hard negatives, seed 42. Inference và paired evaluation đã hoàn tất; kết quả được ghi bên dưới và snapshot trong `results/reveal-test-2026-10-17/`. Head CE được train làm control nhưng không nằm trong lượt GRACE + LLM này.

## Nhánh và artifact theo dataset

- **devign** giữ code, ba notebook đã chạy và README kết quả tại commit `e177b64`.
- **reveal** bắt đầu từ cùng commit, mặc định chạy ReVeal; các notebook đã được chạy cho lượt ReVeal và kết quả được ghi trong README này.
- Artifact Devign local được gom vào [archives/devign/2026-10-16/](archives/devign/2026-10-16/), giữ nguyên cấu trúc đường dẫn tương đối và checksum. Git không lưu các artifact lớn này.
- Graph, response cache, embedding gốc và Joern runtime được giữ dùng chung. Head, run selection và output bundle Devign đã rời đường dẫn mặc định để không bị nhập vào lượt ReVeal.
- Colab ReVeal dùng **MyDrive/GRACE-H1-transfer-reveal** và runtime mới. Chi tiết archive và cách bắt đầu ở [DATASET_BRANCHES.md](docs/DATASET_BRANCHES.md).

## Pipeline

Cached CodeT5 768d → train-only projection 128d → common top50 → semantic top5 → lexical/AST rerank → một demo có nhãn → CPG + code + demo → Vertex LLM → parser → evaluation paired.

Baseline dùng whitening 256d fit trên train, cùng embedding recipe, candidate pool, reranking, graph, prompt và LLM. Đây là paired experiment mới, không ghép trực tiếp với số baseline CLS cũ. Các primitive baseline được snapshot trong grace_h1/reference.py; nguồn/hash ghi trong provenance/reference_sources.json.

## Kết quả thực nghiệm Devign — snapshot 16/10/2026

Trên nhánh devign đã hoàn tất Kaggle train head → Colab CPU prepare → local Vertex inference và evaluation cho **toàn bộ 2.732 mẫu test Devign**, seed **42**. Cả hai arm có 2.732 dự đoán hợp lệ, không có request lỗi còn lại. ReVeal chưa được đánh giá trong lượt này; head CE đã train nhưng chưa chạy arm GRACE + LLM riêng.

### Chất lượng full pipeline

Hai arm dùng cùng CodeT5 masked_mean cache, common candidate pool, reranking, một demonstration, prompt và parser. LLM: **Qwen3 Coder 480B A35B Instruct MaaS trên Vertex AI** (`publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas`), temperature 0, seed 42, max_tokens 256; parser `exact_last_line_binary_v1`.

| Chỉ số trên test Devign | Baseline whitening 256d | H1 SupCon + hard negatives 128d |
|---|---:|---:|
| F1 | **35,73%** | **34,33%** |
| MCC | **0,0587** | **0,0280** |
| Precision | 51,04% | 48,34% |
| Recall | 27,49% | 26,61% |
| Accuracy | 54,58% | 53,22% |
| Balanced accuracy | 52,54% | 51,22% |
| TP / TN / FP / FN | 345 / 1.146 / 331 / 910 | 334 / 1.120 / 357 / 921 |
| Invalid / tổng mẫu | 0 / 2.732 | 0 / 2.732 |

**H1 chưa cải thiện GRACE trong lần chạy Devign này.** Chênh lệch H1 − baseline: F1 **−1,41 điểm phần trăm**, MCC **−0,0307**. Paired bootstrap 1.000 lần trên 2.732 cặp hợp lệ cho CI 95%:

- F1: **[−3,00; +0,12] điểm phần trăm**, khoảng còn chứa 0.
- MCC: **[−0,0513; −0,0090]**, khoảng hoàn toàn âm.

Đây là một seed, bootstrap theo query, chưa điều chỉnh phụ thuộc giữa code/project trùng nhau; chưa thể suy rộng sang ReVeal hoặc các seed khác. Recall của cả hai arm còn thấp. Baseline luôn dự đoán safe đạt accuracy 54,06% và MCC 0; luôn dự đoán vulnerable đạt F1 62,95% và MCC 0, vì vậy cần đọc F1 cùng MCC, precision và recall.

### Train head trên Kaggle

Mỗi head có **98.690 tham số**, train từ embedding đóng băng trên **Tesla T4**, PyTorch 2.11.0+cu128, NumPy 2.1.3. Train giữ 21.762 dòng unique có nhãn không mâu thuẫn từ 21.854 dòng gốc. Validation chọn head có 2.724 dòng sau khi loại 8 dòng trùng code train; tiêu chí chọn là validation MCC của classifier head.

| Head | Best epoch | Validation F1 | Validation MCC | Thời gian train ghi nhận | Peak CUDA allocated |
|---|---:|---:|---:|---:|---:|
| CE control | 16 | 58,79% | 0,2368 | 9,17 giây | 20,09 MiB |
| SupCon + hard negatives | 13 | 57,84% | 0,2222 | 12,12 giây | 21,54 MiB |

Các điểm validation này là **classifier proxy để chọn head**, không phải điểm full GRACE. Peak reserved của PyTorch là 26 MiB mỗi head; số đo allocator cho head không bao gồm toàn bộ bộ nhớ GPU, encoder hay thời gian cài môi trường.

### CPU preparation trên Colab

Output notebook ghi Colab CPU có RAM 12,67 GiB. Manifest ghi Python 3.13.16, NumPy 2.1.3 và RapidFuzz 3.14.6; bước prepare không gọi LLM.

| Arm | Prepare ghi nhận | Graph tạo mới | Graph tái sử dụng | Query có graph nodes + edges |
|---|---:|---:|---:|---:|
| Baseline whitening | 61,19 phút | 247 | 11.951 | 95,61% |
| H1 SupCon + hard negatives | 39,45 phút | 150 | 11.446 | 95,61% |

Hai arm có cùng target identity và common pool SHA-256. Coverage tương ứng 2.612/2.732 query; 120 query không có graph đồng thời chứa nodes và edges. Baseline chạy trước, H1 có thể tận dụng cache thêm; các thời gian này không phải phép đo tốc độ hai arm với cache ban đầu giống nhau.

### Inference và tái sử dụng response cache

**Bảng sau chỉ mô tả lượt inference cuối được lưu**, đã có cache từ trước; không phải tổng chi phí chạy từ cache rỗng. Mỗi arm vẫn gọi LLM cho từng unique prompt chưa có response khớp.

| Số liệu lượt cuối | Baseline whitening | H1 SupCon + hard negatives |
|---|---:|---:|
| Tổng query | 2.732 | 2.732 |
| Unique prompts | 2.731 | 2.730 |
| Query dùng cache | 2.162 | 2.269 |
| Unique request mới / API attempts | 569 / 569 | 463 / 463 |
| Prompt tokens của request mới | 1.853.075 | 1.515.387 |
| Output tokens của request mới | 1.138 | 926 |
| Tổng tokens của request mới | 1.854.213 | 1.516.313 |
| Wall time lượt inference | 269,06 giây | 205,84 giây |

Provider trả đủ usage cho toàn bộ request mới trong bảng. Preflight lưu trước khi chạy cả hai arm ghi H1 thiếu 724 prompt; đến lượt H1 thực chạy còn 463 request mới, tức có thêm 261 cache hit sau lượt baseline. **Chênh lệch request/thời gian trong bảng chưa chứng minh H1 giảm số lần gọi LLM hoặc nhanh hơn nhờ thuật toán.**

### Dữ liệu đối chiếu đã lưu trong Git

Các snapshot nhỏ được copy nguyên nội dung báo cáo lúc ghi README, giữ lại kết quả Devign khi lần chạy sau thay đổi file tổng hợp trong runs/ hoặc artifacts/:

- [Metrics test, paired CI, config và inference usage](results/devign-test-2026-10-16/experiment_summary.json).
- [Training selection, validation metrics, môi trường và hash head](results/devign-test-2026-10-16/training_selection.json).
- [Manifest prepare baseline](results/devign-test-2026-10-16/baseline_whitening_run_manifest.json) và [manifest prepare H1](results/devign-test-2026-10-16/h1_supcon_hard_run_manifest.json).

Raw predictions và request logs đã được chuyển vào archive local; graph/response cache vẫn dùng chung:

- `archives/devign/2026-10-16/runs/devign/test/baseline_whitening/9afe3268729648c6`
- `archives/devign/2026-10-16/runs/devign/test/h1_supcon_hard/7c4a52705268af6f`

## Kết quả thực nghiệm ReVeal — snapshot 17/10/2026

Inference và evaluation đã hoàn tất cho **toàn bộ 2.274 mẫu test ReVeal**, seed **42**. Cả hai arm có 2.274 dự đoán hợp lệ, không có lỗi request hoặc invalid output. LLM là Qwen3 Coder 480B A35B Instruct MaaS trên Vertex AI, temperature 0, max_tokens 256; parser `exact_last_line_binary_v1`.

### Chất lượng full pipeline

| Chỉ số test ReVeal | Baseline whitening 256d | H1 SupCon + hard negatives 128d |
|---|---:|---:|
| F1 | **17,11%** | **16,61%** |
| MCC | **0,0318** | **0,0246** |
| Precision | 11,65% | 11,30% |
| Recall | 32,17% | 31,30% |
| Accuracy | 68,47% | 68,21% |
| Balanced accuracy | 52,36% | 51,83% |
| TP / TN / FP / FN | 74 / 1.483 / 561 / 156 | 72 / 1.479 / 565 / 158 |
| Invalid / tổng mẫu | 0 / 2.274 | 0 / 2.274 |

H1 chưa cải thiện kết quả trên lượt ReVeal này. Chênh lệch H1 − baseline: F1 **−0,50 điểm phần trăm**, MCC **−0,0072**, precision **−0,35 điểm phần trăm**, recall **−0,87 điểm phần trăm**. Paired bootstrap 1.000 lần trên 2.274 query cho CI 95%: F1 **[−1,59; +0,52] điểm phần trăm**, MCC **[−0,0226; +0,0072]**; cả hai khoảng đều chứa 0. Bootstrap theo query chưa điều chỉnh phụ thuộc giữa các mẫu cùng code/project, nên đây là kết quả thăm dò trên một seed.

Accuracy cao một phần do dữ liệu lệch lớp: luôn dự đoán safe đạt **89,89% accuracy** nhưng F1 vulnerable bằng 0 và MCC bằng 0. Vì vậy cần xem đồng thời F1, MCC, precision và recall.

### Inference và usage LLM

Baseline chạy trước nên H1 dùng được response cache chung từ baseline. Bảng dưới là usage phát sinh trong đúng lượt inference đã lưu, không phải phép đo chi phí thuật toán với cache ban đầu ngang nhau.

| Số liệu lượt inference | Baseline whitening | H1 SupCon + hard negatives |
|---|---:|---:|
| Query / unique prompts | 2.274 / 2.270 | 2.274 / 2.270 |
| Query cache hit | 0 | 1.934 |
| Request mới / API attempts | 2.270 / 2.270 | 340 / 340 |
| Prompt tokens request mới | 7.151.713 | 1.007.657 |
| Tổng tokens request mới | 7.156.253 | 1.008.337 |
| Wall time lượt inference | 504,82 giây | 97,50 giây |

H1 có ít request và token mới trong lượt này vì **1.934 query khớp response cache do baseline vừa tạo**; không thể kết luận H1 tự thân giảm 85% số lần gọi LLM hay tăng tốc tương ứng. Hai arm đều hoàn tất đủ 2.274 query.

Snapshot kết quả và manifest:

- [Paired metrics và CI](results/reveal-test-2026-10-17/experiment_summary.json), [inference usage](results/reveal-test-2026-10-17/inference_experiment.json).
- [Baseline manifest](results/reveal-test-2026-10-17/baseline_whitening_run_manifest.json), [H1 manifest](results/reveal-test-2026-10-17/h1_supcon_hard_run_manifest.json).
- [Run selection](results/reveal-test-2026-10-17/run_selection.json), [training selection](results/reveal-test-2026-10-17/training_selection.json).

## Tài nguyên đã copy

- Devign train/validation/test: 21.854 / 2.732 / 2.732 mẫu.
- ReVeal train/validation/test: 18.187 / 2.273 / 2.274 mẫu.
- Sáu embedding matrices masked_mean, cùng checkpoint fingerprint và thứ tự ID/code hash.
- **26.867 graph feature records**, candidate caches và response/checkpoint có sẵn.
- Tổng bytes nguồn đã copy: **598.157.163** (~570,45 MiB), chưa tính canonical JSONL sinh thêm.

provenance/cache_import.json chứa tổng hợp; cache/import_manifest.json ghi từng file, checksum và nguồn. Raw CPG binaries không copy: graph features đã export đủ cho retrieval/prompt; chỉ graph thiếu mới cần Joern. Số graph được copy không đồng nghĩa toàn bộ train/query của mọi arm đều đã có graph.

## Chạy H1 local trên nhánh reveal

Các lệnh trong mục này mô tả H1 lịch sử; dùng nhánh `reveal` để chạy lại. Với V2 trên nhánh hiện tại, dùng notebook đầu vào và runbook V2 ở đầu README.

Mở **notebooks/full-pipeline.ipynb** đầu tiên và Run All. Head đã được train trên Kaggle; CPU preparation ReVeal đã chạy trên WSL `/home` và được nhập vào project. Notebook kiểm tra rồi dùng lại frozen runs, sau đó chạy preflight → Vertex inference → evaluation ở local. Nếu thiếu artifact hợp lệ, có thể chủ động bật handoff Kaggle/Colab theo hướng dẫn trong docs/CLOUD_HANDOFF.md.

Luồng mặc định không train hoặc chạy Joern trên local. Dùng kernel đã có ADC cho inference. Nếu chủ động chạy tất cả local, đặt TRAIN_ON_KAGGLE=False và PREPARE_ON_COLAB=False; khi cần graph mới, đặt JOERN_HOME. Nhánh reveal hiện dùng ba Joern workers, heap tối đa 2 GB mỗi process.

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

Chạy toàn bộ bằng CLI: python -m grace_h1.experiment all --allow-api. Không có --allow-api thì CLI chỉ dùng response cache, chưa gọi API còn thiếu. Notebook mặc định RUN_INFERENCE=True; trên nhánh reveal, nó dùng lại CPU preparation đã nhập và chỉ đến inference sau khi các artifact khớp. Cell preflight hiển thị lượng request thiếu trước đó.

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

Devign đã hoàn tất và được giữ trên nhánh devign cùng archive local. Nhánh reveal đã hoàn tất Kaggle train, CPU preparation trên WSL, Vertex inference và paired evaluation cho ReVeal test (2.274 query). Snapshot kết quả được lưu trong `results/reveal-test-2026-10-17/`. H1 chưa tăng F1/MCC trên test Devign ở seed 42. Lượt chuẩn bị này không chạy test suite.

Git được khởi tạo riêng, chưa có origin. Data/cache/artifacts/runs/bundles/.env và archive Devign được ignore; code/docs/notebook, provenance summary và snapshot kết quả nhỏ trong results/ được version hóa. Mọi stage/commit kiểm tra kích thước file theo skill đã được chỉ định.
