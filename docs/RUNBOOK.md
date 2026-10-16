# Cách chạy và quản lý cache

## Điểm bắt đầu — full pipeline

Mở notebooks/full-pipeline.ipynb trên local trước. Mặc định notebook điều phối Kaggle train head → Colab CPU prepare → local LLM/evaluation; dừng ở Bước1/Bước2 nếu thiếu output phù hợp. Đọc docs/CLOUD_HANDOFF.md để biết chính xác zip cần upload/download.

Nhánh reveal chạy full test ReVeal (2.274 mẫu), hai LLM arm baseline_whitening/h1_supcon_hard, seed 42. Head CE train làm control, chưa gọi LLM. Query cohort luôn full snapshot, không có max_targets. Nhánh devign giữ notebook/kết quả cũ; artifact local và chuyển dataset được mô tả ở docs/DATASET_BRANCHES.md.

Nếu muốn mọi stage chạy local: TRAIN_ON_KAGGLE=False, PREPARE_ON_COLAB=False; local cần PyTorch/Joern. CLI train/prepare/preflight/infer/evaluate/all vẫn chạy local như trước. Thêm --config cho cấu hình khác, --root cho project root; --allow-api để gọi API còn thiếu.

## Môi trường

.env local chứa VERTEX_PROJECT, VERTEX_LOCATION và JOERN_HOME lấy từ cấu hình cũ nếu có. python-dotenv load nhưng không ghi đè biến môi trường đang được kernel đặt. Không copy credential JSON hoặc API keys; dùng ADC của WSL. GOOGLE_APPLICATION_CREDENTIALS có thể trỏ tới file credential đã có trên máy; đường dẫn Windows được map sang /mnt/... khi kernel Linux.

Nếu chạy Windows Python và Joern ở Linux, GraphStore gọi wsl.exe và dịch đường dẫn scratch sang /mnt/... . Nên dùng WSL kernel trực tiếp để giảm I/O và tránh khác môi trường xác thực. Không cần tải model CodeT5 vì embedding đã có.

Notebook local mặc định chỉ cài dependency CPU + google-genai để handoff/inference. Nếu chọn train local và torch chưa có, notebook chọn wheel CPU. Kaggle dùng GPU/PyTorch của runtime nếu đáp ứng requirements. Colab chỉ cài requirements-cpu.txt và Java21, không có bước cài torch/SDK LLM. requirements đặt giới hạn phiên bản lớn, runtime versions ghi trong manifest; không coi đó là lockfile tái lập bit-for-bit.

## Cache đã bàn giao

| Loại | Vị trí | Điều kiện dùng |
|---|---|---|
| Dataset | data/{dataset}/{split}.jsonl | Full file hash, record identity và code hash |
| Frozen embedding | cache/embeddings/{dataset}/{split}/ | NPY/index/hash, shape/dtype, encoder checkpoint/pooling/max_length đồng nhất |
| Graph features | cache/graphs/{code_sha256}/ | Content fingerprint/schema/code hash |
| Candidate pool cũ | cache/legacy_candidates/ | Target ID/code và đúng50 train IDs/code hashes, loại exact query code |
| Projected vectors mới | cache/projected/ | Geometry/input/head fingerprints và file checksums |
| Model head | artifacts/heads/ | Config/train/validation/source fingerprint; weights checksum |
| Response cũ/mới | cache/prior_runs/, cache/responses/ | Exact prompt hash + model/provider/decoding fingerprint |

Bộ copy hiện có 26.867 graph features trong 49.641 code hash duy nhất của sáu split. Con số còn thiếu không phải số phải parse cho một run: prepare chỉ cần query và semantic top5 demos của arm đó. Những graph cần mà chưa có sẽ được parse, export và cache. Đặc biệt ReVeal có thể cần thêm CPU/Joern dù embedding đầy đủ.

Không copy raw CPG binaries; snapshot features tự chứa node/edge/AST sequence cần dùng, cùng provenance. Đổi graph extractor hay schema là experiment khác; đừng trộn cache tùy tiện. Các artifact legacy được đánh dấu, không được nhận là CPG binary nguyên bản.

## Resume

- Train: last.pt lưu optimizer/model sau mỗi epoch; projection.npz và best.json giữ best checkpoint. Nếu đã có manifest hoàn tất và weights khớp thì bỏ qua training.
- Prepare: projected vectors, graph features và common candidates được cache. freeze.json khóa prompt/retrieval/manifest/contract/template/data manifest. Run đã freeze khớp được dùng lại.
- Inference: từng unique request thành công ghi ngay vào cache/responses; cùng prompt giữa các arm chỉ cần một response. Request exhausted không được cache như thành công; chạy lại chỉ gọi phần chưa thành công.
- Nếu thay model/temperature/seed/max tokens hoặc prompt, cache key mới được dùng. Không reuse chỉ vì cùng ID query.
- Chạy một coordinator/notebook trên thư mục project này tại một thời điểm; Kaggle dùng bản copy độc lập. Atomic writes bảo vệ checkpoint bị ngắt nhưng không phải cơ chế scheduling nhiều run cùng lúc.

## Kaggle

Tạo bundle bằng python tools/bundle.py training-inputs. Bundle chứa code/config, canonical train/validation và bốn embedding sets, không có test code/labels, Joern, raw responses hay credentials. data/manifest.json vẫn chứa hash/count metadata của cả sáu split để giữ training fingerprint; các test payload không được đưa vào zip.

Notebook Kaggle nhận cả zip còn nguyên hoặc thư mục bundle đã được giải nén dưới /kaggle/input. Có nhiều bundle thì đặt GRACE_H1_BUNDLE trỏ tới đúng zip/thư mục. Bật GPU và Internet, Run All. Giữ training config giống local để head được reuse đúng fingerprint.

Sau khi train, tải grace-h1-trained-heads.zip từ Outputs. Đặt vào bundles/ local (HEADS_BUNDLE mặc định trỏ đến file này), rồi Run All full-pipeline.ipynb lại; hoặc chạy python tools/bundle.py import-heads --path bundles/grace-h1-trained-heads.zip. Import kiểm tra mọi checksum trước khi publish, không ghi đè file khác nội dung.

## Colab CPU

Notebook notebooks/colab_cpu_prepare.ipynb nhận cpu-inputs.zip + joern-runtime.zip + trained-heads.zip qua Drive hoặc Files pane. Chọn accelerator None. Làm việc trên /content; lưu checkpoint zip về Drive khi xử lý graph và sau từng arm. Output cpu-prepared.zip mang frozen runs và cache mới về local. Xem docs/CLOUD_HANDOFF.md cho resume, matching fingerprints và cách đóng gói lại.

## Kết quả cần đọc

runs/experiment_summary.json tổng hợp từng arm và paired comparison. Trong mỗi run:

- run_manifest.json: source/config/data/encoder identities, commit, graph coverage và runtime.
- retrieval.jsonl: common candidates, semantic distances, lexical/AST scores và selected demo.
- prompts.jsonl, freeze.json: prompt đã render và hash.
- requests/: response/attempt/error/usage từng request mới; raw_responses.jsonl và predictions.jsonl: full cohort.
- inference_plan.json: lượng query/request thiếu; inference_summary.json: request mới, attempts, reused queries, token usage, latency.
- evaluation.json: confusion matrix valid-only, F1/MCC, invalid rate, recall_all_positives, trivial baselines.

Token usage không được provider báo sẽ là null, không giả bằng0. provider_usage_new_requests chỉ tính usage biết được của request mới; xem coverage_per_field. Latency là request wall-time có retry, không là kernel-only latency. Chi phí offline training/prepare được ghi riêng; chưa quy đổi ra tiền API.

Status complete phản ánh đủ output request, không bảo đảm điểm cao hoặc mọi response parse hợp lệ. Đọc invalid_rate và request_error_queries. Paired CI chỉ trên common-valid predictions; không che số invalid ở bảng full cohort.

## Thêm dữ liệu/cache sau này

tools/bootstrap_cache.py --workspace C:/MY-FILES/LAB-THAY-THO có thể copy lại nguồn đầy đủ; hash khớp thì reuse. Nếu nguồn thiếu bộ embedding cùng recipe, importer dừng thay vì trộn encoder. Không có fallback embedding ngẫu nhiên hoặc dùng test labels để fit head.

Trong Git chỉ version hóa code/docs/notebook/provenance summary. Runtime data/cache/runs/models/bundles và .env được ignore. tools/commit_step.py in kích thước từng file trước stage/commit; file vượt100MiB phải được ignore bằng đường dẫn cụ thể. Commit convention ở .agents/commit-convention.md.

## Nguồn API đối chiếu khi viết

- [Google GenAI SDK chính thức](https://github.com/googleapis/python-genai): client/generate_content, đóng client, giới hạn SDK trước3.0. Code nhận cả constructor enterprise mới và vertexai cũ.
- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html): checkpoint state dictionaries; load optimizer resume bằng weights_only=True. Retrieval dùng NPZ allow_pickle=False.

Các notebook chưa được thực thi và chưa có benchmark mới trong lượt bàn giao. Đây là implementation để chạy thí nghiệm H1, chưa phải bằng chứng H1 thắng baseline.
