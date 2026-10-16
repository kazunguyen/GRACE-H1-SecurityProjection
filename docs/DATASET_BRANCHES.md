# Nhánh Devign, ReVeal và artifact local

## Điểm đã lưu

Ngày 16/10/2026:

- Nhánh `devign` trỏ tới `e177b64`: giữ ba notebook với output hoàn tất Devign và README kết quả.
- Nhánh `reveal` được tạo từ `devign`, đổi datasets thành `["reveal"]`, giữ seed 42, hai objective và hai arm như Devign.
- Artifact Devign được chuyển vào `archives/devign/2026-10-16/`, tổng **1.084 file / 818.455.614 byte** gồm các file cấu hình/provenance được copy kèm. Không xóa dữ liệu Devign.

Git branch lưu các file được theo dõi. Chuyển branch không tự di chuyển hoặc phục hồi data/cache/artifact đang bị ignore.

## Bên trong archive Devign

Đường dẫn bên dưới tính từ `archives/devign/2026-10-16/`:

| Đường dẫn | Nội dung |
|---|---|
| artifacts/heads/devign/, artifacts/training_selection.json | Weights, checkpoint, history và head selection |
| runs/devign/, runs/comparisons/ | Frozen prompts, retrieval, request logs, predictions, metrics và paired comparison |
| runs/run_selection.json, runs/inference_experiment.json, runs/experiment_summary.json | Các file điều phối/tổng hợp Devign đã hoàn tất |
| cache/candidates/, cache/projected/, cache/cpu_input_inventory.json | Candidate pool, vector đã transform và inventory của Devign |
| bundles/ | Các input/output zip Devign và log/provenance Kaggle |
| configs/experiment.json, provenance/ | Bản copy cấu hình và thông tin bundle trước khi đổi dataset |
| archive_manifest.json | Danh mục từng file: nguồn, vị trí archive, byte và SHA-256 |

Bản tóm tắt archive được theo dõi tại [provenance/devign_archive.json](../provenance/devign_archive.json). Snapshot metrics nhỏ tiếp tục nằm trong [results/devign-test-2026-10-16/](../results/devign-test-2026-10-16/) trên cả hai branch.

Các file run/head bên trong archive giữ nguyên nội dung và đường dẫn gốc trong manifest; không sửa fingerprint của thí nghiệm đã hoàn tất. Nếu cần mở lại pipeline Devign, dùng checkout/thư mục làm việc riêng rồi copy artifact về đúng đường dẫn tương đối, cùng các dependency dùng chung cần thiết. Không trộn các file selection/bundle Devign vào lượt ReVeal đang chạy.

## Cache giữ dùng chung

- `data/` và `cache/embeddings/`: cả Devign và ReVeal, giữ nguyên data manifest để không đổi fingerprint dữ liệu.
- `cache/graphs/`: graph theo code hash, gồm cả graph mới của Devign; ReVeal được reuse nếu khớp.
- `cache/responses/`: response theo exact prompt + model/decoding; chỉ response khớp mới được reuse.
- `cache/legacy_candidates/`, `cache/prior_runs/`: cache lịch sử đã import, giữ cho retrieval/reuse.
- `bundles/grace-h1-joern-runtime.zip`: runtime Joern 4.0.623 dùng lại cho Colab.

Artifact mặc định `artifacts/training_selection.json`, `runs/run_selection.json`, `grace-h1-trained-heads.zip`, `grace-h1-cpu-prepared.zip` hiện chưa có cho ReVeal. Các bước cloud sẽ tạo chúng.

## Bắt đầu ReVeal

1. Mở [notebooks/full-pipeline.ipynb](../notebooks/full-pipeline.ipynb) trên local với kernel .venv WSL.
2. Bước 1: dùng `bundles/grace-h1-training-inputs.zip` mới cho ReVeal. Tạo/update Kaggle Dataset, thay input Devign cũ, upload notebook Kaggle của nhánh reveal và bật GPU/Internet. Notebook kiểm tra config phải là ReVeal trước khi train.
3. Tải `grace-h1-trained-heads.zip` mới về `bundles/`, chạy lại notebook local.
4. Bước 2: upload CPU-inputs mới, Joern runtime dùng chung và trained-heads ReVeal vào **MyDrive/GRACE-H1-transfer-reveal**. Dùng Colab runtime mới và notebook Colab của nhánh reveal. Không copy checkpoint hoặc cpu-prepared Devign vào thư mục này.
5. Tải `grace-h1-cpu-prepared.zip` ReVeal về `bundles/`, chạy lại notebook local để preflight → Vertex → evaluation.

Bundle input vẫn chứa các split của cả hai dataset theo cơ chế đóng gói hiện có, nhưng config chỉ chọn ReVeal. GPU input không có test payload. ReVeal gồm **18.187 train / 2.273 validation / 2.274 test**.

Lượt chuẩn bị này chỉ tách nhánh, di chuyển artifact và đóng gói lại input. Chưa chạy training, preparation, inference, evaluation hoặc test suite cho ReVeal.
