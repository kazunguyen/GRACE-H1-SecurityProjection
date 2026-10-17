# Chạy Projection V2

Nhánh `projection-v2-local-metric`. Điểm vào duy nhất là `notebooks/full-pipeline.ipynb` trên nhánh này. H1 và kết quả hoàn tất Devign/ReVeal đã được giữ trên nhánh `devign`/`reveal` và snapshot trong `results/`.

## Đã triển khai gì

- MLP residual 768→256→256 với teacher whitening256, zero-init ở layer cuối và residual radius0,25. Tổng263.170 trainable parameters với CodeT5 cache768d/hidden256.
- Balanced anchor batches; local positive bank8 và negative bank32; mỗi anchor lấy1 positive/8 negatives. CE + local InfoNCE + semi-hard warmup + cosine-geometry loss.
- Chọn checkpoint theo validation macro label P@5 trong common top50, có per-class drop/teacher-overlap gates. Classifier MCC được báo riêng. `baseline_fallback` không phải head tốt hơn baseline.
- CE control cùng kiến trúc, sampler, geometry và tiêu chí selection. Loss weights/locality/radius cấu hình được để ablate. Không mặc định tăng số demo hoặc thay lexical/AST reranker.
- Export NPZ, NumPy projection ở preparation. Train và chọn checkpoint không đọc test split/response. Không tải lại CodeT5 weights.

Các mặc định là giả thuyết ban đầu cần validation, chưa là hyperparameter tối ưu hay bằng chứng tăng benchmark. Xem nghiên cứu/công thức/control trong [plan](PROJECTION_V2_RESEARCH_AND_PLAN.md).

## 1. Cấu hình và input

Chỉnh `configs/projection_v2.json`. Mặc định dataset `reveal`, split inference `validation`, train `residual_ce` và `residual_local`; chỉ bật LLM arms baseline + residual_local. Đổi `datasets` sang `["devign"]` hoặc cả hai trước khi tạo training bundle; input chỉ chứa dataset được chọn. Pipeline H1 cũ vẫn giữ cấu hình riêng `configs/experiment.json`.

Ở local hiện có venv, embeddings/canonical datasets, graph/candidate/response cache và Joern runtime zip. Notebook full-pipeline dùng các đường dẫn này trực tiếp. Nếu chuyển máy, cài `requirements.txt`; CPU preparation chỉ cần `requirements-cpu.txt`, train chỉ cần `requirements-training.txt` (GPU Kaggle có torch sẵn).

## 2. Kaggle training

1. Cell Bước1 của full-pipeline kiểm tra head. Nếu chưa có, nó tạo `bundles/projection-v2-training-inputs.zip` rồi dừng có chỉ dẫn.
2. Upload zip vào Kaggle dataset/input; mở `notebooks/kaggle_train_projection_v2.ipynb`, bật GPU T4 và Internet, Run All.
3. Log mỗi epoch ghi loss, macro label P@5 và top5 overlap. `last.pt` chứa model/optimizer/scheduler/RNG. Khi interrupt, finally xuất `projection-v2-training-checkpoint.zip`; gán `RESUME_ZIP` ở lần tiếp theo để import phần đã xong. Runtime bị xóa đột ngột trước khi export thì chỉ checkpoint/Output đã lưu ra ngoài còn dùng được.
4. Tải `projection-v2-heads.zip` vào `bundles/` local. Bundle này cũng dùng trên Colab CPU. `EXTRA_SEEDS=[43,44]` chạy replication offline và khôi phục selection seed42; không chọn seed tốt nhất bằng test. Replication summary được lưu kèm artifact.
5. Chạy lại full-pipeline. Nếu arm yêu cầu có `baseline_fallback`, mặc định dừng trước CPU parsing/LLM. Đọc history, sửa config dựa trên validation và train lại. Muốn đo identity control thì bật `allow_identity_fallback_run` có chủ đích.

Training artifact nằm trong `artifacts/projection_v2/<dataset>/<objective>/<fingerprint>/`; selection V2 không ghi đè selection H1. File trọng số đã có final manifest là immutable; checkpoint chưa hoàn tất có thể được cập nhật sau khi xác minh toàn bộ archive.

## 3. CPU local hoặc Colab

Để chạy local WSL, đặt `PREPARE_ON_COLAB=False`, `JOERN_HOME` trỏ đến Joern 4.0.623/Java21. Cache graph hiện có được dùng lại; graph thiếu mới parse.

Để chạy Colab CPU, Bước2 tạo `projection-v2-cpu-inputs.zip`. Đặt zip đó, `projection-v2-heads.zip` và runtime có sẵn `grace-h1-joern-runtime.zip` vào `MyDrive/GRACE-projection-v2-transfer`. Mở `notebooks/colab_cpu_prepare_v2.ipynb`, accelerator **None**, Run All. Parsing chạy trên `/content`, không parse trực tiếp trên Google Drive. Progress bar hiển thị graph hoàn tất, cache hit, parse mới và ID đang chạy; config mặc định3 workers, heap2GB mỗi Joern process.

Checkpoint Drive theo mốc500 graph khi có thay đổi và cuối arm; interrupt cố xuất checkpoint. Input/source/config phải khớp khi resume. Tải `projection-v2-cpu-prepared.zip` về `bundles/` local, rồi chạy full-pipeline từ đầu. Selection CPU V2 nằm riêng `runs/projection_v2/run_selection.json`; frozen arm outputs nằm trong `runs/<dataset>/<split>/<arm>/<fingerprint>/`.

## 4. Vertex inference và đánh giá

Preflight chỉ thống kê request thiếu. Notebook mặc định `RUN_INFERENCE=False`. Đặt True khi muốn gọi Vertex bằng ADC local; SDK/key/credential không được đưa vào cloud bundles. Model, prompt, graph và parser dùng chung giữa hai arm. Cache response chỉ reuse khi exact prompt/settings khớp.

Kết quả riêng V2: `runs/projection_v2/inference_experiment.json` và `experiment_summary.json`. Đọc F1/MCC, precision/recall, confusion matrix, invalid rate và paired CI; usage ghi request mới/attempts/cache/tokens tách biệt. Chênh lệch cache do arm baseline chạy trước không chứng minh thuật toán tự giảm LLM calls.

Sau validation, lưu report và config/checkpoint quyết định; mới đổi `splits=["test"]`, `test_protocol_frozen=true` trong config. Flag này là xác nhận protocol của người chạy, không tự kiểm chứng review validation. Không thay training params/seed theo test. Head được dùng lại khi chỉ đổi inference split; CPU bundle cần tạo lại theo split mới. Test hiện tại đã từng được xem qua các lượt H1 cũ, nên báo đúng giới hạn này.

## CLI tương đương

```bash
python -m grace_h1.portable_v2 create training_inputs --path bundles/projection-v2-training-inputs.zip
python -m grace_h1.experiment_v2 train
python -m grace_h1.portable_v2 create heads --path bundles/projection-v2-heads.zip
python -m grace_h1.portable_v2 import heads --path bundles/projection-v2-heads.zip
python -m grace_h1.experiment_v2 prepare
python -m grace_h1.experiment_v2 preflight
python -m grace_h1.experiment_v2 infer --allow-api
python -m grace_h1.experiment_v2 evaluate
```

Không chạy CLI `grace_h1.experiment all` cho V2: đó là workflow H1 dùng global selection cũ. CLI V2 mặc định đọc config V2; train/prepare/infer/evaluate là các stage riêng để tránh vô tình gọi API khi chưa xem preflight.

## Kiểm tra implementation

```bash
python -m unittest discover -s tests -p test_projection_v2.py -v
```

Kiểm tra identity/whitening, Torch↔NumPy parity, residual bound, positive/negative labels, balanced coverage, no-semi-hard finite gradients, selection gates, deterministic epoch resume, corrupted weights, train không cần test split, full prepare/selection isolation, bundle roundtrip và path traversal. Đây là numerical/integration smoke trên dữ liệu giả lập; không phải kết quả train/evaluation Devign/ReVeal thật. Notebook code cells được parse khi build. GPU/Kaggle/Colab runtime thật cần chạy notebook để đo.
