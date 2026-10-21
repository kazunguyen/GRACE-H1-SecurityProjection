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

Chỉnh `configs/projection_v2.json`. Cấu hình hiện tại dataset `reveal`, split inference `test` với `test_protocol_frozen=true`; train objectives `residual_ce` và `residual_local` đã chọn checkpoint bằng validation proxy; LLM arms baseline + residual_local. Đổi `datasets` sang `["devign"]` hoặc cả hai trước khi tạo training bundle; input chỉ chứa dataset được chọn. Pipeline H1 cũ vẫn giữ cấu hình riêng `configs/experiment.json`.

Ở local hiện có venv, embeddings/canonical datasets, graph/candidate/response cache và Joern runtime zip. Notebook full-pipeline dùng các đường dẫn này trực tiếp. Nếu chuyển máy, cài `requirements.txt`; CPU preparation chỉ cần `requirements-cpu.txt`, train chỉ cần `requirements-training.txt` (GPU Kaggle có torch sẵn).

## 2. Kaggle training

1. Cell Bước1 của full-pipeline kiểm tra head. Nếu chưa có, nó tạo `bundles/projection-v2-training-inputs.zip` rồi dừng có chỉ dẫn.
2. Upload zip vào Kaggle dataset/input; mở `notebooks/kaggle_train_projection_v2.ipynb`, bật GPU T4 và Internet, Run All.
3. Log mỗi epoch ghi loss, macro label P@5 và top5 overlap. `last.pt` chứa model/optimizer/scheduler/RNG. Khi interrupt, finally xuất `projection-v2-training-checkpoint.zip`; gán `RESUME_ZIP` ở lần tiếp theo để import phần đã xong. Runtime bị xóa đột ngột trước khi export thì chỉ checkpoint/Output đã lưu ra ngoài còn dùng được.
4. Tải `projection-v2-heads.zip` vào `bundles/` local. Bundle CPU được notebook WSL local nhập trực tiếp. `EXTRA_SEEDS=[43,44]` chạy replication offline và khôi phục selection seed42; không chọn seed tốt nhất bằng test. Replication summary được lưu kèm artifact.
5. Chạy lại full-pipeline. Nếu arm yêu cầu có `baseline_fallback`, mặc định dừng trước CPU parsing/LLM. Đọc history, sửa config dựa trên validation và train lại. Muốn đo identity control thì bật `allow_identity_fallback_run` có chủ đích.

Training artifact nằm trong `artifacts/projection_v2/<dataset>/<objective>/<fingerprint>/`; selection V2 không ghi đè selection H1. File trọng số đã có final manifest là immutable; checkpoint chưa hoàn tất có thể được cập nhật sau khi xác minh toàn bộ archive.

## 3. CPU local offline trên WSL

Mở `notebooks/local_cpu_prepare_v2.ipynb` bằng kernel WSL. Notebook nhập các bundle local vào staging workspace dưới `/home`, dùng cache sẵn có, chỉ parse graph còn thiếu, rồi xuất checkpoint/prepared bundle về `bundles/` trên ổ D. Không có pip/apt install hay truy cập mạng ở bước này. Cần dependencies đã cài trong kernel, Joern C/C++ runtime local và Java21. Cache graph được lưu ngay trong workspace nên có thể chạy lại sau khi ngắt.

Config mặc định3 workers, heap2GB mỗi Joern process. Progress bar hiển thị cache hit, graph parse mới và record đang chạy. Sau khi notebook local tạo `projection-v2-cpu-prepared.zip`, chạy lại `full-pipeline.ipynb`; nó import prepared bundle tại chỗ. Selection V2 nằm riêng `runs/projection_v2/run_selection.json`; frozen outputs nằm trong `runs/<dataset>/<split>/<arm>/<fingerprint>/`.

`colab_cpu_prepare_v2.ipynb` vẫn được giữ để đường dẫn cũ không hỏng, nhưng hiện chỉ chuyển tới notebook WSL local.

## 4. Vertex inference và đánh giá

Preflight không gọi API. Notebook hiện bật `RUN_INFERENCE=True` và `FORCE_FRESH_INFERENCE=True` cho lượt chạy lại: gửi request mới cho mọi prompt unique ở từng arm, bỏ qua cache cũ và response mà arm trước vừa tạo. Dự kiến khoảng 2.270 prompt unique mỗi arm; các lần retry có thể làm số API attempts cao hơn. Muốn chạy cache-aware ở lượt khác, đặt `FORCE_FRESH_INFERENCE=False`. Dùng Vertex bằng ADC local; SDK/key/credential không được đưa vào bundles.

Kết quả riêng V2: `runs/projection_v2/inference_experiment.json` và `experiment_summary.json`. Đọc F1/MCC, precision/recall, confusion matrix, invalid rate và paired CI; usage ghi request mới/attempts/cache/tokens tách biệt. Chênh lệch cache do arm baseline chạy trước không chứng minh thuật toán tự giảm LLM calls.

Head và cấu hình train đã được chốt sau lựa chọn bằng validation retrieval proxy. Config hiện chuyển sang `splits=["test"]`, `test_protocol_frozen=true` cho lượt benchmark cuối; không thay training params/seed/model theo test. Nếu chỉ đổi inference split, head được dùng lại vì training identity không đổi; CPU bundle được tạo lại để đóng gói test embeddings và graph cache. Test đã từng được xem qua các lượt H1 cũ, nên báo đúng giới hạn này.

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

Kiểm tra identity/whitening, Torch↔NumPy parity, residual bound, positive/negative labels, balanced coverage, no-semi-hard finite gradients, selection gates, deterministic epoch resume, corrupted weights, train không cần test split, full prepare/selection isolation, bundle roundtrip và path traversal. Đây là numerical/integration smoke trên dữ liệu giả lập; không phải kết quả train/evaluation Devign/ReVeal thật. Notebook code cells được parse khi build. GPU/Kaggle và WSL local runtime thật cần chạy notebook để đo.
