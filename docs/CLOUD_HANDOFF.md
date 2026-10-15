# Chạy qua local, Kaggle và Colab

## Điểm bắt đầu duy nhất

Mở **D:/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/notebooks/full-pipeline.ipynb** trước. Notebook này điều phối toàn bộ; tên full_pipeline.ipynb cũ chỉ còn link chuyển đến tên mới.

| Stage | Môi trường mặc định | Công việc |
|---|---|---|
| 1 | Kaggle GPU | Train CE control và SupCon + hard negatives trên embedding cache |
| 2 | Colab CPU, accelerator None | Whitening train-only, transform vectors, candidates, top5, Joern missing graphs, lexical/AST rerank, freeze prompts |
| 3 | Local | Preflight, Vertex inference với ADC/response cache sẵn có |
| 4 | Local | Tổng hợp metrics và paired bootstrap; phần này nhỏ, dùng predictions vừa nhận |

Notebook local dừng chủ động ở Bước 1/Bước 2 nếu thiếu artifact khớp. Sau mỗi lần tải output về bundles/, chạy Run All lại từ đầu. Không tự train hoặc prepare local khi hai cờ cloud còn bật.

## Các bundle đầu vào đã đóng gói

Trong bundles/ của project:

| File | Mang đến đâu | Nội dung |
|---|---|---|
| grace-h1-training-inputs.zip | Kaggle Dataset | Code/config, train/validation canonical JSONL, bốn embedding sets; không có test payload |
| grace-h1-cpu-inputs.zip | Colab | Code/config, canonical JSONL và sáu embedding sets Devign/ReVeal, graph features, legacy/common candidate caches, projected cache nếu đã có |
| grace-h1-joern-runtime.zip | Colab | Snapshot Joern 4.0.623 đang cài ở WSL: launchers, bin/lib/conf và frontend C/C++ |

Kích thước/checksum thực ghi trong provenance/cloud_bundles.json. Các zip lớn được ignore theo đường dẫn cụ thể và giữ trên đĩa. Không có .env, API keys, ADC, raw responses hay CPG binaries cũ trong các bundle input. CPU bundle có query code/labels để giữ đúng snapshot/hash; retrieval mở query bằng labels=False.

Head chưa train trong lượt bàn giao. grace-h1-trained-heads.zip được tạo sau khi chạy Kaggle; không có file weights giả thay thế. Cả input bundle CPU và Kaggle chứa hai dataset nhưng experiment mặc định chỉ chạy full test Devign.

## Lần chạy đầu tiên

### 1. Local → Kaggle

Mở full-pipeline.ipynb, Run All. Bước 1 dừng khi chưa có matching heads, đồng thời tạo/cập nhật training-inputs.zip.

Upload kaggle_train_projection.ipynb vào Kaggle, bật GPU/Internet, đính kèm training-inputs.zip dưới dạng Dataset. Notebook hỗ trợ zip còn nguyên hoặc Kaggle tự giải nén. Run All và tải grace-h1-trained-heads.zip từ Outputs về bundles/ local.

### 2. Quay lại local → Colab

Run All full-pipeline.ipynb lần nữa. Bước 1 nhập head; Bước 2 dừng khi chưa có matching prepared output, đồng thời tạo/cập nhật cpu-inputs.zip.

Tạo thư mục MyDrive/GRACE-H1-transfer trên Google Drive và upload:

1. grace-h1-cpu-inputs.zip.
2. grace-h1-joern-runtime.zip.
3. grace-h1-trained-heads.zip vừa tải từ Kaggle.

Upload colab_cpu_prepare.ipynb vào Colab. Chọn Runtime → Change runtime type → Hardware accelerator **None**. Run All, mount Drive và giữ TRANSFER đúng thư mục trên. Có thể đặt USE_GOOGLE_DRIVE=False và upload zip qua Files pane vào /content/transfer; khi đó phải tải outputs/checkpoint trước khi runtime mất.

Colab chỉ cài requirements-cpu.txt và Java 21, không cài Torch/CUDA/CodeT5 hoặc SDK LLM. Graph features cũ được reuse; Joern chỉ xử lý graph thiếu. JOERN_HOME đặt qua environment tại Colab, giữ pipeline.joern_home=null trong experiment để đường dẫn runtime không đổi config fingerprint.

### 3. Colab → local LLM

Sau khi Colab hoàn tất, tải **grace-h1-cpu-prepared.zip** từ thư mục transfer về bundles/ local. Giữ trained-heads.zip ở đó.

Run All full-pipeline.ipynb lần nữa: nhập heads + prepared bundle, kiểm tra nguồn/config/data/weights/freeze, rồi chuyển đến preflight và inference. Không gọi lại preparation local. RUN_INFERENCE=True sẽ gọi Vertex cho các exact prompts chưa có response; False chỉ sử dụng cache.

Metrics và paired comparison được ghi vào runs/experiment_summary.json sau inference. Không cần mang predictions lên Colab để làm phần tổng hợp nhỏ này.

## Output Colab và resume

- cpu-prepared.zip: các run đã freeze, retrieval/prompts/graph index, run_selection.json, candidate/projected cache và graph mới phát sinh. Graph có sẵn trong input không được đóng gói lại. Không có response hoặc credential.
- cpu-checkpoint.zip: candidate/projected cache, graph mới và các run đã freeze tại thời điểm checkpoint; không phải output hoàn tất để local gọi LLM. Colab tự nhập checkpoint này khi bắt đầu lại.
- Checkpoint được ghi ở mốc mỗi500 graph đã xử lý nếu có graph mới từ lần checkpoint trước (mốc đầu lưu candidate/projected cache), sau mỗi arm mới hoàn tất và trong finally khi bấm Interrupt.
- Các file làm việc nằm trong /content để đọc hàng nghìn graph nhanh hơn; chỉ zip được copy về Drive. VM bị xóa đột ngột có thể mất phần sau checkpoint cuối. Đừng chạy hai coordinator cùng một thư mục transfer.
- Dùng runtime mới nếu đổi input/config/source; đổi tên/thư mục transfer cho experiment mới để không tự nhập checkpoint cũ. Không dùng thủ thuật duy trì runtime; Colab có giới hạn tài nguyên và thời gian thay đổi.
- Import kiểm tra toàn bộ member checksum trước khi publish. File hiện có khác nội dung bị chặn; hãy dùng project/runtime mới cho experiment khác hoặc đổi tên output bundle cũ khỏi đường dẫn mặc định. Không ghi đè một run đã freeze.

## Khi đổi cấu hình

Sửa configs/experiment.json ở local **trước khi xuất bundle**. Ví dụ datasets=[devign,reveal], thêm h1_ce vào llm_arms hoặc giảm graph_workers. Tạo lại input zip tương ứng; giữ training config giống nhau giữa Kaggle và local. Nếu thay code training/data/metrics, head cũ không khớp và phải train lại. Thay source preparation làm prepared output cũ không khớp.

Không sửa trực tiếp config ở Colab sau khi unpack; input inventory khóa config/source/data để tránh trả về kết quả của experiment khác.

Nếu chủ động muốn chạy tất cả local, đặt TRAIN_ON_KAGGLE=False và PREPARE_ON_COLAB=False ở cell đầu. Khi đó local cần PyTorch, RAM và Joern như RUNBOOK. Luồng mặc định dùng cloud cho các bước nặng.

## Đóng gói lại bằng CLI

Chạy ở project root local:

~~~bash
python tools/bundle.py training-inputs
python tools/bundle.py cpu-inputs
python tools/bundle.py import-heads --path bundles/grace-h1-trained-heads.zip
python tools/bundle.py import-prepared --path bundles/grace-h1-cpu-prepared.zip
~~~

Đóng gói Joern runtime từ kernel/shell WSL, không phải Windows Python:

~~~bash
python3 /mnt/d/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/tools/bundle.py joern-runtime --joern-home /home/yungn/tools/joern/y/joern-cli
~~~

Runtime bundle giữ executable bits, version đọc từ jar tên io.joern.joern-cli-4.0.623.jar. Java 21 là major version đã đọc từ môi trường WSL nguồn. Không tự đổi sang latest khi chuyển Colab.

## Nguồn đối chiếu

[Colab FAQ chính thức](https://research.google.com/colaboratory/faq.html) mô tả upload notebook, VM có thể bị xóa và giới hạn tài nguyên không cố định. [Joern installation](https://docs.joern.io/installation/) mô tả binary distributions và JVM heap flags; project dùng snapshot cài tại lab cùng Java major version thực tế thay vì suy ra phiên bản từ ví dụ trong tài liệu.

## Trạng thái

Đã tạo notebook và đóng gói input/runtime. Chưa chạy notebook Colab/Kaggle, preparation, training, LLM hoặc tests trong lượt này; chưa xác nhận runtime cloud hay có benchmark mới.
