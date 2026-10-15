# Thứ tự triển khai

1. Khởi tạo project và quy tắc commit.
2. Copy dataset/cache có provenance; chuẩn hóa ID và embedding recipe; không sửa nguồn cũ.
3. Học projection train-only, chọn checkpoint bằng validation; có control CE và baseline whitening.
4. Retrieval chung top50 → top5 semantic → lexical/AST → một demo → graph/prompt → inference/resume → metrics paired.
5. Notebook full pipeline local và notebook Kaggle GPU, bundle/import/export portable, hướng dẫn vận hành.

Mục tiêu H1: thay representation retrieval; giữ graph/prompt/LLM giống nhau giữa baseline và H1. Có cache masked_mean đồng nhất của cả sáu split Devign/ReVeal nên ưu tiên recipe này cho paired experiment mới. Không ngầm coi số liệu baseline CLS cũ là paired comparator.

Không chạy training, benchmark hoặc API trong lượt authoring này; notebook cung cấp các stage đó để Run All. Không thêm/chạy tests khi người dùng chưa yêu cầu.

## Trạng thái bàn giao

Đã hoàn thành năm nhóm triển khai ở trên: workspace Git riêng; snapshot dataset/cache; projection và controls; pipeline retrieval–graph–inference–evaluation; hai notebook và bundle Kaggle. Repo không có origin. Bundle training input đã tạo sẵn; provenance/training_bundle.json ghi kích thước và SHA-256.

Chưa thực thi notebook, huấn luyện head, gọi LLM hoặc đo benchmark mới. Bước sử dụng tiếp theo là mở notebooks/full_pipeline.ipynb bằng kernel WSL hiện có và Run All; hoặc train head bằng notebook Kaggle rồi import về local.
