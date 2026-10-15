# Thứ tự triển khai

1. Khởi tạo project và quy tắc commit.
2. Copy dataset/cache có provenance; chuẩn hóa ID và embedding recipe; không sửa nguồn cũ.
3. Học projection train-only, chọn checkpoint bằng validation; có control CE và baseline whitening.
4. Retrieval chung top50 → top5 semantic → lexical/AST → một demo → graph/prompt → inference/resume → metrics paired.
5. Notebook full pipeline local và notebook Kaggle GPU, bundle/import/export portable, hướng dẫn vận hành.

Mục tiêu H1: thay representation retrieval; giữ graph/prompt/LLM giống nhau giữa baseline và H1. Có cache masked_mean đồng nhất của cả sáu split Devign/ReVeal nên ưu tiên recipe này cho paired experiment mới. Không ngầm coi số liệu baseline CLS cũ là paired comparator.

Không chạy training, benchmark hoặc API trong lượt authoring này; notebook cung cấp các stage đó để Run All. Không thêm/chạy tests khi người dùng chưa yêu cầu.
