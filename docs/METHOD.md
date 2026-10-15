# Phương pháp H1 và đối chứng

## Mục tiêu

Thay representation phục vụ retrieval; giữ một lần suy luận LLM cho mỗi query cần gọi mới. Đây không phải cascade H3, không dùng GraphVerb/slicing mới, không tăng số demo. Nguồn ý tưởng: SCL-CVD (P061), COCL (P073), CodeGATNet (P001) trong bộ nghiên cứu. Không tuyên bố tái lập nguyên mô hình của các paper.

## Encoder và cache

Sáu split Devign/ReVeal có cache T5Encoder, masked_mean, max_length=512, seed=42 và cùng checkpoint content hashes. Pipeline đọc vector bằng NumPy mmap và kiểm tra ordered record_id/code_sha256. Không trộn CLS hoặc legacy encoder không có checkpoint fingerprint.

Chọn masked_mean để tái dùng đủ sáu cache có recipe đồng nhất. Baseline mới và H1 dùng cùng cache này; kết quả CLS lịch sử chỉ là tài liệu tham khảo. Trọng số encoder không cần có trên máy chạy head/retrieval.

## Học projection

Với embedding e_i, fit mean và scale trên train duy nhất, không lấy validation/test để tính normalization:

$$u_i=(e_i-\mu)/\sigma,\quad z_i=normalize(Wu_i+b),\quad W\in R^{128\times768}.$$

Head phân loại: logits=Vz_i+c (2 lớp), dùng weighted CE theo tần suất train. Projection có 98.304 weights và 128 bias; classifier thêm 258 tham số: tổng 98.690 tham số. Đây là phép đếm thiết kế, chưa là đo hardware.

$$L_{SupCon}=-\frac1{|I'|}\sum_{i\in I'}\frac1{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i^Tz_p/\tau)}{\sum_{a\ne i}\exp(z_i^Tz_a/\tau)}.$$

P(i): các sample cùng nhãn trong batch, bỏ self; I' chỉ có anchor với ít nhất một positive. Không nhân mask 0 với -infinity; không tạo loss NaN khi batch không có đủ positive.

Hard-negative term dùng negative cùng batch có cosine cao nhất:

$$L_{hard}=mean_i\,max(0,m+max_{n:y_n\ne y_i}(z_i^Tz_n)-mean_{p\in P(i)}(z_i^Tz_p)).$$

Mặc định H1: L=WCE+0.1 L_SupCon+0.1 L_hard, tau=0.1, margin=0.2. Đây là objective chuyển giao của project, không gọi là công thức nguyên bản SCL-CVD/COCL. Hard mining là in-batch, chưa có curriculum hoặc patch-pair mining.

Control CE: cùng affine projection, normalization, optimizer, split, seed và WCE, nhưng không có SupCon/hard-negative. AdamW lr=1e-3, weight_decay=1e-4, batch256, tối đa30 epoch, patience6; best checkpoint theo head validation MCC.

Train loại các nhóm code hash có nhãn mâu thuẫn và deduplicate code cùng nhãn **chỉ trong tập học head**. Validation selector bỏ code trùng train. Retrieval pool vẫn là toàn bộ train; full validation/test inference không bị giảm cohort. Các số retained/excluded ghi trong model manifest.

## Retrieval và prompt

1. Dùng candidate pool top50 có sẵn nếu ID/code khớp; nếu thiếu, tính label-blind tokens/calls/guards Jaccard với trọng số .55/.30/.15 và tie policy tương ứng nguồn hiện có.
2. Chọn top5 bằng squared L2 trên vector đã normalize (H1 hoặc whitening baseline).
3. Rerank bằng .7 lexical Jaccard + .3 AST edge sequence similarity; edit distance dùng insertion/deletion1, substitution2, giống primitive code-based đã snapshot.
4. Chọn một demo có nhãn train. Query label chỉ mở khi evaluation.
5. Graph raw prefix tối đa40 nodes/80 edges; giữ parser và fitter của baseline. Cache legacy graph được đánh dấu schema riêng; không giả vờ node IDs/CPG binary còn nguyên.
6. Target code/demo prefix theo contract, context bound dùng UTF-8 bytes bảo thủ; không gọi đó là số token đã đo. Token thực lấy từ provider nếu có.

Whitening fit covariance trên train đầy đủ, dimension256, không fit chung với query. H1 dimension128. Nếu muốn tách tác động của giảm chiều, chạy thêm whitening128 hoặc CE128; mặc định CE head đã có, chưa bật thêm LLM control để tránh tăng calls ngay từ đầu.

## Inference và đánh giá

Model/temperature/seed/output cap theo reference_contract.yaml, mặc định model Qwen trên Vertex đang dùng trong dự án. Exact request key dùng prompt hash và inference spec. Candidate/demo đổi dẫn tới prompt đổi thì không reuse response chỉ dựa record ID. Raw responses có thể reuse xuyên arm khi thực sự trùng prompt/settings.

Request lỗi không biến thành safe; parser trả None. Metrics F1/MCC/precision/recall/balanced accuracy condition trên valid predictions và luôn báo invalid rate; thêm accuracy_all và recall_all_positives. Paired bootstrap dùng cùng query valid ở hai arm, seed42,1000 lượt; chưa cluster bootstrap theo project/commit.

Ghi candidate identity, query graph hash, encoder recipe và contract hash để kiểm tra các điều kiện paired. Số graph cần cho từng arm có thể khác vì demo shortlist khác; graph của cùng query phải giống.

## Giới hạn cần giữ khi kết luận

- Checkpoint theo head MCC chỉ là proxy; classifier tốt không chứng minh demo hữu ích cho LLM.
- Binary SupCon có thể gom các CWE không liên quan và làm mất cấu trúc similarity; cần đọc kết quả CE/LLM control, không hứa gain.
- Projection không khôi phục thông tin bảo mật encoder đã đánh mất; fine-tune encoder/LoRA là nghiên cứu khác, chưa triển khai ở đây.
- H1 không mặc định giảm LLM calls. Tiết kiệm ở cache reuse được báo riêng với lợi ích thuật toán.
- Một seed ban đầu, chưa có kết quả mới. Nên xác nhận nhiều seed và thêm ReVeal sau full Devign.
- Vẫn có giới hạn function-level, graph parsing và prefix budget của baseline; việc sửa implementation cũ được để sau theo chỉ đạo.
