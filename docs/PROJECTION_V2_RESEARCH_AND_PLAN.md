# Projection V2: nghiên cứu, quyết định và kế hoạch triển khai

Ngày: 17/10/2026. Nhánh: `projection-v2-local-metric`; xuất phát từ mốc ReVeal `b8964c6`.

## 1. Câu hỏi nghiên cứu và bằng chứng hiện có

Học một representation dùng để chọn demonstration hữu ích hơn cho GRACE với cache CodeT5 hiện có. Chất lượng cuối cùng là F1/MCC của GRACE + LLM trên cùng cohort, prompt, graph và generation settings. Điểm classifier phụ và độ khớp nhãn của retrieval chỉ là proxy.

H1 cũ đã là một mạng neural nhưng chỉ có affine projection 768→128, L2 normalization và classifier 128→2. Chuyển sang mạng sâu hơn là tăng năng lực biểu diễn, không tự bảo đảm tăng chất lượng. Test Devign: whitening F1 35,73%, MCC 0,0587; H1 F1 34,33%, MCC 0,0280. Test ReVeal: whitening 17,11%/0,0318; H1 16,61%/0,0246. ReVeal test có 230 vulnerable / 2.044 safe; Devign 1.255 / 1.477. Các điểm đã xem là test lịch sử; không dùng nhãn hoặc response test để tạo cặp train, chọn checkpoint hay chọn hyperparameter V2.

Ba giả thuyết cần kiểm chứng: (a) projection tuyến tính 128d có thể làm mất cấu trúc semantic; (b) kéo mọi hàm cùng nhãn nhị phân vào một cụm có thể phá neighborhood chức năng; (c) random batch + hardest negative duy nhất và chọn checkpoint theo classifier MCC chưa khớp mục tiêu retrieval. Đây là giả thuyết, chưa là kết luận nguyên nhân từ ablation.

## 2. Đối chiếu nguồn chính

PDF mới được lưu trong `D:/JUNIOR/LabThayTho/Others_papers/Projection-Training/`; URL, SHA-256, số trang trong `projection_v2_sources.json`. Các trang dưới đây là trang PDF.

| Nguồn | Bằng chứng đọc | Giá trị áp dụng | Giới hạn chuyển giao |
|---|---|---|---|
| [SCL-CVD, 2024](https://doi.org/10.1016/j.cose.2024.103994), P061 có sẵn | Trang 5–6: SCL+NLL, R-Drop, GraphCodeBERT+LoRA, MLP; Table 4 trang 14: bỏ contrastive làm F1 ReVeal 47,11→43,11 | Loss và mạng phi tuyến là hai trục riêng cần control | Paper fine-tune encoder/DFG; cached projection của ta không tái lập paper, không được lấy gain classification làm gain GRACE |
| COCL, P073 có sẵn | Trang 4–5: độ khó động theo cosine, curriculum; Table VII: Devign detector 0,553→0,619 với static curriculum | Bật dần loss semi-hard sau warmup, quan sát tỉ lệ cặp active | Không sao chép augmentation code có thể đổi hành vi; không tuyên bố tái lập COCL |
| [SupCon, NeurIPS 2020](https://papers.nips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) | Trang 2–5: same-class positives, nonlinear contrastive head, dùng representation layer khác cho downstream; ResNet-200 ImageNet top1 81,4% | Phải tách mục tiêu classification, contrastive embedding và retrieval usefulness | Nhãn safe/vulnerable chỉ có 2 lớp, chứa nhiều cơ chế/chức năng; local positives là thích nghi của project |
| [Sampling Matters, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf) | Trang 3–4: hard nhất gây gradient nhiễu/collapse; semi-hard có thể hết cặp active; distance-weighted sampling phân bố độ khó rộng | Local candidate bank + InfoNCE trên nhiều negative; semi-hard chỉ là term phụ có warmup | Ta không triển khai distance-weighted distribution của paper; không suy ra paper chứng minh lỗi của H1 |
| [BCL, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf) | Trang 3–4: gradient head-class domination, class averaging/complement | Balanced anchor batches, metric macro theo lớp; tránh vừa oversample vừa inverse-frequency CE | Balanced sampler không phải BCL đầy đủ; không dùng prototypes/class-complement loss |
| [Similarity-Preserving KD, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf) | Trang 3, Eq.2–5: Gram matrix, row normalization, squared Frobenius loss; trang 4: student error giảm 0,5–1,2 điểm trên CIFAR10 setups | Geometry loss giữ pairwise cosine gần teacher whitening, skip connection bắt đầu từ baseline | Cosine-Gram MSE của ta là thích nghi; không phải nguyên Eq.4 row-normalized và không dùng image teacher |
| [RKD, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf) | Quan hệ distance/angle giữa ví dụ thay vì chỉ logits | Củng cố việc giữ quan hệ semantic khi thích nghi | Không triển khai angle-wise RKD; lựa chọn cosine-Gram rẻ hơn |
| [EPR, NAACL 2022](https://aclanthology.org/2022.naacl-main.191.pdf) | Trang 3–4: LM score ứng viên theo P(y|demo,x), train dense retriever; semantic parsing BREAK/MTOP/SMCalFlow | Mục tiêu tốt nhất dài hạn là utility của demo đối với LLM; cần downstream validation trước test | Cached response hiện tại chỉ có một demo/query, không tạo được counterfactual utility labels. Không suy diễn cùng nhãn = demo hữu ích. Huấn luyện theo LLM utility cần thêm dữ liệu train-only có nhiều demo/query |
| [Imbalance loss study, 2025](https://doi.org/10.1016/j.eswa.2025.128504), P057 có sẵn | Trang 8–9 setup: 2 detectors, 4 datasets, 5 lần chạy, 9 imbalance losses + CE; khuyến nghị phụ thuộc độ lệch lớp | Báo cáo theo lớp, nhiều seed và control loss | Focal/logit-adjustment không mặc định tốt hơn cho retrieval; chưa đưa thêm loss trước khi có control |

Đã kiểm tra trực quan trang công thức SPKD. Nguồn classification/image metric learning chỉ cung cấp nguyên lý và control; mọi lựa chọn kết hợp bên dưới là đề xuất thực nghiệm của project.

## 3. Quyết định mô hình: residual MLP trên teacher whitening

Ưu tiên cached embedding để dùng GPU T4 hoặc CPU mà không encode CodeT5 lại. CNN/GNN cần token/graph train và LoRA cần encoder forward: giữ làm hướng tiếp theo nếu cached representation vẫn không đủ. MLP phù hợp với vector 768d hiện có.

Teacher `t=normalize((e−mu_w) W_w)` là whitening 256d fit trên toàn bộ train, không nhìn validation/test. Residual branch dùng z-score train và hai affine layers: `768→256→GELU→dropout(0.1)→256`. Đầu ra:

```
u = (e - mu_train) / sigma_train
r = W2 GELU(W1 u + b1) + b2
delta = r / (1 + ||r||2)
z = normalize(t + 0.25 delta)
logits = classifier(z)  # 256→2, chỉ hỗ trợ train/selection
```

W2/b2 khởi tạo zero: model bắt đầu từ teacher. Residual có norm dưới 0,25 nên adaptation có giới hạn; đây là ràng buộc thiết kế, không bảo đảm thứ tự top-k không đổi. Tổng mặc định 263.170 tham số gồm classifier. Dùng cùng dimension256 với baseline để không gộp tác động dimension128 vào so sánh mới. Runtime xuất NPZ và inference projection bằng NumPy (GELU tanh), không đòi PyTorch/GPU ở CPU preparation.

## 4. Quyết định thuật toán train

- Deduplicate code cùng nhãn, loại nhóm code có nhãn mâu thuẫn, giống H1. Validation selector loại code trùng train; full inference giữ nguyên cohort.
- Fit normalization và teacher bằng train. Mining chỉ dùng train unique/unambiguous; không đọc test tại stage train.
- Precompute nearest bank bằng chunked cosine trên teacher: 8 positive cùng nhãn nhưng gần về semantic, 32 negative khác nhãn. Không lưu ma trận N×N; cache bank theo source/config/input fingerprint.
- Mỗi epoch duyệt majority một lần, oversample minority theo các vòng permutation; anchor batch128 có tỷ lệ nhãn 1:1. CE không inverse-frequency weighting nữa để tránh bù lệch lớp hai lần.
- Mỗi anchor lấy 1 positive trong local top8 và 8 negative từ bank32. Loss local InfoNCE dùng positive + negative đã biết nhãn; không ép mọi sample safe/vulnerable ở xa chức năng thành positive.
- Loss `CE + 0.1 local_InfoNCE + 0.05 ramp(epoch) semi_hard + 1.0 geometry_MSE`. Semi-hard dùng negative có `sim_pos−margin < sim_neg < sim_pos`, margin0,2; khi không có cặp thì term bằng0, InfoNCE vẫn có gradient. Warmup3 epoch, ramp5 epoch. Geometry MSE trên cosine Gram của batch anchor so với teacher, không có nhãn.
- AdamW lr3e-4, weight_decay1e-3, gradient clip1, cosine LR, tối đa30 epoch, patience8, seed42. Mỗi epoch có RNG riêng; resume giữ optimizer/scheduler/torch CPU+CUDA RNG để dropout tiếp nối đúng.

## 5. Chọn checkpoint và đánh giá

Thay classifier MCC bằng retrieval proxy trên validation độc lập: macro label Precision@5 trong cùng common top50 mà pipeline thực dùng. Báo cáo per-class P@5, hit@5, overlap top5 với teacher, classifier MCC riêng. Label relevance chỉ là proxy, không phải utility LLM. Chọn learned checkpoint khi macro P@5 tăng ít nhất0,002 so teacher, per-class P@5 không giảm hơn0,02 và teacher overlap ít nhất0,5. Nếu không đạt, xuất identity fallback và ghi rõ status; không tuyên bố head được cải tiến.

Control cần có: whitening256 identity; residual_ce (CE+geometry, cùng model/sampling/selection); residual_local (đầy đủ). H1 linear cũ giữ làm historical reference. Các term đều cấu hình được để ablate locality, semi-hard, geometry; không chọn cấu hình bằng test. Seed42 dùng để giữ phép so sánh lịch sử; seed43/44 là replication offline, báo cáo riêng, không chọn seed tốt nhất rồi gọi đại diện.

Sau offline validation: chạy full GRACE validation baseline/residual với prompt, graph, K=5, lexical/AST và LLM settings giống nhau; mặc định dùng cache và không gọi API mới. Bật API trong notebook khi chủ động chạy. Chỉ chuyển sang test sau khi freeze config/checkpoint dựa trên validation. Test là cohort đã từng xem kết quả, không gọi là test hoàn toàn chưa chạm. Báo paired bootstrap theo query và usage/cache riêng; nhiều seed/cluster CI là bước xác nhận tiếp theo.

## 6. Triển khai trên branch mới

1. Commit plan này trước khi viết implementation.
2. Thêm module trainer/NPZ NumPy projection V2 riêng; giữ H1 linear để đối chiếu. Cache/checkpoint/selection mới trong `artifacts/projection_v2/` và `cache/projection_v2/`.
3. Tích hợp arm `h1_residual_ce`/`h1_residual_local` vào prepare; selection/inference/evaluation của workflow V2 tách trong `runs/projection_v2/`. V2 không ghi đè `runs/run_selection.json` hoặc training selection H1.
4. Notebook `full-pipeline.ipynb` vẫn là điểm vào. Đặt chỉ dẫn V2 đầu notebook tới `projection-v2.ipynb`. Notebook mới điều phối train ở Kaggle, CPU local/Colab, rồi inference/evaluation local. Train bundle chỉ có train/validation embeddings, không có response/test embedding/credential.
5. Tạo Kaggle notebook và Colab CPU notebook V2, zip có manifest+SHA và kiểm tra source/config khi import. Tái sử dụng embeddings, candidate pool và graph/response cache có sẵn. Không copy caches lớn vào Git.
6. Kiểm tra có ý nghĩa: identity initialization vs whitening, Torch↔NumPy parity, norm bound, local bank loại self/sai nhãn, class balance, semi-hard edge cases/finite gradient, baseline fallback, deterministic resume, artifact corruption, bundle roundtrip/path safety và pipeline integration trên fixture.
7. Smoke synthetic train + prepare/inference cache-only; không dùng smoke làm benchmark. Build input zip và commit từng nhóm implementation/notebook/docs sau kiểm tra byte theo skill.

## 7. Điều kiện hoàn tất lượt implement

- Có plan/source provenance đã commit trước code; module train/selection/resume chạy được và export projection không cần torch ở preparation.
- Có entry notebook và hướng dẫn chuyển môi trường rõ, input/output bundle V2 không ghi đè kết quả H1, hỗ trợ cả Devign/ReVeal.
- Các kiểm tra mục6 pass; chưa claim tăng benchmark cho tới khi chạy validation và test paired. Kết quả nghiên cứu không đồng nghĩa đã train/evaluate full dataset.
