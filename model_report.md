# Model Report — Credit Card Fraud Detection

## 1. Bài toán

Phân loại nhị phân: dự đoán một giao dịch thẻ tín dụng có phải fraud hay không,
dựa trên 30 feature (`Time`, `V1`-`V28` đã PCA-transform, `Amount`).

Dataset: 284,807 giao dịch, **fraud chỉ chiếm 0.173%** (492 dòng) — tỷ lệ ~1:577.
Đây là đặc điểm chi phối toàn bộ quyết định kỹ thuật trong report này: **Accuracy
không được dùng làm metric đánh giá** vì baseline "toàn bộ dự đoán Legit" đã đạt
99.8% accuracy mà vô dụng trên thực tế. Metric chính: **Precision, Recall, F1,
PR-AUC**.

## 2. Data Understanding & EDA (tóm tắt)

- Không có missing value; có 1,081 dòng duplicate (0.38%) — drop **sau** khi
  train/test split để tránh data leakage.
- `Amount` lệch phải mạnh (mean 88.3, max 25,691) — không phải outlier lỗi, mà
  là đặc tính tự nhiên của dữ liệu tiền tệ; giữ nguyên, xử lý bằng log-transform
  thay vì loại bỏ.
- `Time` (giây, trải dài 2 ngày) cho thấy fraud xảy ra tỷ lệ cao bất thường vào
  các khung giờ ít giao dịch (ban đêm).
- Phân tích tách biệt (separation score) + correlation xác định top feature
  phân tách fraud/legit mạnh nhất: **V17, V14, V12, V10, V16**.

## 3. Feature Engineering

| Feature | Xử lý | Lý do |
|---|---|---|
| `Amount` → `Amount_log` | `log1p()` | Giảm lệch phân phối |
| `Time` → `Time_hour` | `(Time % 86400)/3600` | Giữ tín hiệu "giờ trong ngày", bỏ giây tuyệt đối dễ overfit vào đúng 2 ngày thu thập |
| Toàn bộ 30 feature | `RobustScaler` (fit trên train only) | Ít bị ảnh hưởng bởi giá trị cực lớn hơn StandardScaler |

## 4. Xử lý Imbalanced Data

So sánh 4 chiến lược bằng Logistic Regression (model thăm dò nhanh), cùng 1 test
set cố định (không resample):

| Chiến lược | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Baseline (không xử lý) | 0.829 | 0.649 | 0.728 | 0.733 |
| SMOTE | 0.053 | 0.918 | 0.101 | 0.723 |
| Class Weight (balanced) | 0.053 | 0.907 | 0.100 | 0.710 |
| Random Undersampling | 0.046 | 0.907 | 0.087 | 0.653 |

**Nhận xét:** SMOTE/Class Weight/Undersampling đều tăng Recall mạnh nhưng làm
Precision sập xuống ~5% (95% cảnh báo là báo động giả) — do resampling làm lệch
threshold 0.5 mặc định. Random Undersampling tệ nhất vì vứt bỏ 226k+ dòng Legit,
model học từ quá ít dữ liệu.

**Quyết định:** không resample dữ liệu. Mỗi model tự xử lý imbalance qua tham số
riêng — `class_weight='balanced'` (Logistic Regression/Decision Tree/Random
Forest) và `scale_pos_weight` (XGBoost, tính = số Legit/số Fraud trong train).

## 5. Kết quả huấn luyện 4 model (threshold mặc định 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost (tuned)** | 0.9995 | 0.890 | 0.835 | **0.862** | 0.977 | **0.880** |
| XGBoost (baseline) | 0.9995 | 0.880 | 0.835 | 0.857 | 0.968 | 0.879 |
| Random Forest | 0.9994 | 0.830 | 0.804 | 0.817 | 0.978 | 0.836 |
| Logistic Regression | 0.9721 | 0.053 | 0.907 | 0.100 | 0.971 | 0.710 |
| Decision Tree | 0.9889 | 0.113 | 0.804 | 0.198 | 0.901 | 0.459 |

**Nhận xét quan trọng:** cột Accuracy gần như không phân biệt được model tốt/tệ
(chênh lệch chỉ 2-3%), trong khi F1 lệch nhau tới 8 lần (0.10 vs 0.86) — bằng
chứng cụ thể cho thấy Accuracy không phù hợp với bài toán imbalanced này.

**XGBoost thắng** vì gradient boosting tối ưu trực tiếp qua hàm loss có trọng số
(`scale_pos_weight`), phù hợp dữ liệu dạng bảng, trong khi `class_weight` của
3 model kia áp trọng số cực đoan (~577x) một cách thô, dễ làm model "hoảng loạn"
với model tuyến tính/cây đơn.

## 6. Hyperparameter Tuning

`RandomizedSearchCV` (30 tổ hợp × 5-fold Stratified CV, scoring=`average_precision`)
cho XGBoost. Best params: `n_estimators=444, max_depth=9, learning_rate=0.051,
subsample=0.995, colsample_bytree=0.712, min_child_weight=1, gamma=0.271`.

Cải thiện so với baseline: PR-AUC +0.0011 (không đáng kể) — cấu hình mặc định
của XGBoost vốn đã khá tốt cho bài toán này; tuning cho thấy có tồn tại cấu hình
nhỉnh hơn nhưng không phải yếu tố quyết định thành công của model.

## 7. Chọn Threshold

Threshold mặc định 0.5 không phải lựa chọn tối ưu. Quét toàn bộ Precision-Recall
curve, chọn điểm F1 cao nhất:

- **Threshold tối ưu: 0.549**
- Precision: 0.900 | Recall: 0.835 | F1: 0.866 (so với F1=0.862 ở threshold 0.5)

## 8. Explainability

**Feature Importance (XGBoost built-in), top 5:** V14 (44.6%), V10 (13.3%),
V12 (9.1%), V4 (4.9%), V17 (2.9%). V14 chiếm ưu thế áp đảo — dấu hiệu các feature
mạnh có thông tin chồng lấn, cây "chọn" V14 làm trục chính ở các lần split đầu.

**SHAP — chiều tác động:**
- V14, V10, V12: giá trị **càng thấp (âm)** → xác suất fraud càng cao.
- V4, V11: giá trị **càng cao** → xác suất fraud càng cao (ngược chiều nhóm trên).
- `Amount_log`, `Time_hour` (2 feature tự tạo) lọt top 10 SHAP importance — xác
  nhận feature engineering có tác dụng thực sự, không chỉ là bước hình thức.

**Đối chiếu:** top feature từ EDA (V17, V14, V12, V10, V16) khớp phần lớn với
top feature importance và SHAP (V14, V10, V12, V4, V17) — 3 phương pháp độc lập
đồng thuận, là bằng chứng model học đúng tín hiệu thật, không phải nhiễu.

## 9. Kết luận & Giới hạn

- Model cuối: **XGBoost (tuned)**, PR-AUC 0.880, F1 0.866 tại threshold 0.549.
- Giới hạn: V1-V28 là PCA-anonymized nên không giải thích được ý nghĩa nghiệp vụ
  cụ thể (chỉ biết "feature nào quan trọng", không biết "tại sao về mặt kinh
  doanh"). Dataset chỉ trải dài 2 ngày — chưa kiểm chứng được model có ổn định
  theo thời gian dài hơn (concept drift) hay không.
- Không dùng `label_encoder.pkl` — `Class` vốn đã là số 0/1, không có biến
  categorical cần encode trong bài toán này.
