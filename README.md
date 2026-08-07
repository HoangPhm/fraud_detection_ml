# Credit Card Fraud Detection — ML Pipeline

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, mlg-ulb) —
284,807 giao dịch thẻ tín dụng của khách hàng châu Âu trong 2 ngày (tháng 9/2013).

- **Time**: số giây kể từ giao dịch đầu tiên
- **V1–V28**: 28 feature đã qua PCA transform (ẩn danh hóa, không rõ ý nghĩa gốc)
- **Amount**: số tiền giao dịch
- **Class**: 0 = hợp lệ, 1 = fraud (target)

**Đặc điểm quan trọng nhất:** dữ liệu mất cân bằng cực đoan — fraud chỉ chiếm
**0.173%** (492/284,807 dòng), tỷ lệ ~1:577. Toàn bộ pipeline được thiết kế
xoay quanh đặc điểm này (xem `docs/business_understanding.md`).

## Feature Engineering

| Feature gốc | Xử lý | Lý do |
|---|---|---|
| `Amount` | `log1p(Amount)` → `Amount_log` | Phân phối lệch phải mạnh (max 25,691 vs Q3 chỉ 77) |
| `Time` | `(Time % 86400) / 3600` → `Time_hour` | Fraud tập trung bất thường vào giờ ít giao dịch (ban đêm); giây tuyệt đối dễ overfit vào đúng 2 ngày thu thập |
| Tất cả 30 feature cuối (V1-V28 + 2 feature trên) | `RobustScaler` | Ít bị ảnh hưởng bởi giá trị cực lớn so với `StandardScaler` |

Logic đầy đủ nằm trong `feature_engineering.py`, dùng chung cho cả training
lẫn `predict()` — đảm bảo train/inference không bao giờ lệch nhau.

## Cách train model

```bash
pip install -r requirements.txt

# 1. Train 4 model baseline (Logistic Regression, Decision Tree, Random Forest, XGBoost)
python train.py --input creditcard.csv --models-dir models

# 2. Hyperparameter tuning cho XGBoost (model tốt nhất theo PR-AUC)
python tune.py --input creditcard.csv --models-dir models --n-iter 30

# 3. Evaluation đầy đủ + chọn threshold tối ưu
python evaluate.py --input creditcard.csv --models-dir models --reports-dir reports

# 4. Explainability (Feature Importance + SHAP)
python explainability.py --input creditcard.csv --models-dir models --reports-dir reports
```

## Xử lý Imbalanced Data

Đã so sánh 4 chiến lược (Baseline, Class Weight, Random Undersampling, SMOTE) —
chi tiết ở `notebooks/03_imbalance_comparison.ipynb`. **Quyết định cuối:** không
resample dữ liệu; mỗi model tự xử lý imbalance qua tham số riêng
(`class_weight='balanced'` cho Logistic Regression/Decision Tree/Random Forest,
`scale_pos_weight` cho XGBoost) — vì SMOTE/Undersampling/Class Weight với
threshold 0.5 mặc định làm Precision sập xuống ~5%, trong khi cách này giữ
được cân bằng Precision/Recall tốt hơn nhiều.

## Kết quả

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost (tuned) — best_model** | 0.890 | 0.835 | 0.862 | **0.880** | 0.977 |
| XGBoost (baseline) | 0.880 | 0.835 | 0.857 | 0.879 | 0.968 |
| Random Forest | 0.830 | 0.804 | 0.817 | 0.836 | 0.978 |
| Logistic Regression | 0.053 | 0.907 | 0.100 | 0.710 | 0.971 |
| Decision Tree | 0.113 | 0.804 | 0.198 | 0.459 | 0.901 |

*(bảng đầy đủ, có Accuracy: `reports/model_comparison.csv`)*

**Threshold được chọn:** 0.549 (thay vì mặc định 0.5), tối ưu theo F1 trên
PR curve — xem `reports/chosen_threshold.json`.

**Top feature quan trọng nhất (SHAP):** V14, V4, V10, V12, V11 — khớp phần lớn
với top feature từ phân tích EDA ban đầu (separation score + correlation),
xác nhận model học đúng tín hiệu thật. Chi tiết chiều tác động
(feature nào đẩy về phía fraud/legit): `reports/shap_summary.png`.

## Cấu trúc thư mục

```
docs/business_understanding.md
notebooks/
  01_data_understanding.ipynb
  02_eda.ipynb
  03_imbalance_comparison.ipynb
preprocessing.py          # Data cleaning + train/test split
feature_engineering.py    # Feature engineering (dùng chung train + inference)
train.py                  # Train 4 model baseline
tune.py                   # Hyperparameter tuning (XGBoost)
evaluate.py                # Đánh giá đầy đủ + chọn threshold
explainability.py          # Feature Importance + SHAP
predict.py                 # Hàm bàn giao cho B (contract API)
models/
  best_model.pkl
  scaler.pkl
  logistic_regression.pkl / decision_tree.pkl / random_forest.pkl / xgboost.pkl
reports/
  model_comparison.csv
  pr_curve.png
  feature_importance.png / .csv
  shap_summary.png / shap_importance_bar.png
  chosen_threshold.json
metrics.json
```



**Contract:**
```python
from predict import predict
predict({"Time": 12345, "V1": 0.12, ..., "Amount": 99.5})
# -> {"prediction": 1, "probability": 0.98}
```
