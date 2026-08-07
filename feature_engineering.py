"""
feature_engineering.py — Scaling, encoding, tạo feature mới

Quyết định thiết kế (từ EDA):
- Amount lệch phải mạnh -> log1p transform.
- Time có pattern rõ theo giờ trong ngày (fraud tập trung ban đêm) ->
  tách feature Time_hour = giờ trong ngày (0-24), bỏ Time gốc (giây tuyệt đối
  không có ý nghĩa ngoài phạm vi 2 ngày thu thập dữ liệu, dễ overfit vào đúng
  2 ngày đó thay vì học pattern "giờ nào trong ngày").
- Dùng RobustScaler (không phải StandardScaler) vì ít bị ảnh hưởng bởi
  giá trị cực lớn trong Amount đã biết từ EDA.

QUAN TRỌNG: các hàm ở đây được dùng CẢ cho training (build_train_features)
LẪN cho inference một dòng JSON từ predict.py (engineer_features nhận cả
DataFrame nhiều dòng lẫn 1 dòng). Nếu sửa logic feature ở đây, train.py
và predict.py sẽ tự động đồng bộ theo — không cần sửa 2 chỗ.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Toàn bộ feature dùng để train model, theo đúng thứ tự cố định.
# Thứ tự này phải khớp giữa lúc fit scaler và lúc transform lúc predict.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Time_hour"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm Amount_log, Time_hour. Nhận vào df có Time, Amount, V1-V28
    (Class có hoặc không đều được — hàm này không đụng vào Class).
    """
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_hour"] = (df["Time"] % 86400) / 3600
    return df


def fit_scaler(train_df: pd.DataFrame) -> RobustScaler:
    """Fit RobustScaler CHỈ trên tập train (sau khi đã engineer_features)."""
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


def transform(df: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    """Áp dụng scaler đã fit sẵn lên df (đã qua engineer_features)."""
    return scaler.transform(df[FEATURE_COLUMNS])


def build_train_features(train_df: pd.DataFrame):
    """
    Pipeline đầy đủ cho training: engineer -> fit scaler -> transform.
    Trả về (X_train, y_train, scaler) — scaler này phải lưu lại (scaler.pkl)
    để predict.py dùng lại y hệt, không fit lại.
    """
    train_df = engineer_features(train_df)
    scaler = fit_scaler(train_df)
    X_train = transform(train_df, scaler)
    y_train = train_df["Class"].values
    return X_train, y_train, scaler


def build_eval_features(df: pd.DataFrame, scaler: RobustScaler):
    """Dùng cho test set — KHÔNG fit lại scaler, chỉ transform."""
    df = engineer_features(df)
    X = transform(df, scaler)
    y = df["Class"].values if "Class" in df.columns else None
    return X, y


def build_inference_features(record: dict, scaler: RobustScaler) -> np.ndarray:
    """
    Dùng cho predict.py — nhận 1 dict JSON theo đúng input API đã thống nhất
    với B: {"Time": ..., "V1": ..., ..., "Amount": ...}
    Trả về mảng 2D (1, n_features) sẵn sàng đưa vào model.predict().
    """
    df = pd.DataFrame([record])
    X = build_eval_features(df, scaler)[0]
    return X
