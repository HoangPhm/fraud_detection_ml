"""
predict.py — hàm bàn giao chính thức cho B, dùng trực tiếp trong FastAPI.

Input API (đã thống nhất với B):
    {"Time": 12345, "V1": 0.12, "V2": -1.45, ..., "Amount": 99.5}

Output API (đã thống nhất với B):
    {"prediction": 1, "probability": 0.98}

B chỉ cần:
    from predict import predict
    result = predict(request_json)

KHÔNG cần biết bên trong dùng model gì, scaler gì, threshold bao nhiêu —
toàn bộ logic đó nằm ở đây, tách biệt khỏi code API/deployment của B.
"""

import os

import joblib

from feature_engineering import build_inference_features

# Đường dẫn mặc định — B có thể đổi qua biến môi trường khi deploy
# (ví dụ khác nhau giữa local, staging, production)
MODELS_DIR = os.environ.get("FRAUD_MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Threshold chọn từ Giai đoạn 5 (evaluate.py, dựa trên F1 tối ưu trên PR curve),
# KHÔNG dùng 0.5 mặc định. Cập nhật số này sau khi chạy evaluate.py xong,
# lấy từ reports/chosen_threshold.json.
THRESHOLD = 0.5490231514   # TODO: thay bằng giá trị thật từ reports/chosen_threshold.json

# Load model + scaler MỘT LẦN lúc import module, không load lại mỗi request
# (load .pkl tốn thời gian — nếu load lại mỗi lần gọi predict() sẽ rất chậm
# khi API nhận nhiều request liên tục).
_model = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def predict(record: dict) -> dict:
    """
    record: dict theo đúng input API, ví dụ:
        {"Time": 12345, "V1": 0.12, ..., "V28": ..., "Amount": 99.5}

    Trả về đúng output API:
        {"prediction": 0 hoặc 1, "probability": float}

    Raises:
        KeyError nếu record thiếu field bắt buộc (Time, V1-V28, Amount).
    """
    model, scaler = _load_artifacts()

    X = build_inference_features(record, scaler)  # (1, n_features)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return {"prediction": prediction, "probability": probability}


if __name__ == "__main__":
    # Test nhanh bằng tay trước khi B tích hợp — điền 1 dòng thật từ test set
    # (KHÔNG dùng dòng đã train) để kiểm tra predict() chạy đúng format.
    sample_record = {
        "Time": 50000, "V1": -1.2, "V2": 0.5, "V3": 1.1, "V4": -0.3,
        "V5": 0.2, "V6": -0.1, "V7": 0.4, "V8": -0.05, "V9": 0.3,
        "V10": -0.6, "V11": 0.1, "V12": -0.4, "V13": 0.2, "V14": -0.9,
        "V15": 0.15, "V16": -0.3, "V17": -0.8, "V18": 0.1, "V19": 0.05,
        "V20": -0.02, "V21": 0.03, "V22": 0.1, "V23": -0.01, "V24": 0.02,
        "V25": 0.04, "V26": -0.03, "V27": 0.01, "V28": 0.005, "Amount": 149.62,
    }
    print(predict(sample_record))
