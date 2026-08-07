"""
train.py — Giai đoạn 3: Machine Learning

Quyết định đã chốt ở bước Imbalanced Data comparison (notebook 03):
- KHÔNG dùng SMOTE / Random Undersampling làm mặc định (làm méo phân phối train,
  khiến threshold 0.5 mặc định trở nên vô nghĩa, Precision sập xuống ~5%).
- Dùng class_weight='balanced' (LogisticRegression, DecisionTree, RandomForest)
  và scale_pos_weight (XGBoost) — model tự học phạt nặng khi đoán sai fraud,
  không cần resample dữ liệu.
- Đánh giá bằng PR-AUC là chính (không chốt Precision/Recall ở threshold 0.5
  mặc định — việc chọn threshold để làm ở Giai đoạn 5 Evaluation).

Import lại preprocessing.py và feature_engineering.py — KHÔNG viết lại logic
clean/split/feature ở đây, để tránh lệch giữa các bước.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Hỗ trợ cả 2 kiểu layout:
# - src/data/preprocessing.py + src/features/feature_engineering.py (VSCode, package đầy đủ)
# - preprocessing.py + feature_engineering.py nằm cùng thư mục (Colab, chạy phẳng)
try:
    from src.data.preprocessing import clean_and_split, load_raw_data
    from src.features.feature_engineering import build_eval_features, build_train_features
except ImportError:
    from preprocessing import clean_and_split, load_raw_data
    from feature_engineering import build_eval_features, build_train_features

RANDOM_STATE = 42


def get_models(y_train: np.ndarray) -> dict:
    """Định nghĩa 4 model, mỗi model tự xử lý imbalance theo cách riêng của nó."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # cho XGBoost

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=10
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=12,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 4 model cho fraud detection")
    parser.add_argument("--input", required=True, help="Đường dẫn tới creditcard.csv")
    parser.add_argument("--models-dir", default="models", help="Thư mục lưu .pkl")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("Đang load + clean + split dữ liệu...")
    df = load_raw_data(args.input)
    train_df, test_df = clean_and_split(df)

    print("Đang tạo feature (fit scaler trên train)...")
    X_train, y_train, scaler = build_train_features(train_df)
    X_test, y_test = build_eval_features(test_df, scaler)

    joblib.dump(scaler, os.path.join(args.models_dir, "scaler.pkl"))
    print(f"Đã lưu scaler.pkl")

    models = get_models(y_train)
    all_results = []

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name)
        all_results.append(result)
        print(result)

        model_path = os.path.join(args.models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Đã lưu {model_path}")

    results_df = pd.DataFrame(all_results).set_index("model")
    results_df = results_df.sort_values("pr_auc", ascending=False)
    print("\n=== So sánh 4 model (sắp xếp theo PR-AUC) ===")
    print(results_df.round(4))

    # Lưu metrics.json — bàn giao cho B, và dùng lại ở Giai đoạn 4 (Hyperparameter tuning)
    metrics_path = os.path.join(args.models_dir, "..", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nĐã lưu {metrics_path}")

    best_model_name = results_df.index[0]
    print(f"\nModel tốt nhất theo PR-AUC (trước khi tune): {best_model_name}")
    print("Bước tiếp theo: Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) cho model này.")


if __name__ == "__main__":
    main()
