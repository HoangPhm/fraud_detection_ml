"""
tune.py — Giai đoạn 4: Hyperparameter Tuning

Chỉ tune XGBoost (model thắng ở Giai đoạn 3, theo PR-AUC).

Dùng RandomizedSearchCV thay vì GridSearchCV vì không gian tham số của
XGBoost khá lớn (7 tham số) — grid đầy đủ sẽ tốn quá nhiều thời gian.
RandomizedSearchCV chỉ thử `n_iter` tổ hợp ngẫu nhiên, đủ tốt để tìm vùng
tham số tốt mà không cần duyệt hết.

scoring='average_precision' = PR-AUC — dùng đúng metric đã chốt để so sánh
model ở Giai đoạn 3, không đổi sang accuracy hay f1 giữa chừng.

cv=StratifiedKFold — bắt buộc vì fraud quá hiếm (383 mẫu trong train); k-fold
thường sẽ vô tình để một số fold gần như không có mẫu fraud nào nếu không
stratify, làm điểm CV không ổn định.
"""

import argparse
import json
import os

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

try:
    from src.data.preprocessing import clean_and_split, load_raw_data
    from src.features.feature_engineering import build_eval_features, build_train_features
except ImportError:
    from preprocessing import clean_and_split, load_raw_data
    from feature_engineering import build_eval_features, build_train_features

RANDOM_STATE = 42

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.29),       # 0.01 - 0.30
    "subsample": uniform(0.6, 0.4),              # 0.6 - 1.0
    "colsample_bytree": uniform(0.6, 0.4),       # 0.6 - 1.0
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
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
    parser = argparse.ArgumentParser(description="Tune XGBoost bằng RandomizedSearchCV")
    parser.add_argument("--input", required=True, help="Đường dẫn tới creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-iter", type=int, default=30, help="Số tổ hợp tham số thử")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("Đang load + clean + split + feature engineering...")
    df = load_raw_data(args.input)
    train_df, test_df = clean_and_split(df)
    X_train, y_train, scaler = build_train_features(train_df)
    X_test, y_test = build_eval_features(test_df, scaler)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        scoring="average_precision",  # = PR-AUC, khớp metric đã chốt
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    print(f"\nBắt đầu RandomizedSearchCV: {args.n_iter} tổ hợp × {args.cv_folds}-fold CV...")
    search.fit(X_train, y_train)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_
    result_tuned = evaluate_model(best_model, X_test, y_test, "xgboost_tuned")
    print(f"\nKết quả trên test set (model đã tune): {result_tuned}")

    # So sánh với model XGBoost baseline (chưa tune) đã lưu ở Giai đoạn 3
    baseline_path = os.path.join(args.models_dir, "xgboost.pkl")
    if os.path.exists(baseline_path):
        baseline_model = joblib.load(baseline_path)
        result_baseline = evaluate_model(baseline_model, X_test, y_test, "xgboost_baseline")
        print(f"Kết quả model baseline (chưa tune):  {result_baseline}")
        improvement = result_tuned["pr_auc"] - result_baseline["pr_auc"]
        print(f"Chênh lệch PR-AUC (tuned - baseline): {improvement:+.4f}")

    # Lưu best model đúng tên theo cấu trúc đóng gói đã thống nhất với B
    best_model_path = os.path.join(args.models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\nĐã lưu {best_model_path}")

    with open(os.path.join(args.models_dir, "best_params.json"), "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print(f"Đã lưu {os.path.join(args.models_dir, 'best_params.json')}")


if __name__ == "__main__":
    main()
