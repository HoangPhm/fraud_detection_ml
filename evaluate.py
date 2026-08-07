"""
evaluate.py — Giai đoạn 5: Evaluation

1. So sánh đầy đủ metric (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
   cho TẤT CẢ model đã lưu trong models-dir, trong 1 bảng duy nhất.
2. Vẽ Precision-Recall curve cho best_model, tìm threshold tối ưu (thay vì
   dùng mặc định 0.5) — dựa trên điểm F1 cao nhất trên đường cong.

Lưu ý: Accuracy được tính ở đây CHỈ để đối chiếu/minh họa trong báo cáo
(cho thấy accuracy cao giả tạo ~99.9% dù model có thể tệ) — không dùng
Accuracy để xếp hạng hay chọn model, đúng theo business_understanding.md.
"""

import argparse
import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.data.preprocessing import clean_and_split, load_raw_data
    from src.features.feature_engineering import build_eval_features
except ImportError:
    from preprocessing import clean_and_split, load_raw_data
    from feature_engineering import build_eval_features


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def find_best_threshold(y_test, y_proba) -> dict:
    """Quét toàn bộ PR curve, chọn threshold cho F1 cao nhất."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve trả về precisions/recalls dài hơn thresholds 1 phần tử
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return {
        "threshold": float(thresholds[best_idx]),
        "precision_at_threshold": float(precisions[best_idx]),
        "recall_at_threshold": float(recalls[best_idx]),
        "f1_at_threshold": float(f1_scores[best_idx]),
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh toàn bộ model")
    parser.add_argument("--input", required=True, help="Đường dẫn tới creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--best-model-file",
        default="best_model.pkl",
        help="Tên file model chính (để vẽ PR curve + chọn threshold)",
    )
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    print("Đang load + clean + split + feature engineering...")
    df = load_raw_data(args.input)
    _, test_df = clean_and_split(df)

    scaler = joblib.load(os.path.join(args.models_dir, "scaler.pkl"))
    X_test, y_test = build_eval_features(test_df, scaler)

    # Đánh giá toàn bộ model .pkl trong models-dir (trừ scaler.pkl)
    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "*.pkl")))
    model_paths = [p for p in model_paths if not p.endswith("scaler.pkl")]

    all_results = []
    proba_cache = {}
    for model_path in model_paths:
        name = os.path.splitext(os.path.basename(model_path))[0]
        model = joblib.load(model_path)
        result = evaluate_model(model, X_test, y_test, name)
        all_results.append(result)
        proba_cache[name] = model.predict_proba(X_test)[:, 1]
        print(f"{name}: {result}")

    results_df = pd.DataFrame(all_results).set_index("model")
    results_df = results_df.sort_values("pr_auc", ascending=False)

    print("\n=== Bảng so sánh đầy đủ (sắp xếp theo PR-AUC) ===")
    print(results_df.round(4))

    comparison_path = os.path.join(args.reports_dir, "model_comparison.csv")
    results_df.round(4).to_csv(comparison_path)
    print(f"Đã lưu {comparison_path}")

    # --- PR curve + threshold cho best model ---
    best_name = os.path.splitext(args.best_model_file)[0]
    if best_name not in proba_cache:
        best_name = results_df.index[0]
        print(f"Không tìm thấy {args.best_model_file}, dùng model tốt nhất theo PR-AUC: {best_name}")

    threshold_info = find_best_threshold(y_test, proba_cache[best_name])
    print(f"\n=== Threshold tối ưu cho {best_name} (theo F1 cao nhất trên PR curve) ===")
    print(f"Threshold: {threshold_info['threshold']:.4f} (thay vì mặc định 0.5)")
    print(f"Precision tại threshold này: {threshold_info['precision_at_threshold']:.4f}")
    print(f"Recall tại threshold này:    {threshold_info['recall_at_threshold']:.4f}")
    print(f"F1 tại threshold này:        {threshold_info['f1_at_threshold']:.4f}")

    plt.figure(figsize=(7, 6))
    plt.plot(threshold_info["recalls"], threshold_info["precisions"], label=best_name)
    plt.scatter(
        threshold_info["recall_at_threshold"],
        threshold_info["precision_at_threshold"],
        color="red",
        zorder=5,
        label=f"Best F1 threshold={threshold_info['threshold']:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {best_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    pr_curve_path = os.path.join(args.reports_dir, "pr_curve.png")
    plt.savefig(pr_curve_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu {pr_curve_path}")

    # Lưu threshold để predict.py / B dùng lại
    threshold_summary = {
        k: v for k, v in threshold_info.items()
        if k not in ("precisions", "recalls", "thresholds")
    }
    threshold_summary["model"] = best_name
    pd.Series(threshold_summary).to_json(
        os.path.join(args.reports_dir, "chosen_threshold.json"), indent=2
    )
    print(f"Đã lưu {os.path.join(args.reports_dir, 'chosen_threshold.json')}")


if __name__ == "__main__":
    main()
