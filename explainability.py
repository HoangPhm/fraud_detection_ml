"""
explainability.py — Giai đoạn 6: Explainability

1. Feature Importance có sẵn từ XGBoost (nhanh, dựa trên số lần feature
   được dùng để split / mức giảm loss trung bình).
2. SHAP values (chính xác hơn, giải thích được TỪNG dự đoán riêng lẻ,
   không chỉ tổng quan toàn model) — dùng TreeExplainer vì model là XGBoost
   (nhanh hơn nhiều so với KernelExplainer dùng cho model bất kỳ).

Mục đích: đối chiếu top feature quan trọng nhất ở đây với top feature từ
separation score / correlation đã tìm ra ở notebook 02_eda.ipynb (V17, V14,
V12, V10, V16...). Nếu khớp phần lớn → bằng chứng model học đúng tín hiệu
thật, không phải học nhiễu — nội dung quan trọng cần đưa vào Model_Report.
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

try:
    from src.data.preprocessing import clean_and_split, load_raw_data
    from src.features.feature_engineering import FEATURE_COLUMNS, build_eval_features
except ImportError:
    from preprocessing import clean_and_split, load_raw_data
    from feature_engineering import FEATURE_COLUMNS, build_eval_features


def main():
    parser = argparse.ArgumentParser(description="Feature importance + SHAP cho best model")
    parser.add_argument("--input", required=True, help="Đường dẫn tới creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--model-file", default="best_model.pkl")
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=2000,
        help="Số dòng test lấy mẫu để tính SHAP (toàn bộ test set sẽ chậm)",
    )
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    print("Đang load dữ liệu + model...")
    df = load_raw_data(args.input)
    _, test_df = clean_and_split(df)
    scaler = joblib.load(os.path.join(args.models_dir, "scaler.pkl"))
    X_test, y_test = build_eval_features(test_df, scaler)
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)

    model = joblib.load(os.path.join(args.models_dir, args.model_file))

    # --- 1. Feature Importance (built-in XGBoost) ---
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        importance = importance.sort_values(ascending=False)

        print("\n=== Top 10 Feature Importance (XGBoost built-in) ===")
        print(importance.head(10))

        plt.figure(figsize=(9, 6))
        importance.head(15).sort_values().plot(kind="barh")
        plt.title("Feature Importance (top 15) — XGBoost")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_path = os.path.join(args.reports_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu {fi_path}")

        importance.to_csv(os.path.join(args.reports_dir, "feature_importance.csv"))
    else:
        print(f"Model {args.model_file} không có feature_importances_, bỏ qua bước này.")

    # --- 2. SHAP ---
    print(f"\nĐang tính SHAP trên mẫu {args.shap_sample_size} dòng test (để chạy nhanh hơn)...")
    sample = X_test_df.sample(
        n=min(args.shap_sample_size, len(X_test_df)), random_state=42
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    shap_summary_path = os.path.join(args.reports_dir, "shap_summary.png")
    plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu {shap_summary_path}")

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    shap_bar_path = os.path.join(args.reports_dir, "shap_importance_bar.png")
    plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu {shap_bar_path}")

    print("\nGợi ý nhận xét cần viết trong Model_Report.md:")
    print("- Đối chiếu top feature ở feature_importance.csv với top feature từ")
    print("  separation score/correlation trong notebooks/02_eda.ipynb — có khớp không?")
    print("- shap_summary.png cho biết feature nào đẩy dự đoán về phía fraud (đỏ, bên phải)")
    print("  hay về phía legit (xanh, bên trái) — không chỉ mức độ quan trọng mà cả CHIỀU tác động.")


if __name__ == "__main__":
    main()
