"""
preprocessing.py — Data Cleaning + Train/Test Split

Quyết định thiết kế (đã thống nhất ở giai đoạn Data Understanding/EDA):
1. Duplicate (1,081 dòng, 0.38%) chỉ được drop SAU khi split, không phải trước.
   Lý do: nếu drop trước, một cặp duplicate có thể bị tách ra — 1 bản rơi vào
   train, 1 bản rơi vào test -> data leakage (model "nhìn thấy" test qua bản sao).
2. Split bắt buộc dùng stratify=Class vì fraud chỉ chiếm 0.17% -> nếu không
   stratify, tỷ lệ fraud giữa train/test có thể lệch nhau đáng kể.
3. Không xử lý outlier ở Amount (đã xác nhận ở EDA: không phải lỗi dữ liệu).
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Đọc file creditcard.csv gốc."""
    df = pd.read_csv(csv_path)
    return df


def clean_and_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split trước (stratified theo Class), sau đó dedupe từng tập riêng biệt.
    Trả về (train_df, test_df) đã sẵn sàng cho bước feature engineering.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Class"],
    )

    n_dup_train = train_df.duplicated().sum()
    n_dup_test = test_df.duplicated().sum()

    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    print(f"Train: {len(train_df)} dòng (đã drop {n_dup_train} duplicate)")
    print(f"Test:  {len(test_df)} dòng (đã drop {n_dup_test} duplicate)")
    print(f"Train fraud rate: {train_df['Class'].mean()*100:.4f}%")
    print(f"Test  fraud rate: {test_df['Class'].mean()*100:.4f}%")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Data cleaning + train/test split")
    parser.add_argument("--input", required=True, help="Đường dẫn tới creditcard.csv")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Thư mục lưu train.csv/test.csv"
    )
    args = parser.parse_args()

    df = load_raw_data(args.input)
    train_df, test_df = clean_and_split(df)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
    print(f"Đã lưu train.csv, test.csv vào {args.output_dir}")


if __name__ == "__main__":
    main()
