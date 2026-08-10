# Business Understanding — Credit Card Fraud Detection

## 1. Mô tả bài toán
Ngân hàng/tổ chức phát hành thẻ tín dụng cần phát hiện các giao dịch gian lận (fraud)
ngay khi chúng xảy ra, để chặn hoặc gắn cờ trước khi gây thiệt hại tài chính.
Đây là bài toán **phân loại nhị phân (binary classification)**: mỗi giao dịch được
gán nhãn là **fraud (1)** hoặc **hợp lệ (0)**.

Đặc điểm quan trọng nhất của bài toán này: dữ liệu **cực kỳ mất cân bằng**
(fraud thường chỉ chiếm ~0.17% tổng số giao dịch). Vì vậy accuracy thông thường
sẽ đánh lừa — một model dự đoán "toàn bộ là hợp lệ" vẫn đạt accuracy >99.8%
nhưng vô dụng trong thực tế.

## 2. Biến đầu vào / đầu ra

**Đầu vào (features):**
- `Time`: số giây trôi qua kể từ giao dịch đầu tiên trong tập dữ liệu
- `V1` – `V28`: các thành phần đã qua biến đổi PCA (ẩn danh hóa để bảo mật thông tin
  gốc của khách hàng/ngân hàng) — không biết ý nghĩa gốc của từng feature
- `Amount`: số tiền giao dịch

**Đầu ra (target):**
- `Class`: 0 = giao dịch hợp lệ, 1 = giao dịch gian lận

## 3. Metric đánh giá

Vì dữ liệu mất cân bằng nặng, **không dùng Accuracy làm metric chính**. Ưu tiên:

| Metric | Vai trò |
|---|---|
| **Precision** | Trong số giao dịch model gắn cờ fraud, bao nhiêu % thực sự là fraud? (giảm false positive — tránh làm phiền khách hàng thật) |
| **Recall** | Trong số fraud thực tế, model bắt được bao nhiêu %? (giảm false negative — đây là cái tốn tiền nhất) |
| **F1-score** | Cân bằng giữa Precision và Recall |
| **PR-AUC (Precision-Recall AUC)** | Metric tổng quan phù hợp nhất cho bài toán imbalanced — quan trọng hơn ROC-AUC |
| **ROC-AUC** | Vẫn tính để tham khảo, nhưng dễ "trông đẹp" một cách gây hiểu lầm khi data lệch |

**Trade-off cần cân nhắc:** ngưỡng threshold quyết định fraud/không fraud ảnh hưởng
trực tiếp đến Precision vs Recall. Ngân hàng thường chấp nhận Recall cao hơn
(chặn nhầm một vài giao dịch thật còn hơn bỏ lọt fraud), nhưng đây là quyết định
business, không phải thuần kỹ thuật — sẽ bàn kỹ hơn ở phần Evaluation.
