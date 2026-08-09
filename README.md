# 🔨 ForgeML

**ForgeML** là một công cụ MLOps CLI thu nhỏ, được thiết kế để tự động hóa hoàn toàn quy trình huấn luyện mô hình Machine Learning từ môi trường máy tính cá nhân (local) lên máy chủ Kaggle.

Thay vì phải nén code thủ công, tạo dataset, copy/paste lên Kaggle Notebook và tải kết quả về bằng tay, ForgeML giúp bạn thực hiện toàn bộ vòng đời này chỉ với **một dòng lệnh duy nhất**.

## ✨ Tính Năng Nổi Bật

* **🚀 Tự Động Hóa Toàn Diện (End-to-End Automation):** Tự động đóng gói mã nguồn (`src`, `configs`, `requirements.txt`), tải lên private dataset của Kaggle, khởi tạo Kernel, và kích hoạt tiến trình huấn luyện.
* **🛡️ Xử Lý Lỗi Phần Cứng Thông Minh:** Tự động phát hiện sự cố không tương thích GPU trên Kaggle (ví dụ GPU P100 sm_60 với PyTorch 12.x+) và fallback mượt mà về CPU để đảm bảo tiến trình không bao giờ bị crash.
* **📊 Theo Dõi Thử Nghiệm (Experiment Tracking):** Tích hợp sẵn cơ sở dữ liệu SQLite tại local (`forge.sqlite`). Theo dõi thời gian chạy, tiến độ, và tự động thu thập Metrics (AUROC, Accuracy, v.v...) của mọi lượt chạy.
* **📦 Tự Động Tải Kết Quả (Artifact Collection):** Trọng số mô hình (`.pkl`, `.pth`) và log kết quả được tự động tải thẳng về máy tính của bạn sau khi Kernel chạy xong.
* **🤖 Giao Tiếp Bằng Ngôn Ngữ Tự Nhiên (NLP):** Tích hợp LLM giúp bạn có thể yêu cầu chạy mô hình bằng tiếng Việt thông qua lệnh `forge ask` thay vì phải nhớ các tham số dòng lệnh phức tạp.
* **🔄 Khả Năng Khôi Phục (Resume):** Có thể chạy tiếp tiến trình bị gián đoạn (do tắt terminal) bằng ID của lượt chạy đó mà không cần upload lại dữ liệu.

## 🛠️ Cài Đặt

Môi trường yêu cầu: Python 3.10+

1. Clone mã nguồn ForgeML.
2. Cài đặt vào môi trường ảo bằng `pip`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. Xác thực Kaggle API trên máy của bạn (nếu chưa có):
   ```bash
   kaggle auth login
   ```

## 📖 Hướng Dẫn Sử Dụng
Vui lòng xem file [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md) để biết cách setup và chạy ForgeML cho một dự án AI bất kỳ.
