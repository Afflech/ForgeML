# 📘 Hướng Dẫn Sử Dụng ForgeML

Tài liệu này hướng dẫn bạn cách thiết lập và sử dụng ForgeML cho một dự án AI/Machine Learning hoàn toàn mới. 

---

## Bước 1: Tổ Chức Thư Mục Mã Nguồn

ForgeML không can thiệp vào logic code AI của bạn. Bạn chỉ cần tổ chức thư mục dự án (ví dụ `MyNewProject/`) theo cấu trúc cơ bản sau:

```text
MyNewProject/
├── src/                <-- Chứa toàn bộ code Python (ví dụ: mô hình, dataset loader, utils)
├── configs/            <-- (Tùy chọn) Chứa các file YAML/JSON cấu hình siêu tham số
└── requirements.txt    <-- Khai báo các thư viện cần cài (ví dụ: torch, numpy, scikit-learn)
```
*Lưu ý: ForgeML sẽ tự động cài đặt các thư viện trong `requirements.txt` vào máy ảo Kaggle trước khi chạy.*

---

## Bước 2: Khởi Tạo Dự Án (`forge init`)

Mở terminal, di chuyển vào thư mục dự án của bạn (đã có mã nguồn ở Bước 1), và chạy lệnh khởi tạo:

```bash
cd /path/to/MyNewProject
forge init
```

Lệnh này sẽ tạo ra một file cấu hình mang tên `forge.yaml` nằm ngay trong thư mục dự án của bạn.

---

## Bước 3: Cấu Hình Tài Nguyên Kaggle

Mở file `forge.yaml` vừa được tạo ra. Nội dung của nó sẽ trông như sau:

```yaml
project:
  name: my_project_name              # Đặt tên dự án của bạn

provider:
  name: kaggle

kaggle:
  kernel: my-training-kernel         # Đặt tên cho Kernel trên Kaggle (tùy ý)
  dataset: my-source-dataset         # Đặt tên cho Private Dataset chứa code (tùy ý)
  mvtec_dataset: "ipythonx/mvtec-ad" # <-- Dataset dữ liệu huấn luyện công khai trên Kaggle
  accelerator: NvidiaTeslaT4
  internet: true
```

**Thành phần quan trọng nhất:** 
Tại trường `mvtec_dataset` (có thể đổi tên key này trong code nếu bạn dùng dataset khác), bạn chỉ cần dán **slug của dataset** trên Kaggle vào. 
Ví dụ: Bạn tìm thấy dataset chó mèo trên Kaggle có URL là `kaggle.com/datasets/johndoe/cats-and-dogs`, hãy điền `johndoe/cats-and-dogs` vào đây. ForgeML sẽ tự động đính kèm dataset khổng lồ này vào Kernel cho bạn mà bạn không cần phải tải về máy local!

---

## Bước 4: Kích Hoạt Huấn Luyện (`forge run`)

Khi code đã sẵn sàng, bạn chỉ cần gõ 1 dòng lệnh duy nhất để đưa mọi thứ lên mây:

```bash
forge run --model <tên-mô-hình> --category <nhãn-dữ-liệu>
```
*(Các biến số này sẽ được truyền thẳng vào file `run_config.json` để mã nguồn trong thư mục `src` của bạn đọc và xử lý).*

Quá trình này diễn ra hoàn toàn tự động:
1. Đóng gói mã nguồn thành `bundle.tar.gz`.
2. Tạo Private Dataset trên Kaggle và tải mã nguồn lên.
3. Kích hoạt Script Kernel và bắt đầu huấn luyện.
4. Tự động gom file trọng số `.pkl`/`.pth` và `metrics.json` từ Kaggle tải thẳng về thư mục `artifacts/<run_id>/output/` trên máy bạn!

---

## Bước 5: Theo Dõi & Quản Lý

Trong lúc Kernel đang chạy trên Kaggle (thường kéo dài vài chục phút đến vài giờ), bạn có thể kiểm tra trạng thái bằng các lệnh sau:

- **Xem trạng thái hiện tại:**
  ```bash
  forge status
  ```
  *(Sẽ hiển thị trạng thái như `PACKAGING`, `UPLOADING`, `QUEUED`, `RUNNING`, `COMPLETED`...)*

- **Xem lịch sử các lần chạy (Tracking):**
  ```bash
  forge history
  ```
  *(Sẽ hiển thị một bảng tổng hợp chứa Run ID, Mô hình, Thời gian chạy, và các chỉ số Metrics).*

---

## 🤖 Tính Năng Nâng Cao

### 1. Khôi phục tiến trình (Resume)
Nếu bạn lỡ tay tắt terminal trong lúc đang đợi Kaggle chạy, đừng lo! ForgeML vẫn đang âm thầm giám sát. Bạn chỉ cần lấy lại `Run ID` (bằng lệnh `forge history`) và chạy tiếp:
```bash
forge run --run-id "20260809T103602Z-687f"
```
ForgeML sẽ bỏ qua bước upload code và tải kết quả về khi Kaggle chạy xong.

### 2. Giao tiếp bằng Ngôn Ngữ Tự Nhiên
Nếu bạn đã thiết lập `OPENAI_API_KEY` trong file `.env` của dự án, bạn có thể ra lệnh cho ForgeML bằng tiếng Việt:
```bash
forge ask "Hãy huấn luyện mô hình fastflow trên dữ liệu viên thuốc với seed 123"
```
AI sẽ tự động phân tích câu nói của bạn thành các tham số cấu hình chuẩn xác và xác nhận trước khi thực sự chạy!
