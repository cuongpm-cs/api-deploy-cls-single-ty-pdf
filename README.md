# Hướng dẫn cài đặt và sử dụng API Phân loại văn bản (Cục CDS Bộ NN&MT)
Dự án này được phát triền để triển khai mô hình phân loại văn bản hành chính tiếng Việt. Hệ thống bao gồm các mô-đun xử lý đầu vào PDF, trích xuất nội dung văn bản bằng OCR (nếu cần), và áp dụng các mô hình học sâu đã huấn luyện để xác định các nhãn phân loại phù hợp.

## Cài đặt
Đây là RESTful API được xây dựng bằng Python Flask, cho phép:
- Trích xuất văn bản từ PDF
- Tiền xử lý văn bản
- Phân loại văn bản sử dụng mô hình đã huấn luyện trước (pretrained models), bồm ba nhãn:
    - Cục Biến đổi khí hậu
    - Cục Công nghệ thông tin
    - Cục đo đạc bản đồ
- Hỗ trợ OCR với Tesseract nếu là scan PDF

---

### Cấu trúc thư mục

```
api_deploy_full_cls/
├── models-full-v2/ # Mô hình phân loại dựa trên trích yếu + cơ quan ban hành + nội dung văn bản (PDF)
├── models-ty-cqbh/ # Mô hình phân loại dựa trên trích yếu + cơ quan ban hành
├── poppler-24.07.0/ # Hỗ trợ đọc PDF
├── Tesseract-OCR/ # Thư viện OCR
├── mlb.pkl # MultiLabelBinarizer (pickle)
├── pdf_converter.py # Hàm trích xuất văn bản từ PDF
├── run_api.py # File chính để chạy API
├── requirements.txt # Thư viện cần cài đặt
└── README.md # Tài liệu này
```

---
1. **Giải nén mã nguồn**

    Tải và giải nén file `api_deploy_full_cls.zip` vào một thư mục trên máy của bạn.

2. **Di chuyển vào thư mục dự án**

    Mở terminal (hoặc Command Prompt), sau đó chuyển đến thư mục vừa giải nén:

    ```bash
    cd path/to/api_deploy_full_cls
    ```
3. **Cài đặt thư viện cần thiết**
    ```bash
    pip install -r requirements.txt
    ```
4. **Chạy API**
    ```bash
    python run_api.py
    ```
    API sẽ khởi động ở ```http://127.0.0.1:8888```

## Request API

### 1. Endpoint: `POST /cls_ty_cqbh`

API phân loại văn bản dựa trên 2 trường đầu vào:

- `trich_yeu`: nội dung trích yếu
- `co_quan_ban_hanh`: tên cơ quan ban hành

Ví dụ gọi bằng `curl`:

```bash
curl -X POST http://localhost:8888/cls_full \
  -H "Content-Type: application/json" \
  -d '{
        "trich_yeu": "v v góp ý dự thảo kế hoạch của bộ tài nguyên và môi trường triển khai chiến lược quốc gia phát triển kinh tế số và xã hội số đến năm 2025 định hướng đến năm 2030 theo quyết định số 411 qđ ttg",
        "co_quan_ban_hanh": "trường đại học tài nguyên và môi trường tp hcm"
      }'
```

Response: `{'output': ['Cục Công Nghệ Thông Tin']}`

### 2. Endpoint: `POST /cls_full`

API phân loại văn bản dựa trên 3 trường đầu vào:
A
- `trich_yeu`: nội dung trích yếu
- `co_quan_ban_hanh`: tên cơ quan ban hành
- `pdf_file`: nội dung văn bản PDF

Ví dụ gọi bằng `curl`:

```bash
curl -X POST http://localhost:8888/cls_full \
  -H "Content-Type: application/json" \
  -d '{
        "trich_yeu": "v v góp ý dự thảo kế hoạch của bộ tài nguyên và môi trường triển khai chiến lược quốc gia phát triển kinh tế số và xã hội số đến năm 2025 định hướng đến năm 2030 theo quyết định số 411 qđ ttg",
        "co_quan_ban_hanh": "trường đại học tài nguyên và môi trường tp hcm",
        "pdf_file": "path/to/1295 Góp ý dự thảo kinh tế số_Signed.pdf"
      }'
```

Response: `{'output': ['Cục Công Nghệ Thông Tin']}`