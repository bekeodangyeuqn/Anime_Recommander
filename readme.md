# Hướng dẫn cách chạy:

## Với Airflow:

- Cài đặt Docker trên máy (nếu chưa có)
- Mở docker desktop (Đối với windows)
- Mở terminal vào thư mục gốc của ứng dụng gõ các câu lệnh sau:
  - docker build . --tag pyrequire_airflow:2.3.2
  - docker-compose up -d --build
- Chờ tầm 1, 2 phút vào trình duyệt gõ localhost:5000

## Với web:

- Cài đặt thư viện flask nếu chưa có: pip install flask
- Mở terminal vào thư mục gốc của ứng dụng gõ các câu lệnh sau:
  - set FLASK_APP=app
  - flask run
- Chờ tầm 1, 2 phút vào trình duyệt gõ localhost:80 hoặc nhấn vào liên kết xuất hiện ở terminal sau khi câu lệnh chạy xong

## Ngoài ra ở folder notebooks có 2 file notebook để mô tả và chạy các thao tác liên quan đến Trực quan hóa dữ liệu và Huấn luyện mô hình. Thầy và các bạn có thể sử dụng Jupyter notebook để xem và thao tác trên hai file này.

### (Với Github) Vì folder data chứa dữ liệu khá nặng nên không thể đẩy lên github được, nên có ai cần thì có thể liên hệ mình để lấy
