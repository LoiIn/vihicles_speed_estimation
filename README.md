# vihicles_speed_estimation
# authors: Cấn Đức Lợi
# email: canducloi@gmail.com

---------------------------------------------------------------------------------------
A. Cài đặt chung
- Tạo môi trường biên dịch để cài đặt các thư viện:
            python -m venv .env

- Cài đặt các package:
        python -m install -r requirements.txt (--no-dir-cache)

---------------------------------------------------------------------------------------
B. Đối với server
1. Yêu cầu môi trường:
    - python: 3.7
    - GPU
    - Tensorflow-gpu: 2.3.0
    - cuda-toolkit: 10.1
    - cudnn: 7.6.4
    # chi tiết về các phiên bản xem tại: https://www.tensorflow.org/install/source_windows#gpu

2. Cách cài đặt và sử dụng
    - Download file weight của yolov4 tại địa chỉ:
            https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
        và để trong thư mục ./data 

    - Save model: 
            python -m save_model

    - Tạo thư mực outputs và 3 thư mục con bên trong nó: mp4, csv, imgs 

    - Chạy mô hình tính toán
            python -m speed --input test.mp4 --rwf 5 --rws 12 (--limit) (--save_video)

            Trong đó:
                +) input: video đầu vào, đường dẫn đang được set mặc định là ./data/videos/... Có thể chỉnh trong file ./core/config.py
                +) rwf: chiều dài mép đường thứ nhất
                +) rws: Chiều dài mép đường thứ hai
                +) limit: giới hạn tốc độ mong muốn. Mặc địnhh là 20.
                +) save_model: có lưu video hay không

3. Nếu không có GPU, có thể sử dụng file colab_test.ipynb để sử dụng trên colab
---------------------------------------------------------------------------------------

C. Đối với API
    - Chạy server: python -m uvicorn main:app --port 3000 --reload
