import cv2
import numpy as np
import pandas as pd
import os
import glob

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    si_values = []
    ti_values = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        si_values.append(np.std(magnitude))

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            ti_values.append(np.std(diff))
            
        prev_gray = gray

    cap.release()

    if len(si_values) == 0:
        return None

    return {
        'si_mean': np.mean(si_values),
        'si_std': np.std(si_values),
        'ti_mean': np.mean(ti_values) if ti_values else 0,
        'ti_std': np.std(ti_values) if ti_values else 0
    }

def generate_heuristic_labels(si_mean, ti_mean):
    # Thiết lập cấu hình mặc định (Base)
    qp = 26
    gop = 30
    
    # 1. Xác định QP dựa trên tổ hợp của cả chuyển động (TI) và chi tiết (SI)
    if ti_mean < 8:  # Video cực ít chuyển động (như slide, ngồi nói chuyện tĩnh)
        if si_mean > 50:
            qp = 30  # Chi tiết cao nhưng tĩnh, có thể nén mạnh mà không sợ vỡ hình
        else:
            qp = 34  # Tĩnh và ít chi tiết, nén tối đa
            
    elif ti_mean > 20:  # Video chuyển động nhanh, hành động phức tạp
        if si_mean > 50:
            qp = 22  # Chuyển động nhanh + chi tiết cao -> Giữ chất lượng tốt nhất, chấp nhận file to
        else:
            qp = 24  # Chuyển động nhanh nhưng ít chi tiết
            
    else:  # Chuyển động trung bình (8 <= ti_mean <= 20)
        if si_mean > 50:
            qp = 25
        else:
            qp = 28

    # 2. Xác định GOP dựa trên tần suất thay đổi chuyển động (TI)
    # Video càng tĩnh thì khoảng cách giữa các khung khóa (GOP) càng dài để giảm dung lượng
    if ti_mean < 8:
        gop = 90
    elif ti_mean > 20:
        gop = 25
    else:
        gop = 50
        
    return qp, gop

def calculate_video_psnr(video_path_orig, video_path_comp):
    """
    Tính toán chỉ số PSNR trung bình giữa video gốc và video sau nén
    """
    cap_orig = cv2.VideoCapture(video_path_orig)
    cap_comp = cv2.VideoCapture(video_path_comp)
    
    psnr_values = []
    
    while True:
        ret_o, frame_o = cap_orig.read()
        ret_c, frame_c = cap_comp.read()
        
        # Nếu một trong hai video hết khung hình thì dừng
        if not ret_o or not ret_c:
            break
            
        # Chuyển sang hệ xám để tính toán sai số cấu trúc pixel
        gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
        
        # Đảm bảo hai khung hình cùng kích thước (đề phòng trường hợp có thay đổi độ phân giải)
        if gray_o.shape != gray_c.shape:
            gray_c = cv2.resize(gray_c, (gray_o.shape[1], gray_o.shape[0]))
            
        # Tính toán PSNR giữa 2 khung hình bằng hàm có sẵn của OpenCV
        psnr_frame = cv2.PSNR(gray_o, gray_c)
        
        # Loại bỏ các giá trị vô hạn (khi 2 khung hình giống nhau 100%)
        if psnr_frame < 100:
            psnr_values.append(psnr_frame)
            
    cap_orig.release()
    cap_comp.release()
    
    if len(psnr_values) == 0:
        return 100.0  # Hai video giống hệt nhau tuyệt đối
        
    return np.mean(psnr_values)

def process_dataset_to_csv(raw_dir="data/raw", output_csv="data/video_dataset.csv"):
    video_files = glob.glob(os.path.join(raw_dir, "*.mp4"))
    if not video_files:
        return

    dataset = []
    for video_path in video_files:
        filename = os.path.basename(video_path)
        features = extract_video_features(video_path)
        if features:
            qp_label, gop_label = generate_heuristic_labels(features['si_mean'], features['ti_mean'])
            row_data = {'filename': filename, **features, 'target_qp': qp_label, 'target_gop': gop_label}
            dataset.append(row_data)

    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    process_dataset_to_csv()
