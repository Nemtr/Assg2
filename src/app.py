import streamlit as st
import joblib
import os
import pandas as pd
from features import extract_video_features, calculate_video_psnr
from video_coder import encode_video_dynamic

st.set_page_config(page_title="Video ML Optimizer", layout="wide")
st.title("📹 HỆ THỐNG TỐI ƯU HÓA NÉN VIDEO BẰNG AI (2502k)")
st.caption("Sinh viên thực hiện: Trần Lê Hải Nam & Nguyễn Hữu Mạnh")

@st.cache_resource
def load_models():
    qp_model = joblib.load('models/model_qp.pkl')
    gop_model = joblib.load('models/model_gop.pkl')
    return qp_model, gop_model

try:
    model_qp, model_gop = load_models()
except Exception:
    st.error("❌ Không tìm thấy file Model AI trong thư mục models/!")
    st.stop()

st.markdown("### 1. Tải Video Cần Tối Ưu")
uploaded_file = st.file_uploader("Kéo thả file video (.mp4) vào đây", type=["mp4"])

if uploaded_file is not None:
    os.makedirs("data/temp", exist_ok=True)
    
    original_name = uploaded_file.name
    input_path = os.path.join("data/temp", original_name)
    output_path = os.path.join("data/temp", "compressed_" + original_name)
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("✅ Đã tải file lên thành công!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎬 Video Gốc")
        st.video(input_path)
        
    with col2:
        st.markdown("### 🧠 AI Phân Tích & Dự Đoán")
        with st.spinner("Đang tính toán đặc trưng hệ thống..."):
            features = extract_video_features(input_path)
            
        if features:
            st.info(f"📊 **SI Mean:** {features['si_mean']:.2f} | **TI Mean:** {features['ti_mean']:.2f}")
            
            input_df = pd.DataFrame([{
                'si_mean': features['si_mean'],
                'si_std': features['si_std'],
                'ti_mean': features['ti_mean'],
                'ti_std': features['ti_std']
            }])
            
            pred_qp = int(model_qp.predict(input_df)[0])
            pred_gop = int(model_gop.predict(input_df)[0])
            
            m1, m2 = st.columns(2)
            m1.metric("Tham số QP Đề Xuất", pred_qp)
            m2.metric("Tham số GOP Đề Xuất", pred_gop)
            
            st.markdown("---")
            st.markdown("### ⚙️ Bắt Đầu Nén")
            
            if st.button("🚀 Chạy FFmpeg với Cấu hình AI", use_container_width=True):
                with st.spinner("Đang thực hiện tối ưu hóa cấu hình..."):
                    success, orig_sz, comp_sz = encode_video_dynamic(input_path, output_path, pred_qp, pred_gop)

                if success:
                    st.balloons()
                    savings = ((orig_sz - comp_sz) / orig_sz) * 100
                    
                    # TÍNH TOÁN PSNR THỰC TẾ GIỮA 2 FILE
                    with st.spinner("📊 Đang phân tích độ trung thực hình ảnh (PSNR)..."):
                        avg_psnr = calculate_video_psnr(input_path, output_path)
                    
                    st.markdown("#### 📈 Báo cáo Hiệu năng hệ thống (Bitrate Savings & Quality Demo)")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Dung lượng Gốc", f"{orig_sz:.2f} MB")
                    r2.metric("Dung lượng Nén", f"{comp_sz:.2f} MB")
                    r3.metric("Băng thông Tiết kiệm", f"{savings:.1f}%", delta=f"-{savings:.1f}%")
                    
                    # Đánh giá chất lượng dựa trên thang đo tiêu chuẩn PSNR
                    if avg_psnr >= 40:
                        quality_status = "Xuất sắc (Giống gốc)"
                    elif avg_psnr >= 30:
                        quality_status = "Tốt (Mắt thường khó phân biệt)"
                    else:
                        quality_status = "Có suy hao chi tiết"
                        
                    r4.metric("Chất lượng (Avg PSNR)", f"{avg_psnr:.1f} dB", help=quality_status)
                    
                    st.markdown("### 🎬 Video Đã Tối Ưu")
                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    
                    st.download_button(
                        label="💾 Tải Video Sau Nén Về Máy",
                        data=video_bytes,
                        file_name="optimized_" + original_name,
                        mime="video/mp4",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Có lỗi xảy ra trong quá trình nén cấu hình.")
        else:
            st.error("❌ Không thể trích xuất đặc trưng từ video này!")
