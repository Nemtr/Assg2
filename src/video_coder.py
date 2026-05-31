import subprocess
import os

def encode_video_dynamic(input_path, output_path, qp, gop):
    # Sử dụng CRF thay vì QP để tối ưu hóa luồng dữ liệu tự động
    # Giới hạn dưới: Nếu model dự đoán CRF quá thấp (dưới 26), ép lên tối thiểu 26 để chống phình dung lượng
    crf_val = max(int(qp), 26) 
    
    command = [
        'ffmpeg', '-y',             
        '-i', input_path,           
        '-c:v', 'libx264',          
        '-crf', str(crf_val),  # Đổi từ -qp sang -crf      
        '-g', str(int(gop)),        
        '-preset', 'fast', 
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path                 
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False, 0, 0

        orig_size = os.path.getsize(input_path) / (1024 * 1024)
        comp_size = os.path.getsize(output_path) / (1024 * 1024)
        # Nếu file sau nén nặng hơn file gốc, copy đè file gốc sang output
        if comp_size >= orig_size:
            shutil.copyfile(input_path, output_path)
            comp_size = orig_size   
        return True, orig_size, comp_size
    
    except Exception:
        return False, 0, 0