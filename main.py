import os
import io
import base64
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance

app = FastAPI()

# Cấu hình CORS cho phép frontend HTML gọi API tới server Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model dữ liệu: Bắt buộc phải có ảnh gốc vì đây là phần mềm RỬA ẢNH (Image-to-Image / Restoration)
class YeuCauPhucHoi(BaseModel):
    anh_goc_base64: str  # Frontend BẮT BUỘC phải gửi ảnh mờ/ảnh cũ lên đây
    gioi_tinh: str = "Nữ"
    doi_tuong: str = "Người lớn"
    trang_phuc: str = "Giữ nguyên"
    prompt_trang_phuc: str = "" 
    kieu_toc: str = "Giữ nguyên"
    nen_anh: str = "Trắng"
    lam_dep_da: bool = True 
    thanh_truot_lam_dep: int = 50
    thanh_truot_do_sang: int = 50

# Từ điển ánh xạ sang tiếng Anh cho AI SUPIR
ANH_XA_GIOI_TINH = { "Nữ": "woman", "Nam": "man" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { "Xanh": "blue background", "Trắng": "white background", "Xám": "gray background", "Xanh Đậm": "dark navy blue background" }

@app.post("/api/xu-ly-anh")
async def xu_ly_anh(du_lieu: YeuCauPhucHoi):
    # 1. LẤY API KEY TỪ SERVER RENDER (Tuyệt đối không lấy từ Frontend)
    # Trên trang quản trị Render > Environment Variables, bạn phải tạo biến tên là "API_KEY_AI_PROVIDER"
    api_key = os.getenv("API_KEY_AI_PROVIDER")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Lỗi Server Render: Chưa cấu hình biến môi trường API_KEY_AI_PROVIDER."})

    # 2. XÂY DỰNG PROMPT CHI TIẾT CHO SUPIR
    gioi_tinh = ANH_XA_GIOI_TINH.get(du_lieu.gioi_tinh, "person")
    doi_tuong = ANH_XA_DOI_TUONG.get(du_lieu.doi_tuong, "adult")
    nen_anh = ANH_XA_NEN.get(du_lieu.nen_anh, "solid background")
    
    # Xử lý trang phục và tóc
    trang_phuc = du_lieu.prompt_trang_phuc if du_lieu.prompt_trang_phuc.strip() else (
        "original clothes" if du_lieu.trang_phuc == "Giữ nguyên" else f"wearing {du_lieu.trang_phuc}"
    )
    kieu_toc = "original hairstyle" if du_lieu.kieu_toc == "Giữ nguyên" else f"{du_lieu.kieu_toc} hairstyle"

    # Xử lý làm đẹp
    cum_tu_lam_dep = "highly detailed face, realistic skin texture"
    if du_lieu.lam_dep_da:
        if du_lieu.thanh_truot_lam_dep >= 80:
            cum_tu_lam_dep = "flawless smooth skin, clear complexion, high-end studio retouching, perfect skin"
        elif du_lieu.thanh_truot_lam_dep >= 40:
            cum_tu_lam_dep = "smooth skin, professional photo restoration, sharp details"

    # Prompt chuyên dụng cho SUPIR (tập trung vào khôi phục và giữ chi tiết)
    PROMPT_TEXT = f"High quality photo restoration of a {doi_tuong} {gioi_tinh}, {cum_tu_lam_dep}. {trang_phuc}, {kieu_toc}, {nen_anh}. 8k resolution, highly detailed, sharp focus, RAW photo."
    NEGATIVE_PROMPT = "oil painting, cartoon, blur, dirty, oversharpened, deformed, ugly face, unnatural skin, bad anatomy, mutated"

    # 3. GỌI API ĐẾN DỊCH VỤ CUNG CẤP SUPIR (Ví dụ: Novita, Runpod, Replicate...)
    # VUI LÒNG THAY ĐỔI URL DƯỚI ĐÂY THÀNH ENDPOINT SUPIR THỰC TẾ CỦA BẠN
    AI_API_URL = "https://api.your-ai-provider.com/v1/supir/restore" 
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Payload chuẩn để gửi cho mô hình SUPIR
    payload = {
        "image": du_lieu.anh_goc_base64, # Ảnh mờ người dùng upload
        "prompt": PROMPT_TEXT,
        "negative_prompt": NEGATIVE_PROMPT,
        "upscale": 2,            # Tăng độ phân giải gấp đôi
        "fidelity_weight": 0.5,  # Từ 0.0 đến 1.0 (0.5 là mức cân bằng giữa giữ nét cũ và làm đẹp mới)
        "steps": 30,
        "cfg_scale": 7.0
    }

    try:
        # 4. THỰC THI REQUEST
        response = requests.post(AI_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi từ AI Provider: {response.text}"})

        # Xử lý kết quả trả về từ AI
        ket_qua = response.json()
        
        # Lưu ý: Sửa lại key 'output_image' này tùy thuộc vào JSON mà dịch vụ AI của bạn trả về
        chuoi_base64_tu_ai = ket_qua.get('output_image') 
        
        if not chuoi_base64_tu_ai:
            return JSONResponse(status_code=500, content={"error": "AI xử lý thành công nhưng không trả về dữ liệu ảnh."})

        # 5. XỬ LÝ ĐỘ SÁNG ẢNH BẰNG PYTHON (Xử lý hậu kỳ sau khi AI rửa ảnh xong)
        du_lieu_anh_raw = base64.b64decode(chuoi_base64_tu_ai)
        hinh_anh = Image.open(io.BytesIO(du_lieu_anh_raw))

        if du_lieu.thanh_truot_do_sang != 50:
            he_so_sang = du_lieu.thanh_truot_do_sang / 50.0  # Mức 50 = 1.0 (Giữ nguyên gốc)
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so_sang)

        # 6. ĐÓNG GÓI TRẢ VỀ CHO GIAO DIỆN HTML
        bo_dem = io.BytesIO()
        hinh_anh.save(bo_dem, format="JPEG", quality=95)
        chuoi_base64_cuoi_cung = base64.b64encode(bo_dem.getvalue()).decode("utf-8")

        return {"thanh_cong": True, "hinh_anh": f"data:image/jpeg;base64,{chuoi_base64_cuoi_cung}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Lỗi máy chủ Render: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    # Chạy trên 0.0.0.0 để Render có thể expose port ra ngoài
    uvicorn.run(app, host="0.0.0.0", port=8000)
