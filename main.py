import os
import io
import base64
import requests
from fastapi import FastAPI, UploadFile, File, Form
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

# Từ điển ánh xạ sang tiếng Anh cho AI Google
ANH_XA_GIOI_TINH = { "Nữ": "woman", "Nam": "man" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { "Xanh": "blue background", "Trắng": "white background", "Xám": "gray background", "Xanh Đậm": "dark navy blue background" }

# Khớp API Endpoint với địa chỉ fetch trong file HTML của bạn
@app.post("/process-vest")
async def xu_ly_anh(
    # Nhận File ảnh nhị phân trực tiếp từ FormData
    image: UploadFile = File(...),
    outfit_image: UploadFile = File(None), # Ảnh quần áo mẫu (nếu có tải lên)
    
    # Nhận các thông số text từ FormData
    model: str = Form("Nano Banana"),
    gender: str = Form("Nữ"),
    target: str = Form("Người lớn"),
    outfit: str = Form("Giữ nguyên"),
    hair: str = Form("Giữ nguyên"),
    background: str = Form("Trắng"),
    prompt: str = Form(""),
    beauty_enabled: str = Form("true"),
    beauty_level: int = Form(50),
    brightness_level: int = Form(50)
):
    # 1. LẤY API KEY TỪ SERVER RENDER (Bảo mật, không nhận từ Frontend)
    # Trên trang quản trị Render > Environment Variables, tạo biến "GOOGLE_API_KEY"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Lỗi Server: Chưa cấu hình biến môi trường GOOGLE_API_KEY trên Render."})

    try:
        # Chuyển ảnh UploadFile thành dạng Base64 để gửi cho AI của Google
        du_lieu_anh_raw = await image.read()
        anh_goc_base64 = base64.b64encode(du_lieu_anh_raw).decode("utf-8")

        # 2. XÂY DỰNG PROMPT CHI TIẾT
        gioi_tinh_eng = ANH_XA_GIOI_TINH.get(gender, "person")
        doi_tuong_eng = ANH_XA_DOI_TUONG.get(target, "adult")
        nen_anh_eng = ANH_XA_NEN.get(background, "solid background")
        
        # Xử lý trang phục và tóc
        trang_phuc_eng = prompt if prompt.strip() else (
            "original clothes" if outfit == "Giữ nguyên" else f"wearing {outfit}"
        )
        kieu_toc_eng = "original hairstyle" if hair == "Giữ nguyên" else f"{hair} hairstyle"

        # Xử lý làm đẹp
        cum_tu_lam_dep = "highly detailed face, realistic skin texture"
        if beauty_enabled.lower() == "true":
            if beauty_level >= 80:
                cum_tu_lam_dep = "flawless smooth skin, clear complexion, high-end studio retouching, perfect skin"
            elif beauty_level >= 40:
                cum_tu_lam_dep = "smooth skin, professional photo retouching, sharp details"

        # Prompt chuyên dụng cho Google Image Edit
        PROMPT_TEXT = f"A photorealistic portrait of a {doi_tuong_eng} {gioi_tinh_eng}, {cum_tu_lam_dep}. {trang_phuc_eng}, {kieu_toc_eng}, {nen_anh_eng}. 8k resolution, highly detailed, sharp focus, RAW studio lighting."
        NEGATIVE_PROMPT = "cartoon, painting, deformed, ugly face, unnatural skin, bad anatomy, mutated, blurry, bad lighting"

        # 3. GỌI API ĐẾN GOOGLE (Gemini Image Generation / Imagen)
        # Sử dụng cấu trúc payload chuẩn cho Image-to-Image của Google AI
        GOOGLE_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:predict?key={api_key}" 
        
        headers = {
            "Content-Type": "application/json"
        }

        # Payload theo chuẩn Imagen
        payload = {
            "instances": [
                {
                    "prompt": PROMPT_TEXT,
                    "image": {
                        "bytesBase64Encoded": anh_goc_base64
                    }
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "negativePrompt": NEGATIVE_PROMPT,
                "mode": "edit" # Báo cho AI biết đây là tác vụ giữ nguyên mặt, sửa chi tiết
            }
        }

        # 4. THỰC THI REQUEST
        response = requests.post(GOOGLE_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi từ Google AI: {response.text}"})

        # Xử lý kết quả trả về từ Google
        ket_qua = response.json()
        
        # Google Imagen thường trả về base64 trong mảng predictions
        danh_sach_anh = ket_qua.get('predictions', [])
        if not danh_sach_anh or 'bytesBase64Encoded' not in danh_sach_anh[0]:
            return JSONResponse(status_code=500, content={"error": "AI xử lý thành công nhưng không có dữ liệu ảnh trả về."})
            
        chuoi_base64_tu_ai = danh_sach_anh[0]['bytesBase64Encoded']

        # 5. XỬ LÝ ĐỘ SÁNG ẢNH BẰNG PYTHON (Hậu kỳ)
        anh_tu_ai_raw = base64.b64decode(chuoi_base64_tu_ai)
        hinh_anh = Image.open(io.BytesIO(anh_tu_ai_raw))

        if brightness_level != 50:
            he_so_sang = brightness_level / 50.0  # Mức 50 = 1.0 (Giữ nguyên)
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so_sang)

        # 6. ĐÓNG GÓI TRẢ VỀ CHO GIAO DIỆN HTML
        bo_dem = io.BytesIO()
        hinh_anh.save(bo_dem, format="JPEG", quality=95)
        chuoi_base64_cuoi_cung = base64.b64encode(bo_dem.getvalue()).decode("utf-8")

        # Chuỗi khóa "result" phải khớp chuẩn với data.result trong file HTML
        return {"result": f"data:image/jpeg;base64,{chuoi_base64_cuoi_cung}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Lỗi máy chủ: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    # Chạy trên 0.0.0.0 để Render có thể nhận port
    uvicorn.run(app, host="0.0.0.0", port=8000)
