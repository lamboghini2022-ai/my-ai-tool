import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
from google import genai
from google.genai import types

app = FastAPI()

# Bật CORS để HTML Frontend (Render, Vercel, Localhost) có thể gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BỘ TỪ ĐIỂN DỊCH CHUẨN XÁC CHO AI GOOGLE ---
ANH_XA_GIOI_TINH = { "Nữ": "woman", "Nam": "man", "Khác": "person" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { 
    "Xanh": "professional blue studio background", 
    "Trắng": "pure white studio background", 
    "Xám": "neutral gray studio background", 
    "Xanh Đậm": "dark navy blue photography background" 
}

@app.post("/process-vest")
async def xu_ly_anh_chuyen_sau(
    # Nhận File
    image: UploadFile = File(...),
    outfit_image: UploadFile = File(None),
    
    # Nhận Params từ Frontend
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
    # 1. BẢO MẬT: LẤY API KEY TỪ BIẾN MÔI TRƯỜNG RENDER (BẢN FREE GOOGLE AI STUDIO)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Hệ thống chưa cài đặt GOOGLE_API_KEY."})

    try:
        # 2. XỬ LÝ ẢNH ĐẦU VÀO TỪ NGƯỜI DÙNG
        du_lieu_anh_raw = await image.read()
        mime_type = image.content_type or "image/jpeg"
        
        # 3. KỸ THUẬT SIÊU PROMPT (ADVANCED PROMPT ENGINEERING)
        gioi_tinh_eng = ANH_XA_GIOI_TINH.get(gender, "person")
        doi_tuong_eng = ANH_XA_DOI_TUONG.get(target, "adult")
        nen_anh_eng = ANH_XA_NEN.get(background, "solid background")
        
        # Thiết lập trang phục
        if prompt.strip():
            trang_phuc_eng = prompt.strip()
        else:
            trang_phuc_eng = "keep original exact clothing" if outfit == "Giữ nguyên" else f"wearing a {outfit}, perfectly fitted"

        # Thiết lập tóc
        kieu_toc_eng = "keep original exact hairstyle" if hair == "Giữ nguyên" else f"styled with {hair} hair"

        # Tùy chỉnh mức độ làm đẹp da (Skin Retouching)
        cum_tu_lam_dep = "natural skin texture"
        if beauty_enabled.lower() == "true":
            if beauty_level >= 80:
                cum_tu_lam_dep = "flawless glass skin, high-end beauty magazine retouching, completely smooth and clear complexion, radiant skin, eliminate all blemishes"
            elif beauty_level >= 40:
                cum_tu_lam_dep = "smooth skin, professional portrait retouching, remove acne and spots, maintain natural pores"
            else:
                cum_tu_lam_dep = "slight skin enhancement, very natural look"

        MASTER_PROMPT = f"""
        TASK: Professional Image Editing and Retouching.
        INSTRUCTION: You must strictly preserve the exact facial identity, facial features, and pose of the person in the provided input image. Do not change their face.
        
        SUBJECT: A {doi_tuong_eng} {gioi_tinh_eng}.
        EDITING INSTRUCTIONS:
        1. SKIN/BEAUTY: {cum_tu_lam_dep}.
        2. OUTFIT: {trang_phuc_eng}. Ensure the clothes look realistic and match the lighting.
        3. HAIR: {kieu_toc_eng}.
        4. BACKGROUND: Change the background to a {nen_anh_eng}. The subject must be perfectly cut out with no background bleeding.
        
        STYLE: Photorealistic, 8k resolution, RAW studio photography, sharp focus, perfect lighting.
        """

        # 4. KHỞI TẠO CLIENT GOOGLE GENAI
        client = genai.Client(api_key=api_key)

        # Định dạng ảnh đầu vào cho Gemini API
        anh_input = types.Part.from_bytes(
            data=du_lieu_anh_raw,
            mime_type=mime_type
        )

        # 5. GỌI MÔ HÌNH: Đã đổi sang Imagen 4 Ultra theo đúng hạn mức 0/25 của bạn
        model_name = 'imagen-4.0-ultra-generate' 
        
        response = client.models.generate_content(
            model=model_name,
            contents=[anh_input, MASTER_PROMPT],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            )
        )

        # 6. TRÍCH XUẤT ẢNH TRẢ VỀ TỪ GOOGLE AI
        chuoi_base64_tu_ai = None
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    chuoi_base64_tu_ai = base64.b64encode(part.inline_data.data).decode("utf-8")
                    break

        if not chuoi_base64_tu_ai:
            return JSONResponse(status_code=500, content={"error": "Google AI đã xử lý nhưng không thể tạo ra ảnh mới theo yêu cầu. Hãy thử thay đổi tùy chọn."})

        # 7. XỬ LÝ HẬU KỲ (POST-PROCESSING)
        anh_tu_ai_raw = base64.b64decode(chuoi_base64_tu_ai)
        hinh_anh = Image.open(io.BytesIO(anh_tu_ai_raw))

        if brightness_level != 50:
            he_so_sang = brightness_level / 50.0 
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so_sang)

        # 8. TRẢ KẾT QUẢ VỀ CHO FRONTEND
        bo_dem = io.BytesIO()
        hinh_anh.save(bo_dem, format="JPEG", quality=98) 
        chuoi_base64_cuoi_cung = base64.b64encode(bo_dem.getvalue()).decode("utf-8")

        return {"result": f"data:image/jpeg;base64,{chuoi_base64_cuoi_cung}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Lỗi trong quá trình xử lý: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
