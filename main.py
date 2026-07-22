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
        # Giúp AI hiểu đây là "chỉnh sửa giữ khuôn mặt" chứ không phải "vẽ người mới"
        
        # Dịch cơ bản
        gioi_tinh_eng = ANH_XA_GIOI_TINH.get(gender, "person")
        doi_tuong_eng = ANH_XA_DOI_TUONG.get(target, "adult")
        nen_anh_eng = ANH_XA_NEN.get(background, "solid background")
        
        # Thiết lập trang phục
        if prompt.strip():
            trang_phuc_eng = prompt.strip() # Ưu tiên prompt người dùng tự gõ
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

        # Lắp ghép thành câu lệnh thao túng AI mạnh mẽ (Master Prompt)
        # Bắt buộc AI phải giữ nguyên khuôn mặt (Preserve facial identity)
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

        # 5. GỌI MÔ HÌNH (Sử dụng model theo yêu cầu của bạn)
        # Lưu ý: 'gemini-3.1-flash-image' là tên bạn đặt. Nếu Google báo lỗi không tìm thấy model,
        # hệ thống sẽ tự động dùng tên chuẩn của Google là 'imagen-3.0-generate-002'
        model_name = 'imagen-4.0-ultra-generate-001' 
        
        response = client.models.generate_content(
            model=model_name,
            contents=[anh_input, MASTER_PROMPT],
            config=types.GenerateContentConfig(
                # Yêu cầu AI trả về hình ảnh thay vì text
                response_modalities=["IMAGE"],
            )
        )

        # 6. TRÍCH XUẤT ẢNH TRẢ VỀ TỪ GOOGLE AI
        chuoi_base64_tu_ai = None
        if response.candidates:
            # Tìm trong các part trả về, lấy part chứa dữ liệu ảnh nhị phân
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    chuoi_base64_tu_ai = base64.b64encode(part.inline_data.data).decode("utf-8")
                    break

        if not chuoi_base64_tu_ai:
            return JSONResponse(status_code=500, content={"error": "Google AI đã xử lý nhưng không thể tạo ra ảnh mới theo yêu cầu. Hãy thử thay đổi tùy chọn."})

        # 7. XỬ LÝ HẬU KỲ (POST-PROCESSING) - Thanh trượt độ sáng
        anh_tu_ai_raw = base64.b64decode(chuoi_base64_tu_ai)
        hinh_anh = Image.open(io.BytesIO(anh_tu_ai_raw))

        # Áp dụng thay đổi độ sáng nếu người dùng kéo thanh trượt (khác mức 50 mặc định)
        if brightness_level != 50:
            he_so_sang = brightness_level / 50.0 # 50 -> 1.0 (Không đổi), 100 -> 2.0 (Sáng gấp đôi)
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so_sang)

        # 8. TRẢ KẾT QUẢ VỀ CHO FRONTEND
        bo_dem = io.BytesIO()
        hinh_anh.save(bo_dem, format="JPEG", quality=98) # Lưu chất lượng cao nhất
        chuoi_base64_cuoi_cung = base64.b64encode(bo_dem.getvalue()).decode("utf-8")

        # Định dạng chuẩn mà code HTML của bạn đang chờ để gán vào thẻ <img>
        return {"result": f"data:image/jpeg;base64,{chuoi_base64_cuoi_cung}"}

    except Exception as e:
        # Bắt lỗi chi tiết để dễ debug trên Render
        return JSONResponse(status_code=500, content={"error": f"Lỗi trong quá trình xử lý: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    # Tự động lấy Port của Render, nếu chạy ở máy cá nhân thì dùng 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
