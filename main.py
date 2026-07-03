import os
import httpx
import urllib.parse
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AI Image Generator Backend")

# ==========================================
# CẤU HÌNH CORS MIDDLEWARE (Cho phép Web gọi API)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bạn có thể thay "*" thành tên miền web của bạn sau này
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === BỘ TỪ ĐIỂN DỊCH GIAO DIỆN SANG PROMPT ===
PROMPT_MAP = {
    "gender": {"Nữ": "female", "Nam": "male"},
    "target": {"Người lớn": "adult", "Thanh niên": "young adult in their 20s", "Trẻ em": "child"},
    "outfit": {
        "Giữ nguyên": "", "Áo Sơ mi": "wearing a casual button-down shirt", 
        "Áo Sơ mi Trắng": "wearing a crisp white dress shirt", "Áo Polo": "wearing a smart polo shirt", 
        "Áo kiểu": "wearing a stylish designer blouse", "Áo phông trơn": "wearing a plain simple t-shirt", 
        "Áo Vest": "wearing a tailored formal suit jacket", "Công sở": "wearing professional business attire", 
        "Vest nữ công sở 2": "wearing a chic female business suit", 
        "Áo trắng & Khăn quàng": "wearing a white top with an elegant silk scarf", 
        "Áo dài trắng": "wearing a traditional Vietnamese white Ao Dai", 
        "Nữ Sinh HQ 1": "wearing a Korean high school uniform", 
        "Nữ Sinh HQ 2": "wearing a stylish Korean student outfit", 
        "Nữ Sinh HQ 3": "wearing a Korean prep school uniform with a tie"
    },
    "hair": {
        "Giữ nguyên": "", "Gọn gàng": "neatly styled hair", "Tóc ngắn": "short hair", 
        "Tóc dài": "long flowing hair", "Tóc dài bồng bềnh": "voluminous long wavy hair", 
        "Thời trang": "trendy fashionable haircut", "Tóc buộc gọn": "hair tied up neatly in a ponytail", 
        "Texture Crop NAM": "textured crop haircut for men", "Rẽ đôi HQ Nam": "Korean two-block parted hair for men", 
        "Xuân Ngắn Nam": "short messy textured hair for men", "Hạt Dẻ Ngố Nam": "chestnut brown bowl cut for men"
    },
    "background": {
        "Xanh": "solid light blue background", "Trắng": "pure white studio background", 
        "Xám": "solid neutral grey background", "Xanh Đậm": "solid dark navy blue background"
    }
}

# ==========================================
# KHAI BÁO DỮ LIỆU NHẬN TỪ WEB (Đã thêm image_base64)
# ==========================================
class GenerateRequest(BaseModel):
    gender: Optional[str] = ""
    target: Optional[str] = ""
    outfit: Optional[str] = ""
    custom_outfit: Optional[str] = ""
    hair: Optional[str] = ""
    background: Optional[str] = ""
    beauty_level: Optional[int] = 50
    brightness_level: Optional[int] = 50
    smooth_skin: Optional[bool] = True
    image_base64: Optional[str] = ""  # Nhận ảnh gốc từ frontend

def generate_base_prompt(data: GenerateRequest) -> str:
    """Tạo câu lệnh thô từ các nút bấm trên giao diện"""
    gender = PROMPT_MAP["gender"].get(data.gender, "person")
    target = PROMPT_MAP["target"].get(data.target, "adult")
    
    prompt_parts = [f"A portrait of a {target} {gender}"]

    if data.custom_outfit: 
        prompt_parts.append(f"wearing {data.custom_outfit.strip()}")
    elif data.outfit and PROMPT_MAP["outfit"].get(data.outfit): 
        prompt_parts.append(PROMPT_MAP["outfit"][data.outfit])
        
    if data.hair and PROMPT_MAP["hair"].get(data.hair): 
        prompt_parts.append(f"with {PROMPT_MAP['hair'][data.hair]}")

    main_prompt = ", ".join(prompt_parts) + "."
    if data.background and PROMPT_MAP["background"].get(data.background): 
        main_prompt += f" Background is {PROMPT_MAP['background'][data.background]}."

    return main_prompt

# ==========================================
# API CHECK SERVER
# ==========================================
@app.get("/")
async def root_endpoint():
    return {"message": "Server FastAPI Tạo & Chỉnh Sửa Ảnh AI đang hoạt động tuyệt vời trên Render!"}

# ==========================================
# API TẠO & CHỈNH SỬA ẢNH CHÍNH
# ==========================================
@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    print("\n========== BẮT ĐẦU QUÁ TRÌNH XỬ LÝ ẢNH ==========")
    
    # 1. Lấy API Key từ Environment Variable
    gemini_key = os.getenv("MY_API_KEY")
    stability_key = os.getenv("STABILITY_API_KEY") # API key mới dùng để sửa ảnh

    if not gemini_key:
        return JSONResponse(status_code=403, content={"error": {"message": "Chưa cấu hình MY_API_KEY cho Gemini!"}})

    # 2. Sinh ra Prompt cơ bản
    base_prompt = generate_base_prompt(req)

    # 3. GỌI GEMINI NÂNG CẤP PROMPT
    print(f"Đang gửi yêu cầu cho Gemini... Prompt thô: {base_prompt}")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    
    instruction = (
        "You are an expert AI prompt engineer for photography. "
        "Rewrite the following basic description into a highly detailed, professional, photorealistic prompt for an ID card style portrait. "
        "ONLY output the English prompt, no conversational text, no markdown. "
        f"Basic description: {base_prompt}"
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(gemini_url, json={"contents": [{"parts": [{"text": instruction}]}]})
            if resp.status_code == 200:
                final_prompt = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                final_prompt = base_prompt
    except Exception as e:
        print(f"Lỗi gọi Gemini: {e}")
        final_prompt = base_prompt

    print(f"Prompt cuối cùng: {final_prompt}")

    # ========================================================
    # 4. QUYẾT ĐỊNH: TẠO ẢNH MỚI HAY CHỈNH SỬA ẢNH CŨ?
    # ========================================================
    
    # TRƯỜNG HỢP A: NGƯỜI DÙNG CÓ TẢI ẢNH LÊN (IMAGE-TO-IMAGE)
    if req.image_base64:
        print("Phát hiện ảnh gốc! Đang kích hoạt chế độ chỉnh sửa ảnh (Image-to-Image)...")
        
        # Nếu chưa cài STABILITY_API_KEY, báo lỗi nhẹ nhàng để biết đường cài
        if not stability_key:
            return JSONResponse(status_code=500, content={
                "error": {"message": "Hệ thống cần STABILITY_API_KEY trong biến môi trường Render để dùng chức năng chỉnh sửa ảnh thật."}
            })

        # Xử lý chuỗi base64 (cắt bỏ phần 'data:image/jpeg;base64,' dư thừa)
        base64_data = req.image_base64.split(",")[1] if "," in req.image_base64 else req.image_base64
        image_bytes = base64.b64decode(base64_data)

        # Gọi API của Stability AI (Endpoint chỉnh sửa ảnh)
        stability_url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/image-to-image"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    stability_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {stability_key}"
                    },
                    files={
                        "init_image": ("image.png", image_bytes, "image/png")
                    },
                    data={
                        "image_strength": 0.55, # Độ mạnh (0.0 đến 1.0) - Càng thấp thì AI càng thay đổi nhiều so với ảnh gốc
                        "text_prompts[0][text]": final_prompt,
                        "text_prompts[0][weight]": 1.0,
                        "cfg_scale": 7,
                        "samples": 1,
                        "steps": 30,
                    }
                )

                if response.status_code != 200:
                    return JSONResponse(status_code=500, content={"error": {"message": f"Lỗi từ Stability AI: {response.text}"}})

                data = response.json()
                # Stability AI trả về ảnh dạng base64
                result_base64 = data["artifacts"][0]["base64"]
                final_image_url = f"data:image/png;base64,{result_base64}"
                message = "Đã chỉnh sửa ảnh (Image-to-Image) thành công!"
                
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": {"message": f"Lỗi trong quá trình xử lý ảnh gốc: {str(e)}"}})

    # TRƯỜNG HỢP B: NGƯỜI DÙNG KHÔNG TẢI ẢNH LÊN (TEXT-TO-IMAGE)
    else:
        print("Không có ảnh gốc. Tiến hành tạo ảnh ảo từ văn bản (Text-to-Image)...")
        safe_prompt = urllib.parse.quote(final_prompt)
        final_image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=600&height=800&nologo=true"
        message = "Không có ảnh gốc, đã tạo nhân vật mới thành công!"

    # 5. TRẢ DỮ LIỆU VỀ CHO GIAO DIỆN WEB
    return JSONResponse(content={
        "status": "success",
        "prompt_used": final_prompt,
        "message": message,
        "image_url": final_image_url 
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
