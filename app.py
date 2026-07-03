import os
import httpx
import urllib.parse
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

# === BỘ TỪ ĐIỂN DỊCH GIAO DIỆN SANG PROMPT CƠ BẢN ===
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
# KHAI BÁO DỮ LIỆU NHẬN TỪ WEB (PYDANTIC MODEL)
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
    return {"message": "Server FastAPI Tạo Ảnh AI đang hoạt động tuyệt vời trên Render!"}

# ==========================================
# API TẠO ẢNH CHÍNH (GỌI GEMINI + VẼ ẢNH)
# ==========================================
@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    print("\n========== BẮT ĐẦU QUÁ TRÌNH TẠO ẢNH ==========")
    
    # 1. Lấy API Key từ Environment Variable của Render
    api_key = os.getenv("MY_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=403, 
            content={"error": {"message": "Chưa cấu hình MY_API_KEY trên biến môi trường Render!"}}
        )

    # 2. Sinh ra Prompt cơ bản
    base_prompt = generate_base_prompt(req)

    # 3. GỌI GEMINI 2.5 FLASH NÂNG CẤP PROMPT
    print(f"Đang gửi yêu cầu cho Gemini... Prompt thô: {base_prompt}")
    model_name = "gemini-2.5-flash"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    instruction = (
        "You are an expert AI prompt engineer for photography. "
        "Rewrite the following basic description into a highly detailed, professional, 8k resolution, photorealistic prompt for an ID card style portrait. "
        "ONLY output the English prompt, no conversational text, no markdown. "
        f"Basic description: {base_prompt}"
    )
    
    payload = {"contents": [{"parts": [{"text": instruction}]}]}

    try:
        # Dùng httpx để chạy bất đồng bộ, giúp server không bị đơ
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(gemini_url, json=payload)
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=500, 
                    content={"error": {"message": f"Lỗi từ Google Gemini: {response.text}"}}
                )

            data = response.json()
            final_prompt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
    except Exception as e:
        print(f"Lỗi khi gọi Gemini: {e}")
        final_prompt = base_prompt  # Nếu Gemini lỡ tay báo lỗi, dùng tạm bản thô

    print(f"Gemini đã viết xong: {final_prompt}")

    # 4. TRẢ VỀ ẢNH THẬT
    # Biến url thành định dạng an toàn cho duyệt web
    safe_prompt = urllib.parse.quote(final_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=600&height=800&nologo=true"

    # 5. Trả dữ liệu về giao diện web
    return JSONResponse(content={
        "status": "success",
        "prompt_used": final_prompt,
        "message": "Đã tạo ảnh chân dung thành công với Gemini 2.5 Flash!",
        "image_url": image_url 
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
