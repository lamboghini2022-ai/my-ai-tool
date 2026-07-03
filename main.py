import os
import httpx
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AI Photo Editor Backend (Image-to-Image)")

# ==========================================
# CẤU HÌNH CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === TỪ ĐIỂN PROMPT ===
PROMPT_MAP = {
    "gender": {"Nữ": "female", "Nam": "male"},
    "target": {"Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child"},
    "outfit": {
        "Giữ nguyên": "", 
        "Áo Sơ mi": "wearing a casual button-down shirt", 
        "Áo Sơ mi Trắng": "wearing a crisp white dress shirt", 
        "Áo Polo": "wearing a smart polo shirt", 
        "Áo kiểu": "wearing a stylish designer blouse", 
        "Công sở": "wearing professional business attire"
    },
    "hair": {
        "Giữ nguyên": "", 
        "Gọn gàng": "neatly styled hair", 
        "Tóc ngắn": "short hair", 
        "Tóc dài": "long flowing hair"
    },
    "background": {
        "Xanh": "blue background", 
        "Trắng": "pure white studio background", 
        "Xám": "grey background"
    }
}

# ==========================================
# KHAI BÁO DỮ LIỆU TỪ GIAO DIỆN GỬI LÊN (Khớp với JS)
# ==========================================
class GenerateRequest(BaseModel):
    api_key: str = "" # Nhận API key từ ô input trên web
    gender: Optional[str] = ""
    target: Optional[str] = ""
    outfit: Optional[str] = ""
    custom_outfit: Optional[str] = ""
    hair: Optional[str] = ""
    background: Optional[str] = ""
    image_base64: str = "" # Dữ liệu ảnh gốc

def build_edit_prompt(data: GenerateRequest) -> str:
    """Tạo câu lệnh tiếng Anh mô tả BỨC ẢNH MONG MUỐN SAU KHI SỬA"""
    gender = PROMPT_MAP["gender"].get(data.gender, "person")
    target = PROMPT_MAP["target"].get(data.target, "")
    
    parts = [f"A photorealistic portrait of a {target} {gender}"]

    # Trang phục
    if data.custom_outfit:
        parts.append(f"wearing {data.custom_outfit.strip()}")
    elif data.outfit and PROMPT_MAP["outfit"].get(data.outfit):
        parts.append(PROMPT_MAP["outfit"][data.outfit])

    # Tóc
    if data.hair and PROMPT_MAP["hair"].get(data.hair):
        parts.append(f"with {PROMPT_MAP['hair'][data.hair]}")

    prompt = ", ".join(parts) + ", high quality, 8k resolution, highly detailed face."
    
    # Nền
    if data.background and PROMPT_MAP["background"].get(data.background):
        prompt += f" {PROMPT_MAP['background'][data.background]}."

    return prompt

@app.get("/")
async def root():
    return {"message": "Server Image-to-Image đang hoạt động!"}

# ==========================================
# API XỬ LÝ CHÍNH
# ==========================================
@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    print("\n========== NHẬN YÊU CẦU CHỈNH SỬA ẢNH ==========")
    
    if not req.image_base64:
        return JSONResponse(status_code=400, content={"error": "Vui lòng tải ảnh gốc lên."})
        
    if not req.api_key:
        return JSONResponse(status_code=403, content={"error": "Vui lòng nhập API Key của Stability AI trên giao diện web."})

    # 1. Tiền xử lý chuỗi Base64 từ Frontend
    # Frontend gửi dạng "data:image/jpeg;base64,/9j/4AAQ..." -> Phải cắt bỏ phần đầu
    if "," in req.image_base64:
        clean_base64 = req.image_base64.split(",")[1]
    else:
        clean_base64 = req.image_base64

    # 2. Xây dựng câu lệnh Prompt
    final_prompt = build_edit_prompt(req)
    print(f"Lệnh yêu cầu AI: {final_prompt}")

    # 3. Cấu hình gọi Stability AI (Image-to-Image API)
    # Đây là AI chuyên dụng để giữ lại khuôn mặt và thay đổi áo/tóc theo prompt
    engine_id = "stable-diffusion-v1-6"
    api_host = "https://api.stability.ai"
    url = f"{api_host}/v1/generation/{engine_id}/image-to-image"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {req.api_key}" # Dùng key người dùng nhập trên web
    }

    # payload gửi đi
    payload = {
        "init_image": clean_base64,
        "init_image_mode": "IMAGE_STRENGTH",
        "image_strength": 0.5, # Số từ 0-1. 0.5 nghĩa là đổi mới 50% (đủ để thay áo, giữ lại mặt)
        "text_prompts": [
            {
                "text": final_prompt,
                "weight": 1
            }
        ],
        "cfg_scale": 7,
        "samples": 1,
        "steps": 30,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"Lỗi API: {response.text}")
                return JSONResponse(
                    status_code=500, 
                    content={"error": f"Lỗi AI: Vui lòng kiểm tra lại API Key hoặc ảnh. Chi tiết: {response.json().get('message', '')}"}
                )

            # 4. Nhận ảnh trả về từ AI
            data = response.json()
            # Stability AI trả về ảnh dưới dạng Base64
            output_base64 = data["artifacts"][0]["base64"]
            
            # Biến ngược lại thành dạng URL Data để Frontend có thể cho vào thẻ <img src="...">
            output_data_uri = f"data:image/png;base64,{output_base64}"

            return JSONResponse(content={
                "status": "success",
                "prompt_used": final_prompt,
                "image_url": output_data_uri 
            })

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return JSONResponse(status_code=500, content={"error": f"Lỗi máy chủ: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
