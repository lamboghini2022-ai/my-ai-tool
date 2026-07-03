from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

# Lấy từ Environment Variable trên Render (nếu không có thì mới lấy từ web)
api_key = os.environ.get('MY_API_KEY') or payload.get('api_key')

if not api_key:
    return jsonify({"error": {"message": "Thiếu API Key!"}}), 403

# === BỘ TỪ ĐIỂN DỊCH GIAO DIỆN SANG PROMPT (BẰNG TIẾNG ANH) ===
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

def generate_perfect_prompt(data):
    """Hàm tạo câu lệnh Prompt chi tiết từ các lựa chọn gửi lên"""
    gender_vi = data.get('gender', '')
    target_vi = data.get('target', '')
    outfit_vi = data.get('outfit', '')
    custom_outfit = data.get('custom_outfit', '').strip()
    hair_vi = data.get('hair', '')
    bg_vi = data.get('background', '')
    beauty_level = int(data.get('beauty_level', 50))
    brightness_level = int(data.get('brightness_level', 50))
    smooth_skin = data.get('smooth_skin', True)

    gender = PROMPT_MAP["gender"].get(gender_vi, "person")
    target = PROMPT_MAP["target"].get(target_vi, "adult")
    
    prompt_parts = [f"A photorealistic, highly detailed ID card style portrait of a {target} {gender}"]

    if custom_outfit: prompt_parts.append(f"wearing {custom_outfit}")
    elif outfit_vi and PROMPT_MAP["outfit"].get(outfit_vi): prompt_parts.append(PROMPT_MAP["outfit"][outfit_vi])

    if hair_vi and PROMPT_MAP["hair"].get(hair_vi): prompt_parts.append(f"with {PROMPT_MAP['hair'][hair_vi]}")

    main_prompt = ", ".join(prompt_parts) + "."

    if bg_vi and PROMPT_MAP["background"].get(bg_vi): main_prompt += f" Shot against a {PROMPT_MAP['background'][bg_vi]}."

    style_tags = ["8k resolution", "DSLR", "front-facing portrait", "professional studio lighting"]
    if smooth_skin:
        if beauty_level > 70: style_tags.extend(["flawless perfect skin", "airbrushed texture"])
        elif beauty_level > 30: style_tags.extend(["clear skin", "smooth texture"])
    if brightness_level > 70: style_tags.append("bright well-lit exposure")
    elif brightness_level < 30: style_tags.append("moody dramatic lighting")

    return main_prompt + " " + ", ".join(style_tags) + "."

# --- ĐƯỜNG DẪN KIỂM TRA SERVER ---
@app.route('/')
def home():
    return "Server AI Backend đang hoạt động tốt trên Render! Sẵn sàng nhận request từ file index.html."

# --- API NHẬN REQUEST TỪ INDEX.HTML ---
@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        payload = request.json
        api_key = payload.get('api_key')
        
        if not api_key:
            return jsonify({
                "error": {"code": 403, "message": "The caller does not have permission. Vui lòng nhập API Key.", "status": "PERMISSION_DENIED"}
            }), 403

        final_prompt = generate_perfect_prompt(payload)
        
        # NOTE: Khi bạn có API của Nano Banana hay OpenAI thật, bạn gọi code ở đây.
        # Hiện tại trả về ảnh minh họa (Mock Data) để giao diện hoạt động.
        
        return jsonify({
            "status": "success",
            "prompt_used": final_prompt,
            "message": "Gọi API thành công!",
            "image_url": "https://via.placeholder.com/600x400/1e293b/FFFFFF?text=Anh+Moi+Tao+Tu+Server" 
        })

    except Exception as e:
        return jsonify({"error": {"code": 500, "message": str(e), "status": "INTERNAL_SERVER_ERROR"}}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
