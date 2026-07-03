from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app) # Cho phép giao diện (Frontend) gọi API (Backend) mượt mà

# === TỪ ĐIỂN DỊCH TỪ GIAO DIỆN SANG PROMPT CHUẨN (BỘ NÃO) ===
# Bạn có thể tinh chỉnh các từ khóa tiếng Anh ở đây để AI vẽ đẹp nhất
PROMPT_MAP = {
    "gender": {
        "Nữ": "female",
        "Nam": "male"
    },
    "target": {
        "Người lớn": "adult",
        "Thanh niên": "young adult in their 20s",
        "Trẻ em": "child"
    },
    "outfit": {
        "Giữ nguyên": "",
        "Áo Sơ mi": "wearing a casual button-down shirt",
        "Áo Sơ mi Trắng": "wearing a crisp white dress shirt",
        "Áo Polo": "wearing a smart polo shirt",
        "Áo kiểu": "wearing a stylish designer blouse",
        "Áo phông trơn": "wearing a plain simple t-shirt",
        "Áo Vest": "wearing a tailored formal suit jacket",
        "Công sở": "wearing professional business attire",
        "Vest nữ công sở 2": "wearing a chic female business suit",
        "Áo trắng & Khăn quàng": "wearing a white top with an elegant silk scarf",
        "Áo dài trắng": "wearing a traditional Vietnamese white Ao Dai",
        "Nữ Sinh HQ 1": "wearing a Korean high school uniform",
        "Nữ Sinh HQ 2": "wearing a stylish Korean student outfit",
        "Nữ Sinh HQ 3": "wearing a Korean prep school uniform with a tie"
    },
    "hair": {
        "Giữ nguyên": "",
        "Gọn gàng": "neatly styled hair",
        "Tóc ngắn": "short hair",
        "Tóc dài": "long flowing hair",
        "Tóc dài bồng bềnh": "voluminous long wavy hair",
        "Thời trang": "trendy fashionable haircut",
        "Tóc buộc gọn": "hair tied up neatly in a ponytail",
        "Texture Crop NAM": "textured crop haircut for men",
        "Rẽ đôi HQ Nam": "Korean two-block parted hair for men",
        "Xuân Ngắn Nam": "short messy textured hair for men",
        "Hạt Dẻ Ngố Nam": "chestnut brown bowl cut for men"
    },
    "background": {
        "Xanh": "solid light blue background",
        "Trắng": "pure white studio background",
        "Xám": "solid neutral grey background",
        "Xanh Đậm": "solid dark navy blue background"
    }
}

def generate_perfect_prompt(data):
    """
    Hàm này là 'bộ não' phân tích các option được gửi lên và tạo ra Prompt.
    """
    # Lấy các giá trị từ giao diện
    gender_vi = data.get('gender', '')
    target_vi = data.get('target', '')
    outfit_vi = data.get('outfit', '')
    custom_outfit = data.get('custom_outfit', '').strip()
    hair_vi = data.get('hair', '')
    bg_vi = data.get('background', '')
    
    # Sliders
    beauty_level = int(data.get('beauty_level', 50))
    brightness_level = int(data.get('brightness_level', 50))
    smooth_skin = data.get('smooth_skin', True)

    # 1. Cấu trúc đối tượng chính
    gender = PROMPT_MAP["gender"].get(gender_vi, "person")
    target = PROMPT_MAP["target"].get(target_vi, "adult")
    
    # Base prompt (Khung xương nhiếp ảnh)
    prompt_parts = [
        f"A photorealistic, highly detailed ID card style portrait of a {target} {gender}"
    ]

    # 2. Xử lý Trang phục
    if custom_outfit:
        # Ưu tiên prompt người dùng tự gõ
        prompt_parts.append(f"wearing {custom_outfit}")
    elif outfit_vi and PROMPT_MAP["outfit"].get(outfit_vi):
        prompt_parts.append(PROMPT_MAP["outfit"][outfit_vi])

    # 3. Xử lý Tóc
    if hair_vi and PROMPT_MAP["hair"].get(hair_vi):
        prompt_parts.append(f"with {PROMPT_MAP['hair'][hair_vi]}")

    # 4. Gộp câu chính
    main_prompt = ", ".join(prompt_parts) + "."

    # 5. Xử lý Nền (Background)
    if bg_vi and PROMPT_MAP["background"].get(bg_vi):
        main_prompt += f" Shot against a {PROMPT_MAP['background'][bg_vi]}."

    # 6. Xử lý Ánh sáng và Mịn da (Sliders)
    style_tags = ["8k resolution", "DSLR", "front-facing portrait", "professional studio lighting"]
    
    if smooth_skin:
        if beauty_level > 70:
            style_tags.extend(["flawless perfect skin", "airbrushed texture", "symmetrical face"])
        elif beauty_level > 30:
            style_tags.extend(["clear skin", "smooth texture"])
            
    if brightness_level > 70:
        style_tags.append("bright well-lit exposure")
    elif brightness_level < 30:
        style_tags.append("moody dramatic lighting")

    # Hoàn thiện Prompt
    final_prompt = main_prompt + " " + ", ".join(style_tags) + "."
    
    return final_prompt

# --- CÁC ĐƯỜNG DẪN (ROUTES) ---

@app.route('/')
def home():
    # Render file index.html (đảm bảo file index.html nằm chung thư mục hoặc trong folder 'templates')
    return send_file('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        # Lấy dữ liệu và API Key từ Frontend gửi lên
        payload = request.json
        api_key = payload.get('api_key')
        
        if not api_key:
            # Mô phỏng chính xác lỗi 403 như trong ảnh của bạn
            return jsonify({
                "error": {
                    "code": 403, 
                    "message": "The caller does not have permission (Missing API Key)", 
                    "status": "PERMISSION_DENIED"
                }
            }), 403

        # Dùng 'bộ não' tạo Prompt
        final_prompt = generate_perfect_prompt(payload)
        
        # IN RA LOG ĐỂ KIỂM TRA PROMPT TRÊN RENDER
        print(">>> GENERATED PROMPT:", final_prompt)
        print(">>> MODEL SELECTED:", payload.get('model'))

        # =======================================================
        # KHU VỰC GỌI API THỰC TẾ (VÍ DỤ: STABLE DIFFUSION, MIDJOURNEY, OPENAI)
        # Thay URL này bằng endpoint API thực tế (Nano Banana) của bạn
        # =======================================================
        API_URL = "https://api.your-image-generator.com/v1/generate"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_data = {
            "prompt": final_prompt,
            "model": "nano-banana-pro" if "Pro" in payload.get('model', '') else "nano-banana",
            # Các thông số nhiếp ảnh khác nếu API của bạn hỗ trợ
            "negative_prompt": "ugly, deformed, disfigured, poor lighting, bad anatomy, mutated, blurry",
        }

        # Bỏ comment 3 dòng dưới đây khi bạn có API URL thật
        # response = requests.post(API_URL, headers=headers, json=api_data)
        # response.raise_for_status() 
        # return jsonify(response.json())

        # TRẢ VỀ MOCK DATA ĐỂ TEST TRƯỚC:
        return jsonify({
            "status": "success",
            "prompt_used": final_prompt,
            "message": "Thành công! Hãy thay đổi API_URL trong file main.py để nhận ảnh thật.",
            "image_url": "https://via.placeholder.com/600x400/1e293b/FFFFFF?text=Generated+Image+Here" 
        })

    except requests.exceptions.RequestException as e:
        # Bắt lỗi API (403, 500, v.v...) giống hệt giao diện
        error_msg = str(e)
        status_code = e.response.status_code if e.response else 500
        return jsonify({
            "error": {"code": status_code, "message": error_msg, "status": "API_ERROR"}
        }), status_code
    except Exception as e:
        return jsonify({"error": {"code": 500, "message": str(e), "status": "INTERNAL_SERVER_ERROR"}}), 500

if __name__ == '__main__':
    # Chạy ở port 5000 cho dev mode. Render sẽ tự cấp Port qua biến môi trường.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)