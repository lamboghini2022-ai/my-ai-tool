import os
import sys
import time
import uuid
import base64
import hashlib
import logging
import traceback
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, Optional

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache

import replicate

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
class AppConfig:
    ENV: str = os.getenv("FLASK_ENV", "production")
    DEBUG: bool = ENV == "development"
    
    BASE_DIR: Path = Path(__file__).resolve().parent
    TEMP_DIR: Path = BASE_DIR / "temp_workspace"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    MAX_IMAGE_SIZE_MB: float = 5.0
    MAX_IMAGE_DIMENSION: int = 1024
    
    # Sử dụng InstructPix2Pix - Model chuyên dùng để CHỈNH SỬA ảnh bằng lệnh
    DEFAULT_MODEL: str = "timbrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f"
    
    MAX_RETRIES: int = 3
    CACHE_MAX_SIZE: int = 100
    CACHE_TTL_SECONDS: int = 3600

config_manager = AppConfig()

# ==============================================================================
# 2. HỆ THỐNG LOGGING
# ==============================================================================
from logging.handlers import RotatingFileHandler

def setup_logger() -> logging.Logger:
    logger_instance = logging.getLogger("AI_Photo_Editor")
    logger_instance.setLevel(logging.DEBUG if config_manager.DEBUG else logging.INFO)
    
    if not logger_instance.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)
    return logger_instance

logger = setup_logger()

# ==============================================================================
# 3. TIỆN ÍCH DỌN DẸP HỆ THỐNG
# ==============================================================================
def create_success_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    return jsonify(data), status_code

def create_error_response(message: str, error_code: str, status_code: int = 400) -> tuple:
    response = {"error": True, "code": error_code, "message": message}
    logger.error(f"Lỗi API: {response}")
    return jsonify(response), status_code

class TempWorkspaceManager:
    @staticmethod
    def cleanup_file(file_path: Optional[str]) -> None:
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Lỗi dọn file tạm: {str(e)}")

# ==============================================================================
# 4. XÂY DỰNG PROMPT CHỈNH SỬA (EDITING INSTRUCTIONS)
# ==============================================================================
class PromptBuilder:
    TRANSLATION_MAP = {
        "outfit": {
            "Áo Sơ mi": "change the outfit to a crisp button-down shirt",
            "Áo Sơ mi Trắng": "change the outfit to a clean white shirt",
            "Áo Polo": "change the outfit to a smart polo shirt",
            "Công sở": "change the outfit to formal business suit attire"
        },
        "background": {
            "Xanh": "change the background to a solid studio blue color",
            "Trắng": "change the background to a pure white color",
            "Xám": "change the background to a neutral grey color"
        }
    }

    @classmethod
    def build_instruction(cls, data: Dict[str, Any]) -> str:
        instructions = []
        
        # Ưu tiên custom prompt nếu user có nhập
        custom_outfit = data.get("custom_outfit")
        if custom_outfit and custom_outfit.strip():
            instructions.append(f"change the outfit to {custom_outfit.strip()}")
        else:
            outfit = data.get("outfit")
            if outfit and outfit in cls.TRANSLATION_MAP["outfit"]:
                instructions.append(cls.TRANSLATION_MAP["outfit"][outfit])

        background = data.get("background")
        if background and background in cls.TRANSLATION_MAP["background"]:
            instructions.append(cls.TRANSLATION_MAP["background"][background])

        if not instructions:
            return "make the photo look more professional"
            
        final_prompt = " and ".join(instructions)
        logger.info(f"Lệnh chỉnh sửa AI (Instruction): {final_prompt}")
        return final_prompt

# ==============================================================================
# 5. XỬ LÝ HÌNH ẢNH (IMAGE PROCESSOR)
# ==============================================================================
class ImageProcessor:
    @staticmethod
    def decode_base64_to_temp(base64_str: str) -> str:
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
                
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
                
            # Đảm bảo ảnh không quá lớn để Replicate xử lý nhanh
            max_dim = config_manager.MAX_IMAGE_DIMENSION
            if image.width > max_dim or image.height > max_dim:
                image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                
            file_name = f"{uuid.uuid4().hex}.jpg"
            file_path = config_manager.TEMP_DIR / file_name
            image.save(file_path, format="JPEG", quality=95)
            
            return str(file_path)
        except Exception as e:
            raise ValueError(f"Dữ liệu ảnh không hợp lệ: {str(e)}")

# ==============================================================================
# 6. DỊCH VỤ REPLICATE (CHỈNH SỬA ẢNH)
# ==============================================================================
result_cache = TTLCache(maxsize=config_manager.CACHE_MAX_SIZE, ttl=config_manager.CACHE_TTL_SECONDS)

class ReplicateService:
    @staticmethod
    @retry(stop=stop_after_attempt(config_manager.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def edit_image(api_key: str, instruction: str, image_path: str) -> str:
        # Cache key dựa trên file ảnh và lệnh chỉnh sửa
        with open(image_path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        cache_key = hashlib.sha256((instruction + img_hash).encode()).hexdigest()
        
        if cache_key in result_cache:
            return result_cache[cache_key]

        file_handle = None
        try:
            replicate_client = replicate.Client(api_token=api_key)
            file_handle = open(image_path, "rb")
            
            input_data = {
                "image": file_handle,
                "prompt": instruction,
                "image_guidance_scale": 1.5, # Mức độ giữ lại ảnh gốc (Cao = giữ nhiều)
                "guidance_scale": 7.5        # Mức độ tuân thủ lệnh prompt (Cao = đổi nhiều)
            }
            
            logger.info("Đang đẩy ảnh lên Replicate để chỉnh sửa...")
            output = replicate_client.run(config_manager.DEFAULT_MODEL, input=input_data)
            
            # Trích xuất URL ảnh trả về
            image_url = output[0] if isinstance(output, list) and output else str(output)
            
            if not image_url.startswith("http"):
                raise ValueError("Replicate không trả về URL hợp lệ.")

            result_cache[cache_key] = image_url
            return image_url
            
        except Exception as e:
            logger.error(f"Lỗi từ Replicate AI: {str(e)}")
            raise RuntimeError(f"Lỗi chỉnh sửa ảnh: {str(e)}")
        finally:
            if file_handle and not file_handle.closed:
                file_handle.close()

# ==============================================================================
# 7. FLASK APP & ROUTING
# ==============================================================================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class GenerateRequestSchema(BaseModel):
    api_key: str = Field(..., min_length=10)
    outfit: Optional[str] = Field(default="")
    custom_outfit: Optional[str] = Field(default="")
    background: Optional[str] = Field(default="")
    image_base64: str = Field(..., description="Bắt buộc phải có ảnh gốc để chỉnh sửa")
    # Tóc và Giới tính tạm bỏ qua trong instruction để tập trung vào quần áo & nền

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ready for editing"}), 200

@app.route('/api/generate', methods=['POST'])
def generate_endpoint():
    temp_image_path = None
    try:
        if not request.is_json:
            return create_error_response("Yêu cầu phải là JSON", "INVALID_CONTENT_TYPE", 415)
            
        try:
            validated_data = GenerateRequestSchema(**request.get_json())
        except ValidationError:
            return create_error_response("Vui lòng tải ảnh gốc lên trước khi chỉnh sửa", "MISSING_IMAGE", 400)

        data_dict = validated_data.model_dump()
        api_key = data_dict.pop('api_key')
        
        # Tạo lệnh (Ví dụ: "change the background to solid blue")
        instruction = PromptBuilder.build_instruction(data_dict)

        # Xử lý ảnh gốc
        temp_image_path = ImageProcessor.decode_base64_to_temp(data_dict['image_base64'])
        
        # Gọi AI chỉnh sửa
        edited_image_url = ReplicateService.edit_image(
            api_key=api_key, 
            instruction=instruction, 
            image_path=temp_image_path
        )

        return create_success_response({
            "image_url": edited_image_url,
            "prompt_used": instruction,
            "status": "success"
        })

    except ValueError as ve:
        return create_error_response(str(ve), "BAD_REQUEST", 400)
    except Exception as e:
        logger.error(f"Lỗi hệ thống: {str(e)}\n{traceback.format_exc()}")
        return create_error_response("Lỗi xử lý server nội bộ.", "INTERNAL_SERVER_ERROR", 500)
    finally:
        TempWorkspaceManager.cleanup_file(temp_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=config_manager.DEBUG)
