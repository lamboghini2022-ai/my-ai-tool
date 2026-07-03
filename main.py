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

import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache

# ==============================================================================
# SỬ DỤNG THƯ VIỆN REPLICATE
# Cài đặt bằng lệnh: pip install replicate
# ==============================================================================
import replicate

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG (CONFIG MANAGER)
# ==============================================================================
class AppConfig:
    """Quản lý cấu hình toàn cục cho ứng dụng"""
    ENV: str = os.getenv("FLASK_ENV", "production")
    DEBUG: bool = ENV == "development"
    
    BASE_DIR: Path = Path(__file__).resolve().parent
    TEMP_DIR: Path = BASE_DIR / "temp_workspace"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    MAX_IMAGE_SIZE_MB: float = 5.0
    MAX_IMAGE_DIMENSION: int = 1024
    
    # Model Replicate mặc định (SDXL - hỗ trợ cả Text2Img và Img2Img)
    # Bạn có thể đổi sang model khác tuỳ ý trên Replicate (ví dụ: Flux, ControlNet)
    DEFAULT_MODEL: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    MAX_RETRIES: int = 3
    
    CACHE_MAX_SIZE: int = 100
    CACHE_TTL_SECONDS: int = 3600

config_manager = AppConfig()

# ==============================================================================
# 2. HỆ THỐNG LOGGING (LOGGER MANAGER)
# ==============================================================================
from logging.handlers import RotatingFileHandler

def setup_logger() -> logging.Logger:
    logger_instance = logging.getLogger("AI_Photo_Editor_Replicate")
    logger_instance.setLevel(logging.DEBUG if config_manager.DEBUG else logging.INFO)
    
    if logger_instance.handlers:
        return logger_instance

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    log_file = config_manager.LOG_DIR / "app.log"
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger_instance.addHandler(file_handler)

    return logger_instance

logger = setup_logger()

# ==============================================================================
# 3. TIỆN ÍCH DỌN DẸP HỆ THỐNG (UTILS)
# ==============================================================================
def create_success_response(data: Dict[str, Any], status_code: int = 200) -> tuple:
    return jsonify(data), status_code

def create_error_response(message: str, error_code: str, status_code: int = 400) -> tuple:
    response = {
        "error": True,
        "code": error_code,
        "message": message,
        "timestamp": int(time.time())
    }
    logger.error(f"API Error Response: {response}")
    return jsonify(response), status_code

class TempWorkspaceManager:
    @staticmethod
    def cleanup_file(file_path: Optional[str]) -> None:
        if not file_path:
            return
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.debug(f"Đã xóa file tạm: {file_path}")
        except Exception as e:
            logger.error(f"Lỗi dọn file tạm {file_path}: {str(e)}")

    @staticmethod
    def cleanup_old_workspaces(max_age_seconds: int = 3600) -> None:
        try:
            current_time = time.time()
            for item in config_manager.TEMP_DIR.iterdir():
                if item.is_file():
                    if (current_time - item.stat().st_mtime) > max_age_seconds:
                        item.unlink()
        except Exception as e:
            logger.error(f"Lỗi dọn rác workspace: {str(e)}")

# ==============================================================================
# 4. XÂY DỰNG PROMPT HÌNH ẢNH (PROMPT BUILDER)
# ==============================================================================
class PromptBuilder:
    TRANSLATION_MAP = {
        "gender": {"Nam": "Male", "Nữ": "Female"},
        "target": {"Người lớn": "Adult", "Thanh niên": "Young adult", "Trẻ em": "Child"},
        "outfit": {
            "Áo Sơ mi": "wearing a crisp button-down shirt",
            "Áo Sơ mi Trắng": "wearing a clean white shirt",
            "Áo Polo": "wearing a smart polo shirt",
            "Công sở": "wearing formal business suit attire",
            "Giữ nguyên": ""
        },
        "hair": {
            "Gọn gàng": "neatly combed hair",
            "Tóc ngắn": "short professional haircut",
            "Tóc dài": "long elegant hair",
            "Giữ nguyên": ""
        },
        "background": {
            "Xanh": "solid blue studio background",
            "Trắng": "pure white studio background"
        }
    }

    # Keyword kích hoạt chất lượng cao cho model ảnh
    BASE_QUALITY = "Masterpiece, ultra-realistic, 8k resolution, highly detailed ID photo portrait, front-facing, professional studio lighting, perfect anatomy."

    @classmethod
    def build(cls, data: Dict[str, Any]) -> str:
        try:
            elements = [cls.BASE_QUALITY]
            
            subject_parts = []
            gender = cls.TRANSLATION_MAP["gender"].get(data.get("gender"), "")
            target = cls.TRANSLATION_MAP["target"].get(data.get("target"), "")
            if target or gender:
                subject_parts.append(f"A {target} {gender}".strip())
            if subject_parts:
                elements.append(" ".join(subject_parts))

            custom_outfit = data.get("custom_outfit")
            if custom_outfit and custom_outfit.strip():
                elements.append(f"wearing {custom_outfit.strip()}")
            else:
                outfit = data.get("outfit")
                if outfit and outfit in cls.TRANSLATION_MAP["outfit"]:
                    elements.append(cls.TRANSLATION_MAP["outfit"][outfit])

            hair = data.get("hair")
            if hair and hair in cls.TRANSLATION_MAP["hair"]:
                elements.append(f"with {cls.TRANSLATION_MAP['hair'][hair]}")

            background = data.get("background")
            if background and background in cls.TRANSLATION_MAP["background"]:
                elements.append(cls.TRANSLATION_MAP["background"][background])

            final_prompt = ", ".join(filter(None, elements))
            logger.info(f"Generated Image Prompt: {final_prompt}")
            return final_prompt
        except Exception as e:
            logger.error(f"Lỗi tạo prompt: {str(e)}")
            return "Professional ID photo, front-facing, studio lighting, high quality"

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
            if len(image_data) > (config_manager.MAX_IMAGE_SIZE_MB * 1024 * 1024):
                raise ValueError(f"Kích thước ảnh vượt quá {config_manager.MAX_IMAGE_SIZE_MB}MB")

            image = Image.open(BytesIO(image_data))
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
                
            image = ImageProcessor._resize_image_if_needed(image)
            file_name = f"{uuid.uuid4().hex}.jpg"
            file_path = config_manager.TEMP_DIR / file_name
            
            image.save(file_path, format="JPEG", quality=95)
            return str(file_path)
        except Exception as e:
            raise ValueError(f"Dữ liệu ảnh không hợp lệ: {str(e)}")

    @staticmethod
    def _resize_image_if_needed(image: Image.Image) -> Image.Image:
        max_dim = config_manager.MAX_IMAGE_DIMENSION
        if image.width > max_dim or image.height > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def detect_face(image_path: str) -> bool:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            img = cv2.imread(image_path)
            if img is None: return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return len(faces) > 0
        except Exception as e:
            return True

# ==============================================================================
# 6. DỊCH VỤ REPLICATE - NƠI GỌI MODEL SINH ẢNH (REPLICATE SERVICE)
# ==============================================================================
result_cache = TTLCache(maxsize=config_manager.CACHE_MAX_SIZE, ttl=config_manager.CACHE_TTL_SECONDS)

class ReplicateService:
    @staticmethod
    def _generate_cache_key(prompt: str, image_path: Optional[str]) -> str:
        key_data = prompt
        if image_path:
            with open(image_path, 'rb') as f:
                key_data += hashlib.md5(f.read()).hexdigest()
        return hashlib.sha256(key_data.encode()).hexdigest()

    @staticmethod
    @retry(
        stop=stop_after_attempt(config_manager.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_image(api_key: str, prompt: str, image_path: Optional[str] = None) -> str:
        cache_key = ReplicateService._generate_cache_key(prompt, image_path)
        if cache_key in result_cache:
            logger.info("Lấy kết quả ảnh từ Cache")
            return result_cache[cache_key]

        file_handle = None
        try:
            # ------------------------------------------------------------------
            # KHỞI TẠO REPLICATE CLIENT (RÕ RÀNG VÀ TƯỜNG MINH)
            # ------------------------------------------------------------------
            replicate_client = replicate.Client(api_token=api_key)
            
            # Cấu hình dữ liệu đầu vào cho model
            input_data = {
                "prompt": prompt,
                "negative_prompt": "ugly, deformed, bad anatomy, bad lighting, watermark, text",
                "prompt_strength": 0.75, # Giữ lại 25% nét ảnh cũ nếu có ảnh gốc
                "num_inference_steps": 30
            }
            
            if image_path:
                logger.info("Đính kèm ảnh gốc vào Replicate để chạy Image-to-Image")
                # Phải mở file dưới dạng binary để đẩy lên Replicate
                file_handle = open(image_path, "rb")
                input_data["image"] = file_handle
                
            logger.info(f"Đang gọi model Replicate: {config_manager.DEFAULT_MODEL}")
            
            # ------------------------------------------------------------------
            # THỰC THI CHẠY MODEL TRÊN REPLICATE
            # ------------------------------------------------------------------
            output = replicate_client.run(
                config_manager.DEFAULT_MODEL,
                input=input_data
            )
            
            # Xử lý kết quả trả về (Replicate thường trả về 1 List chứa URL ảnh)
            if isinstance(output, list) and len(output) > 0:
                image_url = output[0]
            elif isinstance(output, str):
                image_url = output
            else:
                image_url = str(output)

            if not image_url or not image_url.startswith("http"):
                raise ValueError("API Replicate không trả về URL ảnh hợp lệ.")

            result_cache[cache_key] = image_url
            return image_url
            
        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Lỗi từ máy chủ Replicate: {str(e)}")
            raise RuntimeError(f"Lỗi Replicate API: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi hệ thống khi gọi AI: {str(e)}")
            raise RuntimeError(f"Lỗi kết nối hoặc xử lý sinh ảnh: {str(e)}")
        finally:
            # Đóng file an toàn để tránh rò rỉ bộ nhớ
            if file_handle is not None and not file_handle.closed:
                file_handle.close()

# ==============================================================================
# 7. FLASK APP & ROUTING
# ==============================================================================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

class GenerateRequestSchema(BaseModel):
    api_key: str = Field(..., min_length=10, description="Replicate API Token")
    gender: Optional[str] = Field(default="")
    target: Optional[str] = Field(default="")
    outfit: Optional[str] = Field(default="")
    custom_outfit: Optional[str] = Field(default="")
    hair: Optional[str] = Field(default="")
    background: Optional[str] = Field(default="")
    image_base64: Optional[str] = Field(default="")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Replicate API"}), 200

@app.route('/api/generate', methods=['POST'])
def generate_endpoint():
    temp_image_path = None
    try:
        TempWorkspaceManager.cleanup_old_workspaces()

        if not request.is_json:
            return create_error_response("Yêu cầu phải là JSON", "INVALID_CONTENT_TYPE", 415)
            
        try:
            validated_data = GenerateRequestSchema(**request.get_json())
        except ValidationError as ve:
            return create_error_response("Dữ liệu đầu vào sai định dạng", "VALIDATION_ERROR", 400)

        data_dict = validated_data.model_dump()
        api_key = data_dict.pop('api_key')
        prompt = PromptBuilder.build(data_dict)

        if data_dict.get('image_base64'):
            logger.info("Đang xử lý ảnh base64 gửi lên...")
            temp_image_path = ImageProcessor.decode_base64_to_temp(data_dict['image_base64'])
            if not ImageProcessor.detect_face(temp_image_path):
                logger.warning("Cảnh báo: Không tìm thấy mặt người rõ ràng trong ảnh.")
        
        # Gọi thẳng Replicate AI ra sinh ảnh
        generated_result = ReplicateService.generate_image(
            api_key=api_key, 
            prompt=prompt, 
            image_path=temp_image_path
        )

        return create_success_response({
            "image_url": generated_result,
            "prompt_used": prompt,
            "status": "success"
        })

    except ValueError as ve:
        return create_error_response(str(ve), "BAD_REQUEST", 400)
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng: {str(e)}\n{traceback.format_exc()}")
        return create_error_response("Lỗi xử lý server nội bộ.", "INTERNAL_SERVER_ERROR", 500)
    finally:
        if temp_image_path:
            TempWorkspaceManager.cleanup_file(temp_image_path)

# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    logger.info("="*50)
    logger.info("Khởi chạy Backend AI Photo Editor (Sử dụng REPLICATE SDK)")
    logger.info("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=config_manager.DEBUG, threaded=True)
