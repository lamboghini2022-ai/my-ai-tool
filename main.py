from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image, ImageEnhance
import io
import base64

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS để cho phép file giao diện HTML (Frontend) gọi tới máy chủ này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trên Render, cho phép mọi nguồn truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu trúc dữ liệu nhận từ giao diện web gửi lên
class YeuCauAnh(BaseModel):
    gioi_tinh: str = "Nữ"
    doi_tuong: str = "Người lớn"
    trang_phuc: str = "Giữ nguyên"
    kieu_toc: str = "Giữ nguyên"
    nen_anh: str = "Trắng"
    thanh_truot_lam_dep: int = 50
    thanh_truot_do_sang: int = 50

# Từ điển dịch tiếng Việt sang tiếng Anh cho AI hiểu
ANH_XA_GIOI_TINH = { "Nữ": "female", "Nam": "male" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { "Xanh": "blue", "Trắng": "white", "Xám": "gray", "Xanh Đậm": "dark navy blue" }

# Hàm tạo câu lệnh Prompt tiếng Anh
def tao_cau_lenh_prompt(du_lieu: YeuCauAnh) -> str:
    # Xử lý mức độ làm đẹp da
    cum_tu_lam_dep = "natural skin, visible pores, realistic texture"
    if du_lieu.thanh_truot_lam_dep >= 80:
        cum_tu_lam_dep = "(flawless smooth skin, blemish-free, airbrushed skin:1.3)"
    elif du_lieu.thanh_truot_lam_dep >= 50:
        cum_tu_lam_dep = "smooth skin, clear complexion"

    # Dịch dữ liệu
    gioi_tinh_tieng_anh = ANH_XA_GIOI_TINH.get(du_lieu.gioi_tinh, "person")
    doi_tuong_tieng_anh = ANH_XA_DOI_TUONG.get(du_lieu.doi_tuong, "adult")
    nen_tieng_anh = ANH_XA_NEN.get(du_lieu.nen_anh, "solid")

    # Dịch trang phục và kiểu tóc (nếu người dùng không nhập gì thì để trống)
    trang_phuc_tieng_anh = du_lieu.trang_phuc if du_lieu.trang_phuc != "Giữ nguyên" else "current clothes"
    kieu_toc_tieng_anh = du_lieu.kieu_toc if du_lieu.kieu_toc != "Giữ nguyên" else "current hairstyle"

    return (f"A professional studio portrait photography of a {doi_tuong_tieng_anh} {gioi_tinh_tieng_anh}, "
            f"front-facing, looking directly at the camera. The subject is wearing {trang_phuc_tieng_anh}. "
            f"The subject has {kieu_toc_tieng_anh}. Standing in front of a solid {nen_tieng_anh} background. "
            f"{cum_tu_lam_dep}. Masterpiece, 8k resolution, photorealistic, hyper-detailed skin.")

# Hàm API Xử lý ảnh
@app.post("/api/xu-ly-anh")
async def xu_ly_anh(du_lieu: YeuCauAnh, authorization: str = Header(None)):
    # 1. Kiểm tra API Key
    if not authorization:
        raise HTTPException(status_code=403, detail="Vui lòng nhập API Key (HuggingFace Token) trên giao diện!")
    
    # Chuẩn hóa chuẩn Bearer Token cho HuggingFace
    if not authorization.startswith("Bearer "):
        authorization = f"Bearer {authorization}"

    # 2. Tạo câu lệnh
    prompt_hoan_chinh = tao_cau_lenh_prompt(du_lieu)
    print("Đang xử lý với Prompt:", prompt_hoan_chinh)

    # 3. Gửi tới Hugging Face (Mô hình Text-to-Image / Inpainting)
    duong_dan_api = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    tieu_de_headers = {"Authorization": authorization}
    
    # Payload gửi lên AI
    du_lieu_gui_ai = {
        "inputs": prompt_hoan_chinh
    }

    try:
        phan_hoi = requests.post(duong_dan_api, headers=tieu_de_headers, json=du_lieu_gui_ai)
        
        # Bắt lỗi nếu API Key sai hoặc bị giới hạn
        if phan_hoi.status_code != 200:
            loi_chi_tiet = phan_hoi.json().get("error", "Lỗi không xác định")
            print("Lỗi từ HuggingFace:", loi_chi_tiet)
            raise HTTPException(status_code=phan_hoi.status_code, detail=loi_chi_tiet)

        # Lấy file ảnh dưới dạng byte
        du_lieu_anh_raw = phan_hoi.content
        hinh_anh = Image.open(io.BytesIO(du_lieu_anh_raw))

        # 4. Xử lý độ sáng
        if du_lieu.thanh_truot_do_sang != 50:
            he_so_anh_sang = du_lieu.thanh_truot_do_sang / 50.0
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so_anh_sang)

        # 5. Chuyển ảnh sang Base64 để gửi về HTML
        bo_dem_du_lieu = io.BytesIO()
        hinh_anh.save(bo_dem_du_lieu, format="JPEG")
        chuoi_ma_hoa_base64 = base64.b64encode(bo_dem_du_lieu.getvalue()).decode("utf-8")

        return {
            "thanh_cong": True,
            "hinh_anh": f"data:image/jpeg;base64,{chuoi_ma_hoa_base64}",
            "thong_bao": "Đã xử lý ảnh thành công!"
        }

    except Exception as e:
        print("Lỗi hệ thống:", str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

# Khởi chạy server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
