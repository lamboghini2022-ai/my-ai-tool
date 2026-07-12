import os
import io
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance

app = FastAPI()

# Cho phép giao diện web gửi dữ liệu lên server Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khung chứa dữ liệu mà Render sẽ nhận từ index.html (Không còn cần API Key ở đây)
class YeuCauAnh(BaseModel):
    gioi_tinh: str = "Nữ"
    doi_tuong: str = "Người lớn"
    trang_phuc: str = "Giữ nguyên"
    kieu_toc: str = "Giữ nguyên"
    nen_anh: str = "Trắng"
    thanh_truot_lam_dep: int = 50
    thanh_truot_do_sang: int = 50

# Từ điển dịch dữ liệu sang tiếng Anh
ANH_XA_GIOI_TINH = { "Nữ": "female", "Nam": "male" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { "Xanh": "blue", "Trắng": "white", "Xám": "gray", "Xanh Đậm": "dark navy blue" }

# Hàm lắp ráp câu prompt gửi cho mô hình AI
def tao_cau_lenh_prompt(du_lieu: YeuCauAnh) -> str:
    cum_tu_lam_dep = "natural skin, realistic texture"
    if du_lieu.thanh_truot_lam_dep >= 80:
        cum_tu_lam_dep = "(flawless smooth skin, airbrushed:1.3)"
    elif du_lieu.thanh_truot_lam_dep >= 50:
        cum_tu_lam_dep = "smooth skin, clear complexion"

    gioi_tinh_tieng_anh = ANH_XA_GIOI_TINH.get(du_lieu.gioi_tinh, "person")
    doi_tuong_tieng_anh = ANH_XA_DOI_TUONG.get(du_lieu.doi_tuong, "adult")
    nen_tieng_anh = ANH_XA_NEN.get(du_lieu.nen_anh, "solid")

    trang_phuc = du_lieu.trang_phuc if du_lieu.trang_phuc != "Giữ nguyên" else "current clothes"
    kieu_toc = du_lieu.kieu_toc if du_lieu.kieu_toc != "Giữ nguyên" else "current hairstyle"

    return (f"A professional studio portrait photography of a {doi_tuong_tieng_anh} {gioi_tinh_tieng_anh}, "
            f"front-facing, looking directly at the camera. Wearing {trang_phuc}. "
            f"Hairstyle: {kieu_toc}. Standing in front of a solid {nen_tieng_anh} background. "
            f"{cum_tu_lam_dep}. Masterpiece, 8k resolution, photorealistic, RAW photo.")

# Khu vực Render nhận dữ liệu và gọi API xử lý ảnh
@app.post("/api/xu-ly-anh")
async def xu_ly_anh(du_lieu: YeuCauAnh):
    
    # 1. LẤY API KEY TỪ BIẾN MÔI TRƯỜNG TRÊN RENDER (Tên biến: GEMIN_AI)
    api_key = os.environ.get("GEMIN_AI")
    
    # Kiểm tra xem bạn đã cấu hình biến GEMIN_AI trên Render chưa
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="Lỗi cấu hình máy chủ: Chưa thiết lập biến môi trường GEMIN_AI trên Render."
        )

    prompt_hoan_chinh = tao_cau_lenh_prompt(du_lieu)
    print("Prompt sẽ gửi cho AI:", prompt_hoan_chinh)

    # 2. ĐƯỜNG DẪN TỚI MÔ HÌNH AI CỦA BẠN
    duong_dan_api_ai = "https://dia-chi-api-stable-diffusion-cua-ban.com/sdapi/v1/img2img"
    
    # Gắn API Key (GEMIN_AI) vào Header để gửi đi cho máy chủ AI
    tieu_de_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Gói dữ liệu cấu hình ảnh gửi đi
    du_lieu_gui_ai = {
        "prompt": prompt_hoan_chinh,
        "negative_prompt": "ugly, disfigured, low quality, blurry, deformed face, bad anatomy",
        "steps": 25,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.7 
    }

    try:
        # 3. Gửi lệnh cho AI thực thi
        phan_hoi = requests.post(duong_dan_api_ai, headers=tieu_de_headers, json=du_lieu_gui_ai)
        
        if phan_hoi.status_code != 200:
            raise HTTPException(status_code=phan_hoi.status_code, detail=f"AI báo lỗi: {phan_hoi.text}")

        # 4. Nhận ảnh về và xử lý
        ket_qua_json = phan_hoi.json()
        chuoi_base64_tu_ai = ket_qua_json['images'][0]

        du_lieu_anh_raw = base64.b64decode(chuoi_base64_tu_ai)
        hinh_anh = Image.open(io.BytesIO(du_lieu_anh_raw))

        if du_lieu.thanh_truot_do_sang != 50:
            he_so = du_lieu.thanh_truot_do_sang / 50.0
            hinh_anh = ImageEnhance.Brightness(hinh_anh).enhance(he_so)

        # 5. Đóng gói Base64 trả về cho web
        bo_dem = io.BytesIO()
        hinh_anh.save(bo_dem, format="JPEG")
        chuoi_base64_cuoi_cung = base64.b64encode(bo_dem.getvalue()).decode("utf-8")

        return {
            "thanh_cong": True,
            "hinh_anh": f"data:image/jpeg;base64,{chuoi_base64_cuoi_cung}"
        }

    except Exception as e:
        print("Lỗi tại máy chủ Render:", str(e))
        raise HTTPException(status_code=500, detail=f"Lỗi Render: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
