from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from PIL import Image, ImageEnhance
import io
import base64

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS để cho phép tệp giao diện HTML gửi dữ liệu lên server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa cấu trúc dữ liệu nhận từ tệp index.html gửi lên
class YeuCauAnh(BaseModel):
    gioi_tinh: str = ""
    doi_tuong: str = ""
    trang_phuc: str = ""
    kieu_toc: str = ""
    nen_anh: str = ""
    thanh_truot_lam_dep: int = 50
    thanh_truot_do_sang: int = 50

# Từ điển ánh xạ để dịch dữ liệu từ giao diện sang tiếng Anh cho AI hiểu
ANH_XA_GIOI_TINH = { "Nữ": "female", "Nam": "male" }
ANH_XA_DOI_TUONG = { "Người lớn": "adult", "Thanh niên": "young adult", "Trẻ em": "child" }
ANH_XA_NEN = { "Xanh": "blue", "Trắng": "white", "Xám": "gray", "Xanh Đậm": "dark navy blue" }

# Hàm tự động lắp ghép dữ liệu thành câu lệnh Prompt hoàn chỉnh
def tao_cau_lenh_prompt(du_lieu: YeuCauAnh) -> str:
    # Xử lý mức độ làm mịn và làm đẹp da dựa trên thanh trượt
    cum_tu_lam_dep = "natural skin, visible pores, realistic texture"
    if du_lieu.thanh_truot_lam_dep >= 80:
        cum_tu_lam_dep = "(flawless smooth skin, blemish-free, airbrushed skin:1.3)"
    elif du_lieu.thanh_truot_lam_dep >= 50:
        cum_tu_lam_dep = "smooth skin, clear complexion"

    # Dịch các giá trị lựa chọn sang tiếng Anh
    gioi_tinh_tieng_anh = ANH_XA_GIOI_TINH.get(du_lieu.gioi_tinh, "person")
    doi_tuong_tieng_anh = ANH_XA_DOI_TUONG.get(du_lieu.doi_tuong, "adult")
    nen_tieng_anh = ANH_XA_NEN.get(du_lieu.nen_anh, "solid")

    # Trả về chuỗi câu lệnh prompt bằng tiếng Anh để gửi tới mô hình AI tạo ảnh
    return (f"A professional studio portrait photography of a {doi_tuong_tieng_anh} {gioi_tinh_tieng_anh}, "
            f"front-facing, looking directly at the camera. The subject is wearing {du_lieu.trang_phuc}. "
            f"The subject has {du_lieu.kieu_toc}. Standing in front of a solid {nen_tieng_anh} background. "
            f"{cum_tu_lam_dep}. Masterpiece, 8k resolution, photorealistic, hyper-detailed skin.")

# API chính để xử lý hình ảnh
@app.post("/api/xu-ly-anh")
async def xu_ly_anh(du_lieu: YeuCauAnh, authorization: str = Header(None)):
    # Kiểm tra mã API Key được truyền từ tiêu đề Header của giao diện
    if not authorization:
        raise HTTPException(status_code=403, detail="Vui lòng nhập API Key trên giao diện!")

    prompt_hoan_chinh = tao_cau_lenh_prompt(du_lieu)
    print("Đã tạo câu lệnh Prompt thành công:", prompt_hoan_chinh)

    # Đường dẫn kết nối tới mô hình AI (Ví dụ này dùng mô hình Inpainting trên Hugging Face)
    duong_dan_api = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
    tieu_de_headers = {"Authorization": authorization}
    du_lieu_gui_ai = {
        "inputs": prompt_hoan_chinh,
        # Lưu ý kỹ thuật: Khi triển khai thực tế, bạn cần bổ sung thêm tham số dữ liệu ảnh gốc
        # và ảnh mặt nạ (mask) vào cấu trúc này để AI thực hiện thay đổi trang phục/tóc chuẩn xác.
    }

    try:
        # Gửi yêu cầu xử lý sang máy chủ AI
        phan_hoi = requests.post(duong_dan_api, headers=tieu_de_headers, json=du_lieu_gui_ai)
        
        if phan_hoi.status_code != 200:
            thong_bao_loi = phan_hoi.json().get("error", "Lỗi không xác định từ hệ thống AI")
            raise HTTPException(status_code=phan_hoi.status_code, detail=thong_bao_loi)

        # Lấy dữ liệu danh sách byte của tệp ảnh trả về
        du_lieu_anh_raw = phan_hoi.content

        # Sử dụng thư viện Pillow để mở và chỉnh sửa hình ảnh trực tiếp
        hinh_anh = Image.open(io.BytesIO(du_lieu_anh_raw))

        # Điều chỉnh độ sáng của bức ảnh (Mốc số 50 được coi là giữ nguyên gốc)
        if du_lieu.thanh_truot_do_sang != 50:
            he_so_anh_sang = du_lieu.thanh_truot_do_sang / 50.0
            bo_dieu_chinh_sang = ImageEnhance.Brightness(hinh_anh)
            hinh_anh = bo_dieu_chinh_sang.enhance(he_so_anh_sang)

        # Chuyển đổi toàn bộ dữ liệu ảnh sau xử lý thành chuỗi mã hóa Base64
        bo_dem_du_lieu = io.BytesIO()
        hinh_anh.save(bo_dem_du_lieu, format="JPEG")
        chuoi_ma_hoa_base64 = base64.b64encode(bo_dem_du_lieu.getvalue()).decode("utf-8")

        # Trả kết quả thành công về cho trình duyệt hiển thị
        return {
            "thanh_cong": True,
            "hinh_anh": f"data:image/jpeg;base64,{chuoi_ma_hoa_base64}"
        }

    except Exception as e:
        print("Lỗi hệ thống backend:", str(e))
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra trong quá trình xử lý hình ảnh từ hệ thống.")

# Khởi chạy ứng dụng máy chủ
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
