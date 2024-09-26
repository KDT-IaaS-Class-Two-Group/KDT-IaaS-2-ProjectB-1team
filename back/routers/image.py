from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from utils.file_operations import save_image
import io
from fastapi.responses import StreamingResponse

router = APIRouter()

# 이미지 업로드 엔드포인트
@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):  #UploadFile은 FastAPI에서 제공하는 클래스 #= File(...): File()은 이 변수가 파일을 담을 것임을 지정
    file_location = save_image(file)  # 이미지를 저장하는 함수 호출
    return {"img": file, "text": file_location}

from service.image_preprocessing import process_image,convert_cv_to_pil
@router.post("/process-image")
async def process_img(image: UploadFile = File(...)):
    try:
        print("이미지 입력")
        img_data = await image.read()
        process_img = process_image(img_data, 300)
        if len(process_img) > 0:
            # 첫 번째 얼굴 ROI를 사용하여 결과를 반환
            img_pil = convert_cv_to_pil((process_img[0] * 255.0).astype(np.uint8))
        else:
            # 얼굴이 발견되지 않은 경우 에러 처리
            raise HTTPException(status_code=404, detail="No faces detected.")
        
        # 이미지를 바이트스트림으로 변환
        byte_io = io.BytesIO()
        img_pil.save(byte_io, format="PNG")
        byte_io.seek(0)

        return StreamingResponse(byte_io, media_type="image/png")
        
    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return {"error": str(e)}, 400  # HTTP 400 Bad Request