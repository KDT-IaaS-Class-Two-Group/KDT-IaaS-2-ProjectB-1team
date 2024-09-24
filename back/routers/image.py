from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from utils.file_operations import save_image, get_image_path
import os #표준 라이브러리 중 하나로, 운영체제 관련 기능

router = APIRouter()

# 이미지 업로드 엔드포인트
@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):  #UploadFile은 FastAPI에서 제공하는 클래스 #= File(...): File()은 이 변수가 파일을 담을 것임을 지정
    file_location = save_image(file)  # 이미지를 저장하는 함수 호출
    return {"filename": file.filename, "file location": file_location}

# 이미지 다운로드 엔드포인트
@router.get("/download/{image_id}")
async def download_image(image_id: str):
    file_path = get_image_path(image_id)  # 이미지 경로를 가져오는 함수 호출
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
