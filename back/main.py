import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from routers import image
from PIL import Image, ImageOps
import io


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#기본경로
@app.get("/")
def read_root():
    return {"message": "hello word"}

# 이미지 관련 라우터를 등록
app.include_router(image.router)

def create_requirements():
    """requirements.txt 파일 자동 생성"""
    print("Generating requirements.txt...")
    # 현재 디렉토리에 requirements.txt 파일 생성
    with open('requirements.txt', 'w') as f:
        subprocess.run(["pip3.8", "freeze"], stdout=f)

if __name__ == "__main__":
    create_requirements()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
