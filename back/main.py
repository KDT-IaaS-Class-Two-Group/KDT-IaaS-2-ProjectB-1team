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


@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    print('입력 o')
    img = Image.open(image.file)
    inverted_img = ImageOps.invert(img.convert("RGB"))
    
    # 이미지를 바이트스트림으로 변환
    byte_io = io.BytesIO()
    inverted_img.save(byte_io, format="PNG")
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/png")


# 이미지 관련 라우터를 등록
app.include_router(image.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    




