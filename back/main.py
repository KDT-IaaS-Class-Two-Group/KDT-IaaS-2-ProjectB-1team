from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import image

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)