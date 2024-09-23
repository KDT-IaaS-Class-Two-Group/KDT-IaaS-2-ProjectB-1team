import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from main import app
from utils.file_operations import save_image, get_image_path, IMAGE_DIRECTORY
import os
import io

import mimetypes
mimetypes.init()

client = TestClient(app)

@pytest.fixture
def test_image():
    return io.BytesIO(b"fake image content")

@pytest.fixture
def mock_image_path(monkeypatch):
    def mock_get_image_path(image_id):
        return f"img/{image_id}"
    monkeypatch.setattr("utils.file_operations.get_image_path", mock_get_image_path)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello word"}

def test_upload_image(test_image, monkeypatch):
    def mock_save_image(file):
        return os.path.join("img", "test.jpg")  # 이 경로가 실제 저장되는 경로와 일치해야 합니다.
    
    monkeypatch.setattr("utils.file_operations.save_image", mock_save_image)

    response = client.post("/upload/", files={"file": ("test.jpg", test_image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"filename": "test.jpg", "file location": "../img/test.jpg"}


def test_download_image_success(mock_image_path, monkeypatch):
    def mock_guess_type(filename, strict=True):
        return "image/jpeg", None  # MIME 타입을 강제로 반환
    
    monkeypatch.setattr("mimetypes.guess_type", mock_guess_type)

    def mock_isfile(path):
        return True  # 항상 파일이 있다고 가정

    monkeypatch.setattr("os.path.isfile", mock_isfile)

    # 여기에 실제로 반환될 파일 경로를 확인하거나 조정
    response = client.get("/download/test.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"  # 예상 MIME 타입 확인

def test_download_image_not_found(mock_image_path, monkeypatch):
    # os.stat 호출을 모의하여 예외를 발생시키지 않도록 설정
    def mock_stat(path):
        raise FileNotFoundError

    monkeypatch.setattr("os.stat", mock_stat)

    response = client.get("/download/nonexistent.jpg")
    assert response.status_code == 404  # 404 Not Found 응답 확인


def test_save_image(test_image, tmp_path, monkeypatch):
    monkeypatch.setattr("utils.file_operations.IMAGE_DIRECTORY", str(tmp_path))

    file = UploadFile(filename="test.jpg", file=test_image)
    file_location = save_image(file)

    assert os.path.exists(file_location)  # 파일이 저장되었는지 확인
    with open(file_location, "rb") as f:
        content = f.read()
    assert content == b"fake image content"  # 파일 내용 확인

def test_get_image_path():
    image_id = "test.jpg"
    expected_path = os.path.join(IMAGE_DIRECTORY, image_id)
    assert get_image_path(image_id) == expected_path  # 예상 경로 확인
