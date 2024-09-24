import os

# # 이미지 파일이 저장될 경로
IMAGE_DIRECTORY = "../img"  # 이미지가 저장된 디렉토리 경로를 설정

# 이미지 저장 함수
def save_image(file):
    # 경로 생성
    file_location = os.path.join(IMAGE_DIRECTORY, file.filename)
    # 디렉토리가 없다면 생성
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)
    
    # 파일 저장
    with open(file_location, "wb") as file_object:
        file_object.write(file.file.read())
    
    return file_location

# 이미지 경로를 반환하는 함수
def get_image_path(image_id: str):
    return os.path.join(IMAGE_DIRECTORY, image_id)
