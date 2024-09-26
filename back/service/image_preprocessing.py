import cv2 as cv
from PIL import Image
import numpy as np
import io

# PIL 이미지를 OpenCV에서 사용 가능한 BGR 형식으로 변환하는 함수
def convert_pil_to_cv(img):
    # PIL 이미지를 NumPy 배열로 변환
    img_array = np.array(img)
    # OpenCV 형식에 맞게 색상 채널 순서 변경하여 반환 (RGB -> BGR)
    img_bgr = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    return img_bgr

# 이미지 크기 조정 함수 (비율 유지)
def resize_image(img, target_width):
    # 원본 이미지의 크기 가져오기
    original_height, original_width = img.shape[:2]
    # 종횡비 계산
    aspect_ratio = original_width / original_height
    # 종횡비를 유지하며 새로운 높이 계산
    target_height = int(target_width / aspect_ratio)
    # 이미지 크기 조정하여 반환
    return cv.resize(img, (target_width, target_height))

# 그레이스케일 변환
def convert_to_grayscale(img):
    # BGR -> 그레이스케일 변환
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 이미지 블러링 (평활화)
def apply_smoothing(img):
    # 5x5 가우시안 블러 커널 사용
    return cv.GaussianBlur(img, (5, 5), 0)

#얼굴 검출 
def detect_faces(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 얼굴 검출 수행
    return face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)


# 얼굴 영역을 추출하고 정규화
def extract_and_normalize_faces(img, faces):
    rois = []
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출 (Region of Interest)
        roi = img[y:y+h, x:x+w]
        # 0~1 사이 값으로 정규화
        normalized_img = roi.astype('float32') / 255.0
        rois.append(normalized_img)
    return rois

# OpenCV 이미지를 다시 PIL 이미지로 변환하는 함수
def convert_cv_to_pil(img):
    # BGR -> RGB 변환 후 PIL 이미지 생성
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# 이미지 전처리
def process_image(image_pil, target_width):
    img = Image.open(io.BytesIO(image_pil))
    # PIL 이미지를 OpenCV BGR 형식으로 변환
    img_bgr = np.array(img)
    img_bgr = cv.cvtColor(img_bgr, cv.COLOR_RGB2BGR)
    # 이미지 크기 조정 (비율 유지)
    resized_img = resize_image(img_bgr, target_width)
    # 그레이스케일로 변환
    gray_img = convert_to_grayscale(resized_img)
    # 블러처리
    face_img = apply_smoothing(gray_img)
    # 얼굴 검출
    faces = detect_faces(face_img)
    # 얼굴 영역 추출 및 정규화
    face_rois = extract_and_normalize_faces(resized_img, faces)
        
    return face_rois