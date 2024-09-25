import cv2 as cv
# img = cv.imread("./path/to/image.jpg")

# # 이미지가 제대로 읽혔는지 확인
# if img is None:
#     print(f"Error: Could not open or find the image at {img}")
    
# cv.imshow("Display window", img)
# k = cv.waitKey(0) # Wait for a keystroke in the windowx


from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import numpy as np

def image_from_transfrom(img):
    # PIL 이미지를 NumPy 배열로 변환
    img_array = np.array(img)

    # OpenCV 형식에 맞게 색상 채널 순서 변경 (RGB -> BGR)
    img_bgr = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
    
    return img_bgr

#이미지 크기 조정
def image_resized(img, target_width):
    
    # 원본 이미지의 크기 가져오기
    original_height, original_width = img.shape[:2]

    # 새로운 높이 계산 (비율 유지)
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)

    # 이미지 크기 조정
    resized_img = cv.resize(img, (target_width, target_height))

    return resized_img

#그레이스케일 변환
def grayscale_conversion(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray_img

#평활화
def smoothing(img):
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)  # 5x5 커널 사용
    return blurred_img
  
#얼굴 검출 
def face_detection(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
  

# ROI 추출 및 정규화
def roi_and(img, faces):
    rois = []
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]  # ROI 추출
        normalized_img = roi.astype('float32') / 255.0  # 정규화
        rois.append(normalized_img)
    return rois


def image_to_transform(img):
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))