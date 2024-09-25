import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import shutil

# Mediapipe 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# 이미지 처리 함수
def process_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 랜드마크 탐지
    results = face_mesh.process(rgb_image)

    # 랜드마크 연결부위 가져오기
    LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
    RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
    NOSE = mp_face_mesh.FACEMESH_NOSE
    LIPS = mp_face_mesh.FACEMESH_LIPS
    FACE_OVAL = mp_face_mesh.FACEMESH_FACE_OVAL

    # 랜드마크 좌표 저장 리스트
    landmarks_data = {}

    # 랜드마크 좌표 추출 및 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            landmarks = np.array([(landmark.x * w, landmark.y * h) for landmark in face_landmarks.landmark])

            # 벡터 값 출력 함수
            def draw_and_print_vector(start_idx, end_idx, landmarks, part_name):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
                landmarks_data[f"{part_name}_{start_idx}_{end_idx}"] = vector
                cv2.line(image, tuple(map(int, start_point)), tuple(map(int, end_point)), (0, 255, 0), 1)

            # 각 부위별로 연결선 및 벡터 출력
            for connection in LEFT_EYE:
                draw_and_print_vector(connection[0], connection[1], landmarks, 'LEFT_EYE')
            for connection in RIGHT_EYE:
                draw_and_print_vector(connection[0], connection[1], landmarks, 'RIGHT_EYE')
            for connection in NOSE:
                draw_and_print_vector(connection[0], connection[1], landmarks, 'NOSE')
            for connection in LIPS:
                draw_and_print_vector(connection[0], connection[1], landmarks, 'LIPS')
            for connection in FACE_OVAL:
                draw_and_print_vector(connection[0], connection[1], landmarks, 'FACE_OVAL')

        # 성공적으로 랜드마크를 탐지한 경우
        status = "success"
    else:
        # 랜드마크 탐지 실패한 경우
        status = "fail"
        
    # 원하는 저장 폴더 경로 설정
    if status == "success":
        json_folder = os.path.join(os.getcwd(), 'successFolder/successJson')
    else:
        json_folder = os.path.join(os.getcwd(), 'failFolder/failJson')

    os.makedirs(json_folder, exist_ok=True)  # 폴더가 없으면 생성

    # JSON 파일 이름 설정
    json_file_name = f"{status}.json"  # 성공 또는 실패에 따라 파일 이름 변경
    json_file_path = os.path.join(json_folder, json_file_name)

    # 기존 JSON 파일이 있으면 불러오기
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    # 이미지 파일 이름을 키로 사용하여 새로운 데이터 추가
    if status == "success":
        existing_data[f"success_{os.path.basename(image_path)}"] = landmarks_data
    else:
        existing_data[f"fail_{os.path.basename(image_path)}"] = landmarks_data

    # 결과를 JSON 파일로 저장
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False)

    # 원하는 폴더 경로 설정
    if status == "success":
        desired_folder = os.path.join(os.getcwd(), 'successFolder/successImg')
    else:
        desired_folder = os.path.join(os.getcwd(), 'failFolder/failImg')

    os.makedirs(desired_folder, exist_ok=True)  # 폴더가 없으면 생성

    # 이미지 저장 경로 설정
    output_image_path = os.path.join(desired_folder, f"{status}_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)  # 처리된 이미지를 저장하는 코드 추가
    
    # 결과 이미지 출력
    cv2.imshow('Face Features with Vectors', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Processed data saved to {json_file_path}")

# 파일 열기 함수
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

# GUI 설정
root = tk.Tk()
root.title("Face Landmark Processor")
root.geometry("300x150")

open_button = tk.Button(root, text="Select Image", command=open_file)
open_button.pack(expand=True)

root.mainloop()
