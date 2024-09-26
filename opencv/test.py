import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from tkinter import ttk

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
    json_folder = os.path.join(os.getcwd(), 'json')
    os.makedirs(json_folder, exist_ok=True)  # 폴더가 없으면 생성

    # JSON 파일 이름 설정
    json_file_path = os.path.join(json_folder, "integration.json")

    # 기존 JSON 파일이 있으면 불러오기
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}
    
    counter = 1  # 중복 파일을 위한 카운터 초기화
    
    # 이미지 파일 이름을 키로 사용하여 새로운 데이터 추가
    key = os.path.basename(image_path)
    if status == "success":
        while key in existing_data:
            counter += 1
            key = f"{counter}{os.path.basename(image_path)}"
        existing_data[key] = landmarks_data
    else:
        existing_data[key] = {}

    # 결과를 JSON 파일로 저장
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

    data = load_json_data()
    update_treeview(data)

    # 원하는 폴더 경로 설정``
    if status == "success":
        desired_folder = os.path.join(os.getcwd(), 'successFolder/successImg')
    else:
        desired_folder = os.path.join(os.getcwd(), 'failFolder/failImg')
    os.makedirs(desired_folder, exist_ok=True)  # 폴더가 없으면 생성


    # 이미지 저장 경로 설정
    output_image_path = os.path.join(desired_folder, os.path.basename(image_path))
    while os.path.exists(output_image_path):
        output_image_path = os.path.join(desired_folder, f"{counter}{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)  # 처리된 이미지를 저장하는 코드 추가
    
    cv2.destroyAllWindows() # 모든 윈도우 닫기
    messagebox.showinfo("Success", f"Processed data saved to {json_file_path}")

# 파일 열기 함수
def open_file():
    file_path = filedialog.askopenfilename()
    
    # 허용되는 파일 확장자 목록
    valid_extensions = (".jpg", ".jpeg", ".png")
    # 파일 확장자 확인
    if not file_path.lower().endswith(valid_extensions):
        messagebox.showerror("Error", "지원하지 않는 파일 형식입니다.\nJPG, JPEG, PNG 파일만 허용됩니다.")
        return
    
    if file_path:
        process_image(file_path)
        
# integration.json 파일 불러오기
def load_json_data():
    try:
        with open('json/integration.json', 'r') as file:
            # 파일이 비어 있으면 빈 딕셔너리를 반환
            if os.stat('json/integration.json').st_size == 0:
                print("Warning: JSON file is empty.")
                return {}
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for {'json/integration.json'}. File may be corrupted.")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}

# 객체가 비어있는지 확인하는 함수
def check_success(obj):
    return "성공" if obj else "실패"

# Treeview 업데이트
def update_treeview(data):
    # 기존 항목을 명확히 모두 제거
    tree.delete(*tree.get_children())  # 모든 항목을 삭제
    
    for image_name, details in data.items():
        success_status = check_success(details)
        tree.insert("", "end", values=(image_name, success_status))

# 파일 변경 감지 및 업데이트 함수
def check_for_updates():
    global last_modified_time
    
    if not os.path.exists('json/integration.json'):
        print(f"Error: {'json/integration.json'} does not exist.")
        return
    
    current_modified_time = os.path.getmtime('json/integration.json')  # 파일의 마지막 수정 시간 확인
    
    if current_modified_time != last_modified_time:
        last_modified_time = current_modified_time
        data = load_json_data()
        if data is not None:
            update_treeview(data)
    
    # 1000ms(1초) 후에 다시 check_for_updates 함수 실행
    root.after(1000, check_for_updates)

# GUI 설정
root = tk.Tk()
root.title("Face Landmark Processor")
root.geometry("800x500")

open_button = tk.Button(root, text="파일 첨부", command=open_file)
open_button.pack(expand=True)

# 타이틀 스타일 설정
style = ttk.Style(root)
style.configure("Treeview.Heading", font=("Arial", 17, "bold"))  # 열 제목 폰트 크기 설정
style.configure("Treeview", rowheight=25)  # 행 높이 설정

# Treeview 생성
tree = ttk.Treeview(root, columns=("Image", "Status"), show="headings", height=15)
tree.heading("Image", text="이미지 이름")
tree.heading("Status", text="성공 여부")
tree.column("Image", width=100, anchor="center")
tree.column("Status", width=50, anchor="center")
tree.pack(fill=tk.BOTH, expand=True)

# 데이터 로드 및 리스트 업데이트
data = load_json_data()
update_treeview(data)

# 프로그램이 시작할 때 JSON 파일의 존재 여부를 확인하고 초기화
if os.path.exists('json/integration.json'):
    last_modified_time = os.path.getmtime('json/integration.json')
else:
    last_modified_time = None  # 파일이 없는 경우 초기화

# 주기적으로 파일 변경 감지
root.after(1000, check_for_updates)  # 1000ms(1초)마다 파일 업데이트 확인

root.mainloop()
