import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from tkinter import ttk
import math

# Mediapipe 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

current_image_key = None  # 현재 이미지 키를 전역 변수로 설정

# 벡터의 크기, 각도, 정규화된 각도 등을 계산하는 함수
def calculate_vector_details(start_point, end_point):
    vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    angle_degrees = math.degrees(math.atan2(vector[1], vector[0]))
    angle_normalized = (angle_degrees + 360) % 360
    return {
        'vector': vector,
        'magnitude': magnitude,
        'angle_degrees': angle_degrees,
        'angle_normalized': angle_normalized
    }

# connections 리스트에서 각 랜드마크를 연결하고 벡터 정보를 계산
def calculate_vectors(connections, landmarks, width, height):
    vectors = []
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        vector_details = calculate_vector_details(start_point, end_point)

        if vector_details['magnitude'] != 0:
            direction = (vector_details['vector'][0] / vector_details['magnitude'],
                         vector_details['vector'][1] / vector_details['magnitude'])
        else:
            direction = (0, 0)

        vectors.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'vector': vector_details['vector'],
            'magnitude': vector_details['magnitude'],
            'angle_degrees': vector_details['angle_degrees'],
            'angle_normalized': vector_details['angle_normalized'],
            'direction': direction
        })

    return vectors

# 이미지 처리 함수
def process_image(image_path):
    global current_image_key
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)

    landmarks_data = {}
    status = "fail"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            landmarks = np.array([(landmark.x * w, landmark.y * h) for landmark in face_landmarks.landmark])

            for part_name, connections in [
                ('LEFT_EYE', mp_face_mesh.FACEMESH_LEFT_EYE),
                ('RIGHT_EYE', mp_face_mesh.FACEMESH_RIGHT_EYE),
                ('NOSE', mp_face_mesh.FACEMESH_NOSE),
                ('LIPS', mp_face_mesh.FACEMESH_LIPS),
                ('FACE_OVAL', mp_face_mesh.FACEMESH_FACE_OVAL)]:
                
                vectors = calculate_vectors(connections, landmarks, w, h)
                landmarks_data[part_name] = {'values': [
                    {'vector': vec['vector'],
                     'magnitude': vec['magnitude'],
                     'angle_degrees': vec['angle_degrees'],
                     'angle_normalized': vec['angle_normalized'],
                     'direction': vec['direction']} for vec in vectors]}
                
                for vec in vectors:
                    start_pt = tuple(map(int, landmarks[vec['start_idx']]))
                    end_pt = tuple(map(int, landmarks[vec['end_idx']]))
                    cv2.line(image, start_pt, end_pt, (0, 255, 0), 1)
                    cv2.circle(image, start_pt, 1, (255, 0, 0), -1)
                    cv2.circle(image, end_pt, 1, (0, 0, 255), -1)
                    
            left_eye_start = landmarks[33]
            left_eye_end = landmarks[133]
            cv2.circle(image, tuple(map(int, left_eye_start)), 2, (0, 255, 255), -1)
            cv2.circle(image, tuple(map(int, left_eye_end)), 2, (255, 255, 0), -1)
            cv2.line(image, tuple(map(int, left_eye_start)), tuple(map(int, left_eye_end)), (255, 0, 0), 1)
            left_angle_info = calculate_vector_details(left_eye_start, left_eye_end)
            print(f"왼쪽 눈 각도: {left_angle_info['angle_degrees']:.2f}도")
            eye_width = np.linalg.norm(np.array(left_eye_end) - np.array(left_eye_start))
            print(f"왼쪽 눈 너비: {eye_width:.2f} 픽셀")

            nose_tip = landmarks[168]
            nose_base = landmarks[2]
            nose_length = np.linalg.norm(nose_tip - nose_base)
            print(f"코 길이: {nose_length:.2f}픽셀")

            left_nostril = landmarks[48]
            right_nostril = landmarks[278]
            nostril_distance = np.linalg.norm(left_nostril - right_nostril)
            print(f"코 양쪽 끝 거리: {nostril_distance:.2f}픽셀")

            nose_ratio = nose_length / nostril_distance
            print(f"코 비율 (길이/너비): {nose_ratio:.2f}")
            landmarks_data['NOSE']['nose_ratio'] = nose_ratio

        status = "success"

    save_json_data(image_path, landmarks_data, status, nose_ratio)
    save_processed_image(image_path, image, status)

    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Processed image and data.")
    return image

# JSON 파일 로드 함수
def load_json_file(json_file_path):
    if not os.path.exists(json_file_path):
        return {}

    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {json_file_path}. File may be corrupted.")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}

# JSON 파일 저장 함수
def save_json_file(json_file_path, data):
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# JSON 데이터 저장 및 업데이트
def save_json_data(image_path, landmarks_data, status, nose_ratio):
    json_folder = os.path.join(os.getcwd(), 'json')
    os.makedirs(json_folder, exist_ok=True)
    json_file_path = os.path.join(json_folder, "integration.json")

    existing_data = load_json_file(json_file_path)

    key = os.path.basename(image_path)
    counter = 1
    if status == "success":
        while key in existing_data:
            counter += 1
            key = f"{counter}_{os.path.basename(image_path)}"
        existing_data[key] = landmarks_data
    else:
        existing_data[key] = {}

    save_json_file(json_file_path, existing_data)

# 이미지 저장 함수
def save_processed_image(image_path, image, status):
    if status == "success":
        desired_folder = os.path.join(os.getcwd(), 'successFolder', 'successImg')
    else:
        desired_folder = os.path.join(os.getcwd(), 'failFolder', 'failImg')
    os.makedirs(desired_folder, exist_ok=True)

    output_image_path = os.path.join(desired_folder, os.path.basename(image_path))
    counter = 1
    while os.path.exists(output_image_path):
        output_image_path = os.path.join(desired_folder, f"{counter}_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)

# 이미지 표시 함수
def display_image(label, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    label.config(image=img_tk)
    label.image = img_tk

# 파일 선택 및 이미지 표시
def open_file():
    global current_image_key
    file_path = filedialog.askopenfilename()

    valid_extensions = (".jpg", ".jpeg", ".png")
    if not file_path.lower().endswith(valid_extensions):
        messagebox.showerror("Error", "지원하지 않는 파일 형식입니다.\nJPG, JPEG, PNG 파일만 허용됩니다.")
        return

    if file_path:
        current_image_key = os.path.basename(file_path)
        origin_img = cv2.imread(file_path)
        resized_origin_img = cv2.resize(origin_img, (200, 200))
        display_image(left_image_label, resized_origin_img)
        save_folder = os.path.join(os.getcwd(), 'saved_images')
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, current_image_key)
        cv2.imwrite(save_path, origin_img)
        messagebox.showinfo("Success", f"Image saved to {save_path}")

# Treeview 업데이트 함수
def update_treeview(data):
    tree.delete(*tree.get_children())

    for image_name, details in data.items():
        success_status = "성공" if details else "실패"
        tree.insert("", "end", values=(image_name, success_status))

# 파일 변경 감지 함수
def check_for_updates():
    global last_modified_time

    if not os.path.exists('json/integration.json'):
        return

    current_modified_time = os.path.getmtime('json/integration.json')

    if current_modified_time != last_modified_time:
        last_modified_time = current_modified_time
        data = load_json_file('json/integration.json')
        if data is not None:
            update_treeview(data)

    root.after(1000, check_for_updates)

# 이미지 변경 및 라벨 추가
def change_file():
    global current_image_key

    if current_image_key:
        save_path = os.path.join(os.getcwd(), 'saved_images', current_image_key)
        processed_image = process_image(save_path)

        if processed_image is not None:
            resized_landmark_img = cv2.resize(processed_image, (200, 200))
            display_image(right_image_label, resized_landmark_img)

            lips_label = lip_var.get()
            face_label = face_var.get()
            nose_label = nose_var.get()
            eye_label = eye_var.get()

            json_file_path = os.path.join(os.getcwd(), 'json', 'integration.json')
            json_data = load_json_file(json_file_path)

            update_labels_in_json(json_data, current_image_key, lips_label, face_label, nose_label, eye_label)
            save_json_file(json_file_path, json_data)

# 라벨 업데이트 함수
def update_labels_in_json(json_data, key, lips_label, face_label, nose_label, eye_label):
    if key in json_data:
        if 'LIPS' in json_data[key]:
            json_data[key]['LIPS']['lip_label'] = lips_label
        if 'FACE_OVAL' in json_data[key]:
            json_data[key]['FACE_OVAL']['face_label'] = face_label
        if 'NOSE' in json_data[key]:
            json_data[key]['NOSE']['nose_label'] = nose_label
        if 'LEFT_EYE' in json_data[key]:
            json_data[key]['LEFT_EYE']['eye_label'] = eye_label
        if 'RIGHT_EYE' in json_data[key]:
            json_data[key]['RIGHT_EYE']['eye_label'] = eye_label

# GUI 설정
root = tk.Tk()
root.title("Face Landmark Processor")
root.geometry("1200x900")

# Header 프레임
header_frame = tk.Frame(root)
header_frame.pack(pady=10)

open_button = tk.Button(root, text="파일 첨부", command=open_file)
open_button.pack(expand=True)

# 입술, 얼굴형, 눈, 코 메뉴
lip_options = ["얇은 입술", "두툼한 입술", "입꼬리가 내려간 입술", "입꼬리가 올라간 입술"]
lip_var = tk.StringVar(root)
lip_var.set(lip_options[0])
lip_menu = tk.OptionMenu(header_frame, lip_var, *lip_options)
lip_menu.pack(side=tk.LEFT, padx=5)

face_options = ["둥근형", "사각형", "계란형", "역삼각형", "긴형"]
face_var = tk.StringVar(root)
face_var.set(face_options[0])
face_menu = tk.OptionMenu(header_frame, face_var, *face_options)
face_menu.pack(side=tk.LEFT, padx=5)

eye_options = ["큰 눈", "작은 눈", "얇고 가느다란 눈", "동그란 눈", "올라간 눈꼬리", "쳐진 눈꼬리"]
eye_var = tk.StringVar(root)
eye_var.set(eye_options[0])
eye_menu = tk.OptionMenu(header_frame, eye_var, *eye_options)
eye_menu.pack(side=tk.LEFT, padx=5)

nose_options = ["긴 코", "짧은 코", "코 볼이 넓음", "코 볼이 작음"]
nose_var = tk.StringVar(root)
nose_var.set(nose_options[0])
nose_menu = tk.OptionMenu(header_frame, nose_var, *nose_options)
nose_menu.pack(side=tk.LEFT, padx=5)

# 변환 버튼
change_button = tk.Button(root, text="변환", command=change_file)
change_button.pack(expand=True)

# 이미지 박스 프레임
image_frame = tk.Frame(root)
image_frame.pack(pady=20)

# 왼쪽 이미지 박스
left_image_label = tk.Label(image_frame, bg="gray")
left_image_label.place(x=10, y=10, width=300, height=300)
left_image_label.pack(side=tk.LEFT, padx=10)

# 오른쪽 이미지 박스
right_image_label = tk.Label(image_frame, bg="gray")
right_image_label.place(x=220, y=10, width=300, height=300)
right_image_label.pack(side=tk.RIGHT, padx=10)

# Treeview 스타일 설정
style = ttk.Style(root)
style.configure("Treeview.Heading", font=("Arial", 17, "bold"))
style.configure("Treeview", rowheight=25)

# Treeview 생성
tree = ttk.Treeview(root, columns=("Image", "Status", "Label"), show="headings", height=15)
tree.heading("Image", text="이미지 이름")
tree.heading("Status", text="성공 여부")
tree.heading("Label", text="라벨")
tree.column("Image", width=100, anchor="center")
tree.column("Status", width=50, anchor="center")
tree.column("Label", width=50, anchor="center")
tree.pack(fill=tk.BOTH, expand=True)

# 데이터 로드 및 업데이트
data = load_json_file('json/integration.json')
update_treeview(data)

# 파일 감지
if os.path.exists('json/integration.json'):
    last_modified_time = os.path.getmtime('json/integration.json')
else:
    last_modified_time = None

root.after(1000, check_for_updates)
root.mainloop()
