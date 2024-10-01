import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from tkinter import ttk
import math  # 각도 계산을 위해 추가

# Mediapipe 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

current_image_key = None # 현재 이미지 키를 전역 변수로 설정

# 벡터의 크기, 각도, 정규화된 각도 등을 계산하는 함수
def calculate_vector_details(start_point, end_point):
    # 벡터 계산
    vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    
    # 벡터의 크기 (Euclidean Distance)
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    
    # 벡터의 각도 (x축과의 각도, 라디안 -> 도)
    angle_degrees = math.degrees(math.atan2(vector[1], vector[0]))
    
    # 각도를 0~360도로 정규화
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
        
        # 벡터의 세부 정보 계산
        vector_details = calculate_vector_details(start_point, end_point)
        
        # 벡터의 방향을 정규화 (단위 벡터)
        if vector_details['magnitude'] != 0:
            direction = (vector_details['vector'][0] / vector_details['magnitude'],
                         vector_details['vector'][1] / vector_details['magnitude'])
        else:
            direction = (0, 0)
        
        # 필요한 값을 모두 포함하는 딕셔너리를 리스트에 추가
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

def process_image(image_path):
    global current_image_key
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 랜드마크 탐지
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
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

            # 각 부위별 벡터 계산 및 저장
            for part_name, connections in [('LEFT_EYE', LEFT_EYE),
                                           ('RIGHT_EYE', RIGHT_EYE),
                                           ('NOSE', NOSE),
                                           ('LIPS', LIPS),
                                           ('FACE_OVAL', FACE_OVAL)]:
                vectors = calculate_vectors(connections, landmarks, w, h)
                if part_name not in landmarks_data:
                    landmarks_data[part_name] = {'values': []}

                # 'vectors' 리스트에서 필요 없는 start_idx, end_idx를 제외하고 필터링
                filtered_vectors = [{'vector': vec['vector'],               # 벡터 값 추가
                     'magnitude': vec['magnitude'],
                     'angle_degrees': vec['angle_degrees'],  # 각도 값 포함
                     'angle_normalized': vec['angle_normalized'],
                     'direction': vec['direction']} for vec in vectors]

                # 벡터의 크기, 각도, 방향만 저장
                landmarks_data[part_name]['values'] = filtered_vectors

                # 벡터 시각화 (선택 사항)
                for vec in vectors:
                    start_pt = tuple(map(int, landmarks[vec['start_idx']]))
                    end_pt = tuple(map(int, landmarks[vec['end_idx']]))
                    cv2.line(image, start_pt, end_pt, (0, 255, 0), 1)
                    # 시작점과 끝점에 작은 원 표시 (선택 사항)
                    cv2.circle(image, start_pt, 1, (255, 0, 0), -1)
                    cv2.circle(image, end_pt, 1, (0, 0, 255), -1)

        status = "success"
    else:
        status = "fail"

    # JSON 저장 경로 설정
    json_folder = os.path.join(os.getcwd(), 'json')
    os.makedirs(json_folder, exist_ok=True)

    json_file_path = os.path.join(json_folder, "integration.json")

    # 기존 JSON 파일 불러오기
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    counter = 1  # 중복 파일을 위한 카운터 초기화

    # 이미지 파일 이름을 키로 사용하여 새로운 데이터 추가
    key = os.path.basename(image_path)
    if status == "success":
        while key in existing_data:
            counter += 1
            key = f"{counter}_{os.path.basename(image_path)}"
        existing_data[key] = landmarks_data
    else:
        existing_data[key] = {}

    # 결과를 JSON 파일으로 저장
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

    # 데이터 로드 및 UI 업데이트 (함수 정의 필요)
    data = load_json_data()
    update_treeview(data)

    # 이미지 저장 폴더 설정
    if status == "success":
        desired_folder = os.path.join(os.getcwd(), 'successFolder', 'successImg')
    else:
        desired_folder = os.path.join(os.getcwd(), 'failFolder', 'failImg')
    os.makedirs(desired_folder, exist_ok=True)

    # 이미지 저장 경로 설정
    output_image_path = os.path.join(desired_folder, os.path.basename(image_path))
    while os.path.exists(output_image_path):
        output_image_path = os.path.join(desired_folder, f"{counter}_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, image)

    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Processed data saved to {json_file_path}")
    return image

# 라벨링 선택 함수
def label_image(label):
    global current_image_key
    
    if current_image_key is None:
        messagebox.showerror("Error", "이미지를 선택해 주세요.")
        return
    
    # 기존 JSON 데이터 로드
    json_file_path = os.path.join(os.getcwd(), 'json', "integration.json")
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
    
    # 기존 JSON 데이터 로드
    json_file_path = os.path.join(os.getcwd(), 'json', "integration.json")
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
    
    # # 선택한 라벨을 추가
    # if current_image_key in existing_data:
    #     existing_data[current_image_key]['lip_label'] = label
    # else:
    #     # 중복 처리된 이미지 키를 찾기
    #     for key in existing_data.keys():
    #         if key.startswith(os.path.basename(current_image_key)):
    #             existing_data[key]['lip_label'] = label
    #             break
    #     else:
    #         messagebox.showerror("Error", "이미지 키가 존재하지 않습니다.")
    #         return

    # # 결과를 JSON 파일로 저장
    # with open(json_file_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

    # messagebox.showinfo("Success", f"{label} 라벨이 저장되었습니다.")


# 파일 열기 함수
def open_file():
    global current_image_key
    file_path = filedialog.askopenfilename()
    
    # 허용되는 파일 확장자 목록
    valid_extensions = (".jpg", ".jpeg", ".png")
    # 파일 확장자 확인
    if not file_path.lower().endswith(valid_extensions):
        messagebox.showerror("Error", "지원하지 않는 파일 형식입니다.\nJPG, JPEG, PNG 파일만 허용됩니다.")
        return
    
    if file_path:
        current_image_key = os.path.basename(file_path)
        origin_img = cv2.imread(file_path)
        # 이미지 크기 조정
        resized_origin_img = cv2.resize(origin_img, (200, 200))
        # 왼족 이미지 박스에 원본 이미지 표시
        display_image(left_image_label, resized_origin_img)
        # 파일 저장 경로 설정
        save_folder = os.path.join(os.getcwd(), 'saved_images')  # 저장할 폴더
        os.makedirs(save_folder, exist_ok=True)  # 폴더가 없으면 생성
        
        # 이미지 저장
        save_path = os.path.join(save_folder, current_image_key)
        cv2.imwrite(save_path, origin_img)  # 원본 이미지를 저장
        messagebox.showinfo("Success", f"Image saved to {save_path}")
        
# Tkinter 레이블에 이미지 표시 함수
def display_image(label, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    label.config(image=img_tk)
    label.image = img_tk  # 이미지 참조를 유지
    
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

def change_file():
    global current_image_key
    # 랜드마크 이미지 처리
    if current_image_key:  # 현재 이미지 키가 설정되어 있을 때만 처리
        # save_path 변수에 저장할 경로 설정
        save_path = os.path.join(os.getcwd(), 'saved_images', current_image_key)  # 원하는 경로로 설정
        # 이미지 경로를 이용해 랜드마크를 처리
        processed_image = process_image(save_path)  # process_image에 save_path 전달
        if processed_image is not None:  # 이미지가 성공적으로 처리되었을 경우에만
            resized_landmark_img = cv2.resize(processed_image, (200, 200))  # 이미지 크기 조정
            display_image(right_image_label, resized_landmark_img)  # 오른쪽 이미지 박스에 표시
            # 선택한 라벨을 가져오기
            lip_label = lip_var.get()  # 입술 라벨
            face_label = face_var.get()  # 얼굴형 라벨
            eye_label = eye_var.get()   # 눈 라
            nose_label = nose_var.get()  # 코 라벨

            # 기존 JSON 데이터 로드
            json_file_path = os.path.join(os.getcwd(), 'json', "integration.json")
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)

            # 선택한 라벨 추가
            if current_image_key in existing_data:
                if 'LIPS' in existing_data[current_image_key]:
                    existing_data[current_image_key]['LIPS']['lip_label'] = lip_label
    
                if 'FACE_OVAL' in existing_data[current_image_key]:
                    existing_data[current_image_key]['FACE_OVAL']['face_label'] = face_label
    
                if 'NOSE' in existing_data[current_image_key]:
                    existing_data[current_image_key]['NOSE']['nose_label'] = nose_label
    
                if 'LEFT_EYE' in existing_data[current_image_key]:
                    existing_data[current_image_key]['LEFT_EYE']['eye_label'] = eye_label
                    
                if 'RIGHT_EYE' in existing_data[current_image_key]:
                    existing_data[current_image_key]['RIGHT_EYE']['eye_label'] = eye_label

                # 결과를 JSON 파일로 저장
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
# GUI 설정
root = tk.Tk()
root.title("Face Landmark Processor")
root.geometry("1200x900")

# Header 프레임
header_frame = tk.Frame(root)
header_frame.pack(pady=10)

open_button = tk.Button(root, text="파일 첨부", command=open_file)
open_button.pack(expand=True)

# 4개의 목록을 OptionMenu로 변환
# 1. 입술 메뉴
lip_options = ["얇은 입술", "두툼한 입술", "입꼬리가 내려간 입술", "입꼬리가 올라간 입술"]
lip_var = tk.StringVar(root)
lip_var.set(lip_options[0])  # 기본값 설정
lip_menu = tk.OptionMenu(header_frame, lip_var, *lip_options)
lip_menu.pack(side=tk.LEFT, padx=5)

# 2. 얼굴형 메뉴
face_options = ["둥근형", "사각형", "계란형", "역삼각형", "긴형"]
face_var = tk.StringVar(root)
face_var.set(face_options[0])  # 기본값 설정
face_menu = tk.OptionMenu(header_frame, face_var, *face_options)
face_menu.pack(side=tk.LEFT, padx=5)

# 3. 눈 메뉴
eye_options = ["큰 눈", "작은 눈", "얇고 가느다란 눈", "동그란 눈", "올라간 눈꼬리", "쳐진 눈꼬리"]
eye_var = tk.StringVar(root)
eye_var.set(eye_options[0])  # 기본값 설정
eye_menu = tk.OptionMenu(header_frame, eye_var, *eye_options)
eye_menu.pack(side=tk.LEFT, padx=5)

# 4. 코 메뉴
nose_options = ["긴 코", "짧은 코", "코 볼이 넓음", "코 볼이 작음"]
nose_var = tk.StringVar(root)
nose_var.set(nose_options[0])  # 기본값 설정
nose_menu = tk.OptionMenu(header_frame, nose_var, *nose_options)
nose_menu.pack(side=tk.LEFT, padx=5)

#변환 버튼
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

# 타이틀 스타일 설정
style = ttk.Style(root)
style.configure("Treeview.Heading", font=("Arial", 17, "bold"))  # 열 제목 폰트 크기 설정
style.configure("Treeview", rowheight=25)  # 행 높이 설정

# Treeview 생성
tree = ttk.Treeview(root, columns=("Image", "Status", "Label"), show="headings", height=15)
tree.heading("Image", text="이미지 이름")
tree.heading("Status", text="성공 여부")
tree.heading("Label", text="라벨")
tree.column("Image", width=100, anchor="center")
tree.column("Status", width=50, anchor="center")
tree.column("Label", width=50, anchor="center")
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