import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib import font_manager, rc

rc('font', family='AppleGothic')

# 데이터 처리 및 모델 관련 함수들
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lips_data_json = json.load(file)
    
    # 데이터 구조 확인
    if isinstance(lips_data_json, list):
        data_items = [(item.get('image_name', f"image_{i}"), item) for i, item in enumerate(lips_data_json)]
    elif isinstance(lips_data_json, dict):
        data_items = lips_data_json.items()
    else:
        raise ValueError("지원되지 않는 JSON 구조입니다.")
    
    lips_data = []
    labels = []
    
    for key, item in data_items:
        if 'LIPS' in item:
            lips_part = item['LIPS']
            vectors = lips_part.get('values', [])
            lip_label = lips_part.get('lip_label', None)
            
            if lip_label is None:
                print(f"경고: '{key}'에 'lip_label'이 없습니다. 해당 데이터를 스킵합니다.")
                continue
            
            lips_points = []
            for vec in vectors:
                lips_points.extend(vec['vector'])            # (Δx, Δy)
                lips_points.append(vec['magnitude'])         # 벡터의 크기
                lips_points.append(vec['angle_degrees'])     # 각도
                lips_points.append(vec['angle_normalized'])  # 정규화된 각도
                lips_points.extend(vec['direction'])         # 방향 (단위 벡터)
            
            lips_data.append(lips_points)
            labels.append(lip_label)
        else:
            print(f"경고: '{key}'에 'LIPS' 데이터가 없습니다. 해당 데이터를 스킵합니다.")
    
    if len(lips_data) == 0:
        raise ValueError("유효한 'LIPS' 데이터를 찾을 수 없습니다.")
    
    lips_data = np.array(lips_data)
    labels = np.array(labels)
    
    return lips_data, labels

def load_prediction_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lips_data_json = json.load(file)
    
    # 단일 샘플인지 확인
    if isinstance(lips_data_json, list):
        raise ValueError("예측 데이터는 하나의 샘플이어야 합니다.")
    
    if not isinstance(lips_data_json, dict) or len(lips_data_json) != 1:
        raise ValueError("예측 데이터 JSON은 하나의 이미지에 대한 데이터여야 합니다.")
    
    # 이미지 이름을 키로 사용
    image_name, image_data = next(iter(lips_data_json.items()))
    
    if 'LIPS' not in image_data:
        raise ValueError("JSON 파일에 'LIPS' 데이터가 없습니다.")
    
    lips_part = image_data['LIPS']
    vectors = lips_part.get('values', [])
    
    if not vectors:
        raise ValueError("'LIPS' 데이터에 'values' 배열이 없습니다.")
    
    lips_points = []
    for vec in vectors:
        lips_points.extend(vec['vector'])            # (Δx, Δy)
        lips_points.append(vec['magnitude'])         # 벡터의 크기
        lips_points.append(vec['angle_degrees'])     # 각도
        lips_points.append(vec['angle_normalized'])  # 정규화된 각도
        lips_points.extend(vec['direction'])         # 방향 (단위 벡터)
    
    return lips_points

def preprocess_data(lips_data, labels):
    scaler = StandardScaler()
    lips_data_normalized = scaler.fit_transform(lips_data)
    
    label_encoder = LabelEncoder()
    labels_int = label_encoder.fit_transform(labels)
    
    label_counts = Counter(labels_int)
    classes_to_keep = [label for label, count in label_counts.items() if count >= 2]
    other_class = len(classes_to_keep)
    
    labels_mapped = np.array([label if label in classes_to_keep else other_class for label in labels_int])
    
    label_encoder = LabelEncoder()
    labels_encoded_int = label_encoder.fit_transform(labels_mapped)
    
    num_classes = len(label_encoder.classes_)
    labels_encoded = tf.keras.utils.to_categorical(labels_encoded_int, num_classes)
    
    return lips_data_normalized, labels_encoded, labels_encoded_int, label_encoder, scaler, num_classes

def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, parent_frame):
    history_dict = history.history
    acc = history_dict.get('accuracy', [])
    val_acc = history_dict.get('val_accuracy', [])
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])
    
    epochs_range = range(1, len(acc) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # 정확도 그래프
    axs[0].plot(epochs_range, acc, 'bo-', label='훈련 정확도')
    axs[0].plot(epochs_range, val_acc, 'r*-', label='검증 정확도')
    axs[0].set_title('훈련 및 검증 정확도')
    axs[0].set_xlabel('에포크')
    axs[0].set_ylabel('정확도')
    axs[0].legend()
    
    # 손실 그래프
    axs[1].plot(epochs_range, loss, 'bo-', label='훈련 손실')
    axs[1].plot(epochs_range, val_loss, 'r*-', label='검증 손실')
    axs[1].set_title('훈련 및 검증 손실')
    axs[1].set_xlabel('에포크')
    axs[1].set_ylabel('손실')
    axs[1].legend()
    
    fig.tight_layout()
    
    # Tkinter에 matplotlib 그림 추가
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

class LipPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("입술 모양 예측기")
        master.geometry("1200x800")
        
        # 초기 변수
        self.lips_data = None
        self.labels = None
        self.lips_data_normalized = None
        self.labels_encoded = None
        self.labels_encoded_int = None
        self.label_encoder = None
        self.scaler = None
        self.num_classes = None
        self.model = None
        self.history = None
        
        # GUI 구성
        self.create_widgets()
    
    def create_widgets(self):
        # 프레임 나누기
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(pady=10)
        
        self.middle_frame = tk.Frame(self.master)
        self.middle_frame.pack(pady=10)
        
        self.bottom_frame = tk.Frame(self.master)
        self.bottom_frame.pack(pady=10)
        
        # 데이터 로드 버튼
        self.load_button = tk.Button(self.top_frame, text="데이터셋 로드", command=self.load_dataset)
        self.load_button.pack(side=tk.LEFT, padx=10)
        
        # 모델 학습 버튼
        self.train_button = tk.Button(self.top_frame, text="모델 학습", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(side=tk.LEFT, padx=10)
        
        # 모델 평가 버튼
        self.evaluate_button = tk.Button(self.top_frame, text="모델 평가", command=self.evaluate_model, state=tk.DISABLED)
        self.evaluate_button.pack(side=tk.LEFT, padx=10)
        
        # 초기화 버튼 추가
        self.reset_button = tk.Button(self.top_frame, text="초기화", command=self.reset, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        # 라벨 매핑 확인 버튼 추가
        self.show_mapping_button = tk.Button(self.top_frame, text="라벨 매핑 확인", command=self.show_label_mapping, state=tk.DISABLED)
        self.show_mapping_button.pack(side=tk.LEFT, padx=10)
        
        # 예측 섹션
        self.predict_label = tk.Label(self.middle_frame, text="새로운 입술 데이터 입력:")
        self.predict_label.pack()
        
        self.new_data_text = tk.Text(self.middle_frame, height=10, width=100)
        self.new_data_text.pack(pady=5)
        
        # 예측 데이터 로드 버튼 추가
        self.load_predict_button = tk.Button(self.middle_frame, text="예측 데이터 로드", command=self.load_predict_data, state=tk.DISABLED)
        self.load_predict_button.pack(pady=5)
        
        self.predict_button = tk.Button(self.middle_frame, text="예측하기", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=5)
        
        self.prediction_result = tk.Label(self.middle_frame, text="", font=("Helvetica", 14))
        self.prediction_result.pack(pady=5)
        
        # 그래프 표시 섹션
        self.graph_frame = tk.Frame(self.bottom_frame)
        self.graph_frame.pack()
    
    def load_dataset(self):
        filepath = filedialog.askopenfilename(
            title="JSON 파일 선택",
            filetypes=[("JSON 파일", "*.json")]
        )
        if not filepath:
            return
        try:
            self.lips_data, self.labels = load_data(filepath)
            self.lips_data_normalized, self.labels_encoded, self.labels_encoded_int, self.label_encoder, self.scaler, self.num_classes = preprocess_data(self.lips_data, self.labels)
            messagebox.showinfo("성공", f"데이터셋 로드 성공!\n샘플 수: {len(self.lips_data)}\n클래스 수: {self.num_classes}")
            self.train_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)  # 초기화 버튼 활성화
            self.load_predict_button.config(state=tk.NORMAL)  # 예측 데이터 로드 버튼 활성화
            self.show_mapping_button.config(state=tk.NORMAL)  # 라벨 매핑 확인 버튼 활성화
        except Exception as e:
            messagebox.showerror("에러", f"데이터 로드 실패:\n{e}")
    
    def train_model(self):
        try:
            # 데이터셋 분할
            X_train, X_test, y_train, y_test = train_test_split(
                self.lips_data_normalized, self.labels_encoded, test_size=0.2, random_state=42, stratify=self.labels_encoded_int
            )
            
            # 모델 빌드
            input_dim = self.lips_data_normalized.shape[1]
            self.model = build_model(input_dim, self.num_classes)
            
            # 학습 콜백 설정
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # 모델 학습
            self.history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0  # 학습 진행 상황을 GUI로 대체
            )
            
            messagebox.showinfo("성공", f"모델 학습 완료!\n최종 검증 정확도: {self.history.history['val_accuracy'][-1]:.4f}")
            self.evaluate_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.NORMAL)
            
            # 그래프 그리기
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            plot_history(self.history, self.graph_frame)
            
        except Exception as e:
            messagebox.showerror("에러", f"모델 학습 실패:\n{e}")
    
    def evaluate_model(self):
        try:
            if self.model is None:
                messagebox.showwarning("경고", "모델이 학습되지 않았습니다.")
                return
            # 데이터셋 분할
            X_train, X_test, y_train, y_test = train_test_split(
                self.lips_data_normalized, self.labels_encoded, test_size=0.2, random_state=42, stratify=self.labels_encoded_int
            )
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
            messagebox.showinfo("모델 평가", f"테스트 데이터 정확도: {test_acc:.4f}")
        except Exception as e:
            messagebox.showerror("에러", f"모델 평가 실패:\n{e}")
    
    def load_predict_data(self):
        filepath = filedialog.askopenfilename(
            title="예측 데이터 JSON 파일 선택",
            filetypes=[("JSON 파일", "*.json")]
        )
        if not filepath:
            return
        try:
            predict_lips_points = load_prediction_data(filepath)
            # 입력 데이터를 쉼표로 구분된 문자열로 변환
            input_str = ', '.join(map(str, predict_lips_points))
            self.new_data_text.delete("1.0", tk.END)
            self.new_data_text.insert(tk.END, input_str)
            messagebox.showinfo("성공", "예측 데이터가 텍스트 박스에 로드되었습니다.")
            self.predict_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("에러", f"예측 데이터 로드 실패:\n{e}")
    
    def predict(self):
        try:
            if self.model is None:
                messagebox.showwarning("경고", "모델이 학습되지 않았습니다.")
                return
            # 사용자가 입력한 데이터를 가져오기
            raw_input = self.new_data_text.get("1.0", tk.END).strip()
            if not raw_input:
                messagebox.showwarning("경고", "입력 데이터가 비어 있습니다.")
                return
            # 입력 데이터를 리스트로 변환
            input_values = list(map(float, raw_input.split(',')))
            
            # 입력된 값의 개수 확인
            expected_length = self.lips_data_normalized.shape[1]
            if len(input_values) != expected_length:
                messagebox.showerror("에러", f"입력된 값의 개수가 올바르지 않습니다. 예상되는 값의 개수: {expected_length}")
                return
            
            input_array = np.array(input_values).reshape(1, -1)
            
            # 데이터 정규화
            input_normalized = self.scaler.transform(input_array)
            
            # 예측
            prediction = self.model.predict(input_normalized)
            predicted_label_int = np.argmax(prediction)
            predicted_label_name = self.label_encoder.inverse_transform([predicted_label_int])[0]
            
            print(f"예측된 인덱스: {predicted_label_int}")
            print(f"예측된 라벨: {predicted_label_name}")
            
            self.prediction_result.config(text=f"예측된 입모양 카테고리: {predicted_label_name}")
        except ValueError:
            messagebox.showerror("에러", "입력 데이터에 숫자가 아닌 값이 포함되어 있습니다.")
        except Exception as e:
            messagebox.showerror("에러", f"예측 실패:\n{e}")
    
    def reset(self):
        try:
            # 모든 데이터와 모델 변수 초기화
            self.lips_data = None
            self.labels = None
            self.lips_data_normalized = None
            self.labels_encoded = None
            self.labels_encoded_int = None
            self.label_encoder = None
            self.scaler = None
            self.num_classes = None
            self.model = None
            self.history = None
            
            # GUI 요소 초기화
            self.new_data_text.delete("1.0", tk.END)
            self.prediction_result.config(text="")
            
            # 그래프 프레임 초기화
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            
            # 버튼 상태 초기화
            self.train_button.config(state=tk.DISABLED)
            self.evaluate_button.config(state=tk.DISABLED)
            self.predict_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.load_predict_button.config(state=tk.DISABLED)
            self.show_mapping_button.config(state=tk.DISABLED)
            
            messagebox.showinfo("초기화 완료", "모든 데이터와 모델이 초기화되었습니다.")
        except Exception as e:
            messagebox.showerror("에러", f"초기화 실패:\n{e}")
    
    def show_label_mapping(self):
        try:
            if self.label_encoder is None:
                messagebox.showwarning("경고", "라벨 인코더가 초기화되지 않았습니다.")
                return
            classes = self.label_encoder.classes_
            mapping = "\n".join([f"{i}: {cls}" for i, cls in enumerate(classes)])
            # 새로운 창을 열어 라벨 매핑을 표시
            mapping_window = tk.Toplevel(self.master)
            mapping_window.title("라벨 매핑")
            mapping_window.geometry("300x400")
            label = tk.Label(mapping_window, text="라벨 매핑:\n" + mapping, justify=tk.LEFT, anchor='nw')
            label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("에러", f"라벨 매핑 표시 실패:\n{e}")

def main():
    root = tk.Tk()
    app = LipPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
