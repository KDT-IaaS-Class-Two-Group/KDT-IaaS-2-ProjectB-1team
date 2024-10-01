import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from collections import Counter

# 1. JSON 데이터 로드 및 벡터 추출
with open('integration.json', 'r', encoding='utf-8') as file:
    lips_data_json = json.load(file)

# 데이터 구조 확인
print(f"JSON 데이터 유형: {type(lips_data_json)}")
if isinstance(lips_data_json, list):
    print("JSON 데이터는 리스트입니다. 각 요소를 딕셔너리로 처리합니다.")
elif isinstance(lips_data_json, dict):
    print("JSON 데이터는 딕셔너리입니다.")
else:
    raise ValueError("지원되지 않는 JSON 구조입니다.")

# 입술 점 데이터와 라벨 분리
lips_data = []
labels = []

# JSON 데이터가 딕셔너리인 경우
if isinstance(lips_data_json, dict):
    data_items = lips_data_json.items()
elif isinstance(lips_data_json, list):
    data_items = [(item.get('image_name', f"image_{i}"), item) for i, item in enumerate(lips_data_json)]
else:
    data_items = []

for key, item in data_items:
    # 'LIPS' 데이터가 존재하는지 확인
    if 'LIPS' in item:
        lips_part = item['LIPS']
        vectors = lips_part.get('values', [])
        lip_label = lips_part.get('lip_label', None)
        
        # 라벨이 없는 경우 스킵
        if lip_label is None:
            print(f"경고: '{key}'에 'lip_label'이 없습니다. 해당 데이터를 스킵합니다.")
            continue
        
        # 벡터 데이터를 하나의 리스트로 결합
        lips_points = []
        for vec in vectors:
            # 벡터 정보 추출
            lips_points.extend(vec['vector'])            # (Δx, Δy)
            lips_points.append(vec['magnitude'])         # 벡터의 크기
            lips_points.append(vec['angle_degrees'])     # 각도
            lips_points.append(vec['angle_normalized'])  # 정규화된 각도
            lips_points.extend(vec['direction'])         # 방향 (단위 벡터)
        
        lips_data.append(lips_points)
        labels.append(lip_label)
    else:
        print(f"경고: '{key}'에 'LIPS' 데이터가 없습니다. 해당 데이터를 스킵합니다.")

# 데이터 확인 (선택 사항)
if len(lips_data) == 0:
    raise ValueError("유효한 'LIPS' 데이터를 찾을 수 없습니다.")

print(f"입술 데이터 샘플 수: {len(lips_data)}")
print(f"라벨 샘플 수: {len(labels)}")

# NumPy 배열로 변환
lips_data = np.array(lips_data)
labels = np.array(labels)

# 라벨 데이터 검증
print("라벨 데이터 유형:", labels.dtype)
print("라벨 데이터 샘플:", labels[:5])
print("라벨 데이터 타입:", [type(label) for

 label in labels[:5]])

# 2. 데이터 정규화 및 라벨 인코딩
# 데이터 정규화
scaler = StandardScaler()
lips_data_normalized = scaler.fit_transform(lips_data)

# 라벨 인코딩 (문자열을 정수로 변환)
label_encoder = LabelEncoder()
labels_int = label_encoder.fit_transform(labels)

# 라벨 데이터 분포 확인
label_counts = Counter(labels_int)
print("라벨 분포:", label_counts)

# 샘플 수가 2개 미만인 클래스 찾기
min_samples = 2
classes_to_keep = [label for label, count in label_counts.items() if count >= min_samples]
print("유지할 클래스:", classes_to_keep)

# 'Other' 클래스 번호
other_class = len(classes_to_keep)

# 라벨 재매핑
labels_mapped = np.array([label if label in classes_to_keep else other_class for label in labels_int])

# 'Other' 클래스가 있는지 확인
new_label_counts = Counter(labels_mapped)
print("새 라벨 분포:", new_label_counts)

# 라벨 인코딩 (다시)
label_encoder = LabelEncoder()
labels_encoded_int = label_encoder.fit_transform(labels_mapped)

# 원-핫 인코딩 (다중 클래스 분류를 위한 라벨 처리)
num_classes = len(label_encoder.classes_)  # 라벨의 고유한 수
labels_encoded = tf.keras.utils.to_categorical(labels_encoded_int, num_classes)

# 데이터 확인 (선택 사항)
print(f"정규화된 입술 데이터 샘플 수: {lips_data_normalized.shape}")
print(f"인코딩된 라벨 샘플 수: {labels_encoded.shape}")
print(f"정규화된 입술 데이터 샘플 (첫 번째): {lips_data_normalized[0]}")
print(f"인코딩된 라벨 샘플 (첫 번째): {labels_encoded[0]}")

# 3. 데이터셋 분할
# stratify=labels_encoded를 stratify=labels_encoded_int로 변경
X_train, X_test, y_train, y_test = train_test_split(
    lips_data_normalized, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded_int
)

# 데이터셋 크기 확인 (선택 사항)
print(f"훈련 데이터 크기: {X_train.shape}, 훈련 라벨 크기: {y_train.shape}")
print(f"테스트 데이터 크기: {X_test.shape}, 테스트 라벨 크기: {y_test.shape}")

# 4. 신경망 모델 구성
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(lips_data_normalized.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 모델 요약 정보 확인 (선택 사항)
model.summary()

# 5. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 모델 학습
# tf.data.Dataset을 사용하여 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 데이터셋을 배치하고, 셔플링 및 반복 설정
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# EarlyStopping 콜백 설정 (선택 사항)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 모델 학습
history = model.fit(
    train_dataset,
    epochs=30,  # 원하는 에포크 수로 설정 (현재는 100으로 설정)
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

# 7. 모델 평가
test_loss, test_acc = model.evaluate(test_dataset)
print(f"테스트 데이터 정확도: {test_acc:.4f}")

# 8. 학습 결과 시각화
history_dict = history.history

# 정확도와 손실 값 가져오기
acc = history_dict.get('accuracy', [])
val_acc = history_dict.get('val_accuracy', [])
loss = history_dict.get('loss', [])
val_loss = history_dict.get('val_loss', [])

epochs_range = range(1, len(acc) + 1)

# 한글 폰트 설정 (macOS)
rc('font', family='AppleGothic')

# 마이너스 기호가 깨지는 문제 해결
plt.rcParams['axes.unicode_minus'] = False

# 그래프 그리기
plt.figure(figsize=(14, 5))

# 1) 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo-', label='훈련 정확도')
plt.plot(epochs_range, val_acc, 'r*-', label='검증 정확도')
plt.title('훈련 및 검증 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()

# 2) 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo-', label='훈련 손실')
plt.plot(epochs_range, val_loss, 'r*-', label='검증 손실')
plt.title('훈련 및 검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()

plt.tight_layout()
plt.show()

# 9. 새로운 데이터로 예측 (선택 사항)
# 예측을 위한 새로운 입모양 데이터 예시
# new_lips_data = np.array([
#     # 벡터1: [Δx, Δy, magnitude, angle_degrees, angle_normalized, dir_x, dir_y],
#     5.0, -3.0, 5.830951894845301, -30.0, 330.0, 0.8574929257125441, -0.5144957554275265,
#     # 벡터2: [Δx, Δy, magnitude, angle_degrees, angle_normalized, dir_x, dir_y],
#     3.0, -4.0, 5.0, -53.13, 306.87, 0.6, -0.8
#     # 추가 벡터들...
# ])
# new_lips_data = new_lips_data.reshape(1, -1)  # 2D 배열로 변환
# new_lips_data_normalized = scaler.transform(new_lips_data)
# prediction = model.predict(new_lips_data_normalized)
# predicted_label = np.argmax(prediction)
# predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]
# print(f"예측된 입모양 카테고리: {predicted_label_name}")