#### Image_Face_Detect ####

import cv2
import os
import shutil
from mtcnn import MTCNN
import time 
import dlib
import re 

start_time = time.time()

source_folder = "crawled_data"
destination_folder = "face_image"

# MTCNN 모델 로드
mtcnn = MTCNN()

total_images = 0
face_detected_images = 0

for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):  # 이미지 파일 형식을 확인
        total_images += 1
        
        # 이미지 파일의 경로
        image_path = os.path.join(source_folder, filename)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        
        # 이미지 전처리 (RGB로 변환)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 감지
        faces = mtcnn.detect_faces(image_rgb)
        
        # 얼굴이 하나라도 감지된 경우
        if len(faces) > 0:
            face_detected_images += 1
            # 얼굴이 포함된 이미지를 destination_folder에 저장
            shutil.copy2(image_path, os.path.join(destination_folder, filename))
            print("Face detected in", filename)
        else:
            print("No face detected in", filename)

print("Face Detecting completed")
print(f"Total images: {total_images}")
print(f"Face detected images: {face_detected_images}")



#### Video_Face_Detect ####

dir_path = 'crawled_data'
save_dir = 'face_image'

file_num = 0

# mtcnn = MTCNN()
detector = dlib.get_frontal_face_detector()
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#디렉토리 내의 모든 파일을 순회
for filename in os.listdir(dir_path):
    # 파일 확장자가 .mp4 또는 .gif인 경우에만 처리
    if filename.endswith('.mp4') or filename.endswith('.gif'):
        file_path = os.path.join(dir_path, filename)
        cap = cv2.VideoCapture(file_path)
        
        # 비디오의 초당 프레임 수(FPS)를 구함
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 1초에 한 번씩 프레임을 추출하기 위해 프레임 간격을 설정
        frame_interval = int(fps)

        frame_num = 0 
        frame_count = 0
        capture_count = 0  # 캡쳐 횟수를 세는 변수
        
        # 파일명에서 영어만 추출
        filename_without_extension, extension = os.path.splitext(filename)

        # 파일명에서 영어,숫자만 추출
        filename_eng = re.sub('[^a-zA-Z0-9]', '', filename_without_extension)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            # 얼굴 탐지
            faces = detector(img)
            if len(faces) == 0:
                continue
            
             # 1초마다 한 번씩 프레임을 추출
            if frame_count % frame_interval == 0:
                 # 이미지 저장
                cv2.imwrite(f'{save_dir}/{os.path.splitext(filename_eng)[0]}_captured_frame_{frame_num+1}.jpg', img)


                frame_num += 1
                capture_count += 1   # 캡쳐 횟수를 세는 변수
            
            frame_count +=1   
            if capture_count == 5:  # 캡쳐 횟수를 세는 변수 == 5:
                break
            
    file_num += 1 

    
    
#### FaceNet_Euclidean_Calculate ####
    

from random import choice
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image 
from matplotlib import pyplot as plt 

import numpy as np
from numpy import asarray 
from numpy import savez_compressed
from numpy import load, expand_dims, linalg, argmin

from os import listdir
from os.path import join
import time 
import shutil 
import os 
from google.cloud import bigquery

# 모델 불러오기
model = load_model('model/facenet_keras.h5')

# 주어진 사진에서 하나의 얼굴 추출
def extract_face(filename, required_size=(160, 160)):
	# 파일에서 이미지 불러오기
	image = Image.open(filename)
	# RGB로 변환, 필요시
	image = image.convert('RGB')
	# 배열로 변환
	pixels = asarray(image)
	# 감지기 생성, 기본 가중치 이용
	detector = MTCNN()
	# 이미지에서 얼굴 감지
	results = detector.detect_faces(pixels)
	if len(results) ==0:
		print("No face detected in the image",filename)
		return None 
	# 첫 번째 얼굴에서 경계 상자 추출
	x1, y1, width, height = results[0]['box']
	# 버그 수정
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# 얼굴 추출
	face = pixels[y1:y2, x1:x2]
	# 모델 사이즈로 픽셀 재조정
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# 디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출
def load_faces(directory):
    faces = list()
    # 파일 열거
    for filename in listdir(directory):
        # 경로
        path = join(directory, filename) 
        # 얼굴 추출
        face = extract_face(path)
        if face is not None: # 이 부분을 추가합니다.
            # 저장
            faces.append(face)
    return faces

# 이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기
def load_dataset(directory):
    X, y = list(), list()
    for filename in listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg','.jfif')):
            path = join(directory, filename)
            face = extract_face(path)
            if face is not None:
                X.append(face)
                y.append(filename) # 파일 이름을 레이블로 사용
    return asarray(X), asarray(y)

# 훈련 데이터셋 불러오기
trainX, trainy = load_dataset('face_image')
print(trainX.shape, trainy.shape)
# 테스트 데이터셋 불러오기
testX, testy = load_dataset('user_image')
print(testX.shape, testy.shape)

# 배열을 단일 압축 포맷 파일로 저장
savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)

def get_embedding(model, face_pixels):
	# 픽셀 값의 척도
	face_pixels = face_pixels.astype('int32')
	# 채널 간 픽셀값 표준화(전역에 걸쳐)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# 얼굴을 하나의 샘플로 변환
	samples = expand_dims(face_pixels, axis=0)
	# 임베딩을 갖기 위한 예측 생성
	yhat = model.predict(samples)
	return yhat[0]

# 얼굴 데이터셋 불러오기
data = load('faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('불러오기: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# facenet 모델 불러오기
model = load_model('model/facenet_keras.h5')
print('모델 불러오기')

# 훈련 셋에서 각 얼굴을 임베딩으로 변환하기
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# 테스트 셋에서 각 얼굴을 임베딩으로 변환하기
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

# 배열을 하나의 압축 포맷 파일로 저장
savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

# # 테스트 이미지의 임베딩 얻기
# test_embedding = get_embedding(model, testX[0])

# 각 훈련 이미지와의 거리 계산
distances = []
for face_emb in newTrainX:
    distance = linalg.norm(face_emb - newTestX)
    distances.append(distance)

# 유클리드 거리 계산값 보여주기 
print("")
for i, distance in enumerate(distances):
    print(f"Image {trainy[i]}: Euclidean Distance = {distance:.4f}\n")
    

# 유클리드 거리가 9 이하인 경우, 해당 이미지를 candidate_face 디렉토리로 이동
dst_dir = 'candidate_face'
for i, distance in enumerate(distances):
    # 유클리드 거리가 9 이하인 경우
    if distance <= 9:
        # 원본 이미지 파일 경로
        src_path = f'face_image/{trainy[i]}'

        # 이미지 파일을 이동할 경로
        dst_path = os.path.join(dst_dir, trainy[i])

        # 파일 이동
        shutil.copy(src_path, dst_path)


end_time = time.time()

print("Spending time : ",end_time - start_time)




# 가장 가까운 얼굴의 인덱스 찾기
closest_face_index = argmin(distances)

closest_face_filename = trainy[closest_face_index]

# 가장 가까운 얼굴과의 유클리드 거리 계산
closest_distance = distances[closest_face_index]

image_table = 'strange-analog-405708.CrawlingDB.image'
video_table = 'strange-analog-405708.CrawlingDB.video'

# Google Cloud 클라이언트 설정
client = bigquery.Client()

# 이미지 테이블에서 쿼리 실행
image_query = f"""
    SELECT DISTINCT post_url FROM `{image_table}` WHERE img_file_name = '{closest_face_filename}'
"""
image_query_job = client.query(image_query)
image_results = [row.post_url for row in image_query_job.result()]

original_part_of_filename = closest_face_filename.split('_')[0]

print(original_part_of_filename)
# 추출된 부분을 사용하여 쿼리 수정
video_query = f"""
    SELECT DISTINCT post_url FROM `{video_table}` WHERE videofile_name LIKE '%{original_part_of_filename}%'
"""
video_query_job = client.query(video_query)
video_results = [row.post_url for row in video_query_job.result()]

# 결과 병합 및 중복 제거
all_results = list(set(image_results + video_results))

# 결과 출력
print("Unique Post URLs associated with the file:", all_results)



# 유클리드 거리가 Threshold보다 클 경우 "No Matching" 출력
if closest_distance > 9:
    plt.figure(figsize=(6, 6))
    plt.title('')
    plt.text(0.4, 0.3, 'No Matching', fontsize=20, ha='center')
    plt.axis('off')
    plt.show()
else:
    plt.figure(figsize=(16, 5))

    # 사용자 Face 보여주기 
    plt.subplot(1, 2, 1)
    plt.imshow(testX[0])
    plt.title('Test Face')
    plt.axis('off')

    # 가장 닮은 얼굴 보여주기
    plt.subplot(1, 2, 2)
    plt.imshow(trainX[closest_face_index])
    plt.title('Most Similar Face: ' + trainy[closest_face_index])
    plt.axis('off')

    # 유클리드 거리 표시 (전체 그림에 대해 상대적 위치 사용)
    plt.figtext(0.5, 0.1, f'Euclidean Distance: {closest_distance:.4f}', fontsize=10, ha='center')

    # URL 목록 표시
    plt.figtext(0.5, 0.01, "Associated URLs:\n" + "\n".join(all_results), ha="center", fontsize=10)

    plt.savefig('Face_Comparison.png')
    plt.show()

   
    

    
    



