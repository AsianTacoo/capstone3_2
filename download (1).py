from google.cloud import bigquery
from google.cloud import storage
import os

# 빅쿼리 클라이언트 생성
bq_client = bigquery.Client()
local_download_dir = 'Total/crawled_data'
date_to_query = '2023-12-13'

# 쿼리 작성
image_query = f"""
    SELECT file_path
    FROM `strange-analog-405708.CrawlingDB.image`
    WHERE DATE(post_date) = DATE('{date_to_query}')
"""
# 쿼리 실행
image_results = bq_client.query(image_query).result()

# 비디오 쿼리
video_query = f"""
    SELECT file_path
    FROM `strange-analog-405708.CrawlingDB.video`
    WHERE DATE(post_date) = DATE('{date_to_query}')
"""
video_results = bq_client.query(video_query).result()

# 결과 병합 및 중복 제거
all_file_paths = list(set([row.file_path for row in image_results] + [row.file_path for row in video_results]))

# 출력
print(all_file_paths)

# 스토리지 클라이언트 생성
storage_client = storage.Client()

# 결과에 대해 반복하며 파일 다운로드
for file_path_in_bucket in all_file_paths:
    # 버킷 이름과 파일 이름 추출
    bucket_name = file_path_in_bucket.split('/')[2]
    file_name_in_bucket = '/'.join(file_path_in_bucket.split('/')[3:])
    local_file_name = os.path.join(local_download_dir, file_name_in_bucket.split('/')[-1])

    # 버킷과 Blob 가져오기
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name_in_bucket)

    # 다운로드할 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)

    # 파일 다운로드
    blob.download_to_filename(local_file_name)
    print(f"Downloaded {file_name_in_bucket} to {local_file_name}")