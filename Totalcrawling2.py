import requests
from urllib import request
from bs4 import BeautifulSoup
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
from google.cloud import storage
from google.cloud import bigquery
from selenium.common.exceptions import NoSuchElementException
from urllib.parse import urlparse, parse_qs
import logging


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

session = requests.Session()

# 헤더 설정
headers = [
{'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'},
]
proxies={
    'http':'socks5://127.0.0.1:9050',
    'https':'socks5://127.0.0.1:9050',
}

## URL 형식을 검증하는 함수 
def is_valid_url(url):
    
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

## 페이지 내용을 가져오는 함수, 세션 객체를 사용
def fetch_page(url):
    
    try:
        response = session.get(url, headers=headers[0], proxies=proxies, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None
    

## 버킷에 업로드
def upload_to_bucket(blob_name, path_to_file, bucket_name):

    storage_client = storage.Client.from_service_account_json('strange-analog-405708-b529029aaf4c.json')
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    return blob.public_url

image_dir = 'images'
video_dir = 'video'

BASE_URL = input("url을 입력하세요: ")
ARTICLE_BASE_URL = "https://gall.dcinside.com"

# url에서 id 값 파싱
parsed_url = urlparse(BASE_URL)
query_string = parsed_url.query
query_params = parse_qs(query_string)
id_value = query_params.get('id', [''])[0]

## 파싱하는 함수
def fetch_and_parse_article(url):
    """ 게시물 내용을 가져오고 파싱하는 함수 """
    if not is_valid_url(url):
        logger.error(f"Invalid URL: {url}")
        return None

    content = fetch_page(url)
    if content is None:
        return None

    return BeautifulSoup(content, 'html.parser')

## 이미지 크롤링
def process_image_post(tr_item): 
    print('+'*30)
    print("이미지 크롤링")
    #print(tr_item)
    date_tag = tr_item.find(class_='gall_date')
    date = date_tag.get('title')
    date = date.split(' ')[0]
    print("게시 날짜: ",date)
    # 제목 추출
    title_tag = tr_item.find('a', href=True)
    title = title_tag.text



    print("제목: ", title)
    print("주소: ", title_tag['href'])

    # 이미지가 있는 게시물에 request
    arurl = ARTICLE_BASE_URL + title_tag['href']
    article_response = requests.get(arurl, headers=headers[0],proxies=proxies)
    print("url: ", article_response.url)
    article_id = (title_tag['href'].split('no=')[1]).split('&')[0]
    print("게시물 ID : ", article_id)

   # URL 검증 로직 추가
    article_url = ARTICLE_BASE_URL + title_tag['href']
    if not is_valid_url(article_url):
        logger.error(f"Invalid URL: {article_url}")
        return  # 함수 종료

    article_content = fetch_page(article_url)
    if article_content is None:
        return  # 함수 종료

    article_soup = BeautifulSoup(article_response.content, 'html.parser')
    # 게시물 부분의 태그
    article_contents = article_soup.find('div', class_='writing_view_box').find_all('div')
    # 아래 이미지 다운로드 받는 곳에서 시작
    image_download_contents = article_soup.find('div', class_='appending_file_box').find('ul').find_all('li')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    time.sleep(0.5)

    for i, li in enumerate(image_download_contents):
        img_tag = li.find('a', href=True)
        img_url = img_tag['href']
        print("url : "+img_url)
        file_ext = img_url.split('.')[-1]
        savename = os.path.join(image_dir, f"image_{article_id}_{i}." + file_ext)
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'), ('Referer', article_response.url)]
        request.install_opener(opener)
        request.urlretrieve(img_url, savename)
        # 버킷에 저장
        bucket_name = "capstone_image"
        upload_to_bucket(f"image_{article_id}_{i}." + file_ext, savename, bucket_name)

        # 빅쿼리에 저장
        client = bigquery.Client()
        table_id = "strange-analog-405708.CrawlingDB.image"
        bucket_path = f"gs://{bucket_name}/"
        file_path = bucket_path + f"image_{article_id}_{i}.{file_ext}"

        rows_to_insert = [
            {"img_file_name": f"image_{article_id}_{i}.{file_ext}", "post_url": article_response.url, "post_date": datetime.strptime(date, "%Y-%m-%d").isoformat(), "file_path": file_path},
        ]
        errors = client.insert_rows_json(table_id, rows_to_insert)
        if errors == []:
            print("DB에 행추가")
        else:
            print("에러발생: {}".format(errors))

        print("저장한 URL : "+img_url)

    #0.5초 동안 대기
    time.sleep(0.5)


## 비디오 크롤링
def process_video_post(tr_item):
    print('+'*12)
    print("동영상 크롤링")
    #print(tr_item)
    date_tag = tr_item.find(class_='gall_date')
    date = date_tag.get('title')
    date = date.split(' ')[0]
    print("게시 날짜: ",date)
    # 제목 추출
    title_tag = tr_item.find('a', href=True)
    title = title_tag.text
    print("제목: ", title)
    print("주소: ", title_tag['href'])
    # 이미지가 있는 게시물에 request
    arurl = ARTICLE_BASE_URL + title_tag['href']

         # URL 검증 로직 추가
    article_url = ARTICLE_BASE_URL + title_tag['href']
    if not is_valid_url(article_url):
        logger.error(f"Invalid URL: {article_url}")
        return  # 함수 종료

    article_content = fetch_page(article_url)
    if article_content is None:
        return  # 함수 종료

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    proxy = "socks5://127.0.0.1:9050"
    chrome_options.add_argument(f'--proxy-server={proxy}')
    chrome_options.add_experimental_option('prefs',  {
        "download.default_directory": os.path.abspath(video_dir), # 다운로드 경로 설정
        "download.prompt_for_download": False, # 다운로드 시 확인 대화 상자 끄기
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True # PDF 파일을 항상 외부에서 열기
    })
    #service = Service(executable_path=r'C:/Users/tkdwn/chromedriver-win64/chromedriver.exe')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.implicitly_wait(7)
    # 웹사이트 접속
    driver.get(arurl)
    # 모든 iframe 웹 요소를 찾습니다.
    iframes = driver.find_elements(By.TAG_NAME, 'iframe')
    for iframe in iframes:
        iframe_id = iframe.get_attribute('id')
        print(iframe_id)
        if 'movie' in iframe_id:
            # 'movie'가 포함된 id를 가진 iframe으로 전환
            driver.switch_to.frame(iframe)

            try:
                # iframe 내부에서 .btn_down 클래스 이름을 가진 웹 요소를 찾음
                download_button = driver.find_element(By.CLASS_NAME, 'btn_down')
                print(download_button)
                download_button.click()
            except NoSuchElementException:
                print("다운로드 버튼을 찾을 수 없습니다.")
                break

            before_download_files = os.listdir(os.path.abspath(video_dir))

            # 새로운 파일이 생성될 때까지 기다림
            while True:
                time.sleep(1)
                after_download_files = os.listdir(os.path.abspath(video_dir))
                new_files = [f for f in after_download_files if f not in before_download_files and not f.endswith('.crdownload')]
                if new_files:
                    break

            # 새로 생성된 파일의 경로를 가져옴
            file_path = os.path.join(os.path.abspath(video_dir), new_files[0])
            file_name = os.path.basename(file_path)

#                     # 버킷의 하위 폴더 이름을 지정
#                     subfolder_name = 'video'

#                     # blob 이름을 생성
#                     blob_name = os.path.join(subfolder_name, file_name)

            # 파일을 버킷에 업로드
            upload_to_bucket(file_name, file_path, 'capstone_video')

            # 빅쿼리에 저장
            bucket_name = "capstone_video"
            client = bigquery.Client()
            table_id = "strange-analog-405708.CrawlingDB.video"  
            bucket_path = f"gs://{bucket_name}/"
            file_path = bucket_path + file_name  

            rows_to_insert = [
                {"videofile_name": file_name, "post_url": driver.current_url, "post_date": datetime.strptime(date, "%Y-%m-%d").isoformat(), "file_path": file_path},
            ]

            errors = client.insert_rows_json(table_id, rows_to_insert)
            if errors == []:
                print("새로운 행이 추가되었습니다.")
            else:
                print("행 추가 중 에러 발생: {}".format(errors))


            break
            time.sleep(0.5)
    
def process_article_list(article_list):
    """ 게시물 목록을 처리하는 함수 """
    for tr_item in article_list:
        if tr_item.find('em', class_='icon_img icon_pic') is not None:
            process_image_post(tr_item)
        elif tr_item.find('em', class_='icon_img icon_movie') is not None:
            process_video_post(tr_item)

def main():
    for i in range(1, 2):
        page_url = f"{BASE_URL}&page={i}"  # 'id' 파라미터를 중복 추가하지 않음
        page_content = fetch_page(page_url)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            article_list = soup.find('tbody').find_all('tr')
            process_article_list(article_list)

if __name__ == "__main__":
    main()
