from dotenv import load_dotenv
import streamlit as st
import os
import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as genai
import faiss
import math
from datetime import time
from datetime import datetime

# 가장 먼저 set_page_config 호출
st.set_page_config(page_title="🥙🌮🥯제주 맛집 찾아 삼만리🥯🌮🥙")

load_dotenv()

genai_key = os.getenv('GENAI_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')
GEOCODING_API_URL = 'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode'
module_path = 'data/'

device = "cuda" if torch.cuda.is_available() else "cpu"
# gemini-1.5-flash 로드
genai.configure(api_key=genai_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

df = pd.read_csv('data/jeju_data_final.csv')
embeddings = np.load(os.path.join('data/embeddings_array_file.npy'))
index_path = 'data/faiss_index.index'
image_path = 'https://github.com/PeterSong97/bigcon/raw/main/data/%ED%83%80%EC%9D%B4%ED%8B%80%EC%9D%B4%EB%AF%B8%EC%A7%80.png'

# 두 좌표 사이의 거리를 계산하는 Haversine 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km 단위)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # 두 지점 사이의 거리 (km)

# 위치 정보에 따라 데이터를 필터링하는 함수
def filter_restaurants_by_distance(df, user_lat, user_lon, max_distance_km=8):
    df_filtered = df.copy()
    df_filtered['distance'] = df.apply(lambda row: haversine(user_lat, user_lon, row['위도'], row['경도']), axis=1)
    df_filtered = df_filtered[df_filtered['distance'] <= max_distance_km]
    return df_filtered

# 주소를 위도/경도로 변환하는 함수 (네이버 API 사용)
def get_lat_lng_from_address(address):
    headers = {
        'X-NCP-APIGW-API-KEY-ID': NAVER_CLIENT_ID,
        'X-NCP-APIGW-API-KEY': NAVER_CLIENT_SECRET
    }
    params = {
        'query': address
    }
    
    response = requests.get(GEOCODING_API_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('addresses')
        if results:
            location = results[0]
            return float(location['y']), float(location['x'])  # 위도(y), 경도(x)
    
    return None, None

#=============================================필요한 모듈호출, 함수선언 완료====================================================

# Streamlit 앱 설정
st.markdown(
    f"""
    <style>
    .centered-image {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }}
    .centered-title {{
        text-align: center;
        font-size: 35px;
    }}
    .centered-subheader {{
        text-align: center;
        font-size: 18px;
    }}
    </style>
    <img src="{image_path}" class="centered-image">
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered-title">🥙🌮🥯 제주 맛집 찾아 삼만리🥯🌮🥙</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="centered-subheader">제주도 맛집 찾아 오셨어요??</h2>', unsafe_allow_html=True)

# 사이드바에서 '현재 위치에서 가까운 식당 추천' 여부 선택
st.sidebar.header("현재 위치에서 가까운 식당 추천")
use_current_location = st.sidebar.radio("현재 위치를 기반으로 추천할까요?", ('Yes', 'No'))

# 유저 위치 정보를 바탕으로 필터링
user_address = False
if use_current_location == 'Yes':
    st.sidebar.header("현재 위치 주소 입력")
    user_address = st.sidebar.text_input("주소를 입력하세요")

    if user_address:
        latitude, longitude = get_lat_lng_from_address(user_address)
        if latitude and longitude:
            st.sidebar.success(f"위도: {latitude}, 경도: {longitude}")

            # 거리 필터링 옵션 추가
            use_distance_filter = st.sidebar.checkbox("거리를 기준으로 필터링하시겠습니까?", value=True)
            if use_distance_filter:
                max_distance_km = st.sidebar.slider("거리 제한을 설정하세요 (km)", min_value=1, max_value=20, value=5)
                df_filtered = filter_restaurants_by_distance(df, latitude, longitude, max_distance_km=max_distance_km)  # 필터링된 데이터프레임
                st.sidebar.write(f"{max_distance_km} km 이내의 식당을 찾습니다.")
            else:
                df_filtered = df.copy()  # 거리 필터링 사용 안함
        else:
            st.sidebar.error("주소를 찾을 수 없습니다. 다시 입력해주세요.")
            df_filtered = df.copy()  # 에러 시 전체 데이터를 사용
    else:
        latitude, longitude = None, None
        df_filtered = df.copy()  # 전체 데이터를 사용
else:
    latitude, longitude = None, None
    df_filtered = df.copy()

# 사이드바에서 '현지인 맛집' 또는 '관광객 맛집' 여부 선택
st.sidebar.header("현지인 맛집 또는 관광객 맛집 추천")
local_choice = st.sidebar.radio("어떤 맛집을 찾으시나요?", ('제주도민 맛집', '관광객 맛집', '상관없음'))

# 방문 예정 시간 선택
st.sidebar.header("방문 예정 시간대 선택")
visit_time = st.sidebar.time_input('방문할 시간을 선택하세요', value=time(12, 0))

# 선택된 시간을 출력
visit_time = visit_time.strftime("%H:%M")
st.sidebar.write(f"선택한 방문 예정 시간대: {visit_time}")

# 요일 선택 UI - 스크롤바 형식
st.sidebar.header("방문 요일 선택")
day_of_week_list = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
selected_day = st.sidebar.selectbox("방문할 요일을 선택하세요", day_of_week_list, index=datetime.now().weekday())
visit_day = day_of_week_list.index(selected_day)
st.sidebar.write(f"선택한 방문 요일: {selected_day}")

# ====================================================== streamlit UI 지정완료 =============================================================

# 요일 필터링 함수
def filter_by_visit_day(df_filtered, visit_day):
    day_of_week = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    visit_day_str = day_of_week[visit_day]
    df_filtered = df_filtered[df_filtered['영업요일'].str.contains(visit_day_str)]
    if df_filtered.empty:
        return "선택한 요일에 오픈하는 식당이 없습니다."
    return df_filtered

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

def filter_by_visit_time(df_filtered, visit_time):
    if isinstance(visit_time, str):
        visit_time = datetime.strptime(visit_time, '%H:%M')

    visit_hour = visit_time.hour

    if 5 <= visit_hour < 12:
        df_filtered = df_filtered[(df_filtered['5시~11시이용비중'] > 0)].reset_index(drop=True)
    elif 12 <= visit_hour < 14:
        df_filtered = df_filtered[(df_filtered['12시~13시이용비중'] > 0)].reset_index(drop=True)
    elif 14 <= visit_hour < 18:
        df_filtered = df_filtered[(df_filtered['14시~17시이용비중'] > 0)].reset_index(drop=True)
    elif 18 <= visit_hour < 23:
        df_filtered = df_filtered[(df_filtered['18시~22시이용비중'] > 0)].reset_index(drop=True)
    else:
        df_filtered = df_filtered[(df_filtered['23시~4시이용비중'] > 0)].reset_index(drop=True)

    if df_filtered.empty:
        return "선택한 시간대에 이용 가능한 식당이 없습니다."

    return df_filtered

# 전체 필터링 함수
def filter_restaurants(df, visit_time, visit_day, user_lat=None, user_lon=None, local_choice=None, max_distance_km=5):
    df_filtered = filter_by_visit_time(df, visit_time)
    df_filtered = filter_by_visit_day(df_filtered, visit_day)
    if user_lat is not None and user_lon is not None:
        df_filtered = filter_restaurants_by_distance(df_filtered, user_lat, user_lon, max_distance_km=max_distance_km)
    if local_choice == '제주도민 맛집':
        df_filtered = df_filtered[df_filtered['현지인이용비중'] > 0.5]
    elif local_choice == '관광객 맛집':
        df_filtered = df_filtered[df_filtered['현지인이용비중'] < 0.5]
    return df_filtered

# ====================================================== 필요함수 선언완료 =============================================================

def generate_response_with_faiss(question, df, embeddings, model, embed_text, visit_time, visit_day, local_choice, user_lat=None, user_lon=None, max_distance_km=5, index_path=None, max_count=10, k=3, print_prompt=True):
    additional_info = f" 방문 예정 시간은 {visit_time}, 방문 예정 요일은 {visit_day}입니다."

    if user_address:
        additional_info = f" 위치는 {user_address} 입니다."
    
    # 질문에 추가 정보를 결합하여 임베딩에 사용
    full_question = f"{question} {additional_info}"

    index = load_faiss_index(index_path)

    query_embedding = embed_text(full_question).reshape(1, -1)

    distances, indices = index.search(query_embedding, k * 3)

    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return "검색된 결과가 없습니다."
    
    # 유효한 인덱스만 필터링 (데이터프레임 범위를 넘는 인덱스를 제외)
    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return "검색된 결과가 없습니다."

    # 필터링을 진행 (시간, 요일, 거리, 현지인/관광객 옵션)
    filtered_df = df.iloc[valid_indices].copy().reset_index(drop=True)
    filtered_df = filter_restaurants(df=filtered_df, visit_time=visit_time, visit_day=visit_day, user_lat=user_lat, user_lon=user_lon, local_choice=local_choice, max_distance_km=max_distance_km)

    if isinstance(filtered_df, str):  # 필터링 결과가 메시지라면 반환
        return filtered_df

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    # 참고할 정보와 프롬프트 구성
    reference_info = ""
    for idx, row in filtered_df.iterrows():
        reference_info += f"{row['text']}\n"

    # 응답을 받아오기 위한 프롬프트 생성
    prompt = f"""질문: {question}
            {local_choice}을 반영해줘.

            참고할 정보:
            {reference_info} 의 업종을 참고하면 좋을거같아.

            응답 형식:
            1. 추천 식당: 식당명
            2. 추천 이유: 왜 이 식당을 추천하는지 설명해주세요.
            3. 주소: 이 식당의 주소를 알려주세요.
            """

    # 응답 생성
    try:
        response = model.generate_content(prompt)

        # 응답의 텍스트 추출
        if hasattr(response, 'candidates'):
            candidate = response.candidates[0]
            if candidate.content.parts:
                full_response = candidate.content.parts[0].text
            else:
                full_response = "조건에 맞는 식당이 없습니다!"
        else:
            full_response = "응답을 생성하지 못했습니다."
    except Exception as e:
        full_response = f"응답 생성 중 오류가 발생했습니다: {str(e)}"

    return full_response


# 채팅 대화 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 제주 맛집을 찾고 계신가요?"}]

# 기존 대화 표시
for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.markdown(f"**🍊맛집박사:** {message['content']}")  # 어시스턴트 메시지
    else:
        st.markdown(f"**🙋쩝쩝박사:** {message['content']}")  # 유저 메시지

# 유저 입력 받기 (채팅창)
prompt = st.chat_input("어떤 음식을 드시고싶으신가요?")

# 유저가 질문을 입력했을 때만 실행
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 유저 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("찾아보고 있어요!"):
            response = generate_response_with_faiss(
                prompt, df, embeddings, model, embed_text,
                visit_time=visit_time, visit_day=visit_day,
                local_choice=local_choice, user_lat=latitude, user_lon=longitude,
                max_distance_km=5, index_path='data/faiss_index.index'
            )
            # 어시스턴트 응답 표시
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

# Clear chat history 버튼
def clear_chat_history():
    st.session_state.messages = [{"role": "ヾ(•ω•`)o", "content": "어떤 식당을 찾고계세요?"}]
st.sidebar.button('※채팅초기화※', on_click=clear_chat_history)
