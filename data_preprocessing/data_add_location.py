import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import numpy as np
import nest_asyncio
import asyncio

df = pd.read_csv("/content/drive/MyDrive/17. 빅콘테스트/data/JEJU_MCT_DATA_v2.csv", encoding='cp949')
df.head()

# 이벤트 루프 중첩 문제 해결
nest_asyncio.apply()

# API 호출 함수 (기존 코드 재사용)
async def get_lat_lng(session, address):
    url = 'https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode'
    headers = {
        "X-NCP-APIGW-API-KEY-ID": 'API_ID',  # 자신의 API KEY로 변경
        "X-NCP-APIGW-API-KEY": 'API_KEY'
    }
    params = {"query": address}

    try:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                result = await response.json()
                if result['addresses']:
                    lat = result['addresses'][0]['y']  # 위도
                    lng = result['addresses'][0]['x']  # 경도
                    return address, lat, lng
                else:
                    return address, np.nan, np.nan  # 주소가 없을 경우 NaN 반환
            else:
                print(f"Error {response.status}: {response.text}")
                return address, np.nan, np.nan
    except Exception as e:
        print(f"Exception occurred: {e}")
        return address, np.nan, np.nan

async def process_addresses_in_parallel(unique_addresses):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [get_lat_lng(session, addr) for addr in unique_addresses]
        results = await asyncio.gather(*tasks)
    return results


# 1. 'ADDR' 컬럼에서 중복되지 않은 주소 추출
unique_addresses = df['ADDR'].drop_duplicates().tolist()

# 2. API 호출하여 중복되지 않은 주소들의 위/경도 값 가져오기 (비동기 처리)
results = asyncio.run(process_addresses_in_parallel(unique_addresses))

# 3. API 결과를 데이터프레임으로 변환
lat_lng_df = pd.DataFrame(results, columns=['ADDR', '위도', '경도'])

# 4. 원래 데이터프레임과 API 결과를 주소(ADDR)로 병합 (위/경도 값을 채우기)
df = pd.merge(df, lat_lng_df, on='ADDR', how='left')

df.to_csv("jeju_data_location.csv", index=False)
