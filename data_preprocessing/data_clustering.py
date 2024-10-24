import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np

df = pd.read_csv("jeju_data_location.csv")

df.head()

# 1. 기존 컬럼명 변경
column_mapping = {
    'YM': '기준연월',
    'MCT_NM': '식당이름',
    'OP_YMD': '오픈일자',
    'MCT_TYPE': '업종',
    'ADDR': '주소',
    'UE_CNT_GRP': '이용건수구간',
    'UE_AMT_GRP': '이용금액구간',
    'UE_AMT_PER_TRSN_GRP': '건당평균이용금액구간',
    'MON_UE_CNT_RAT': '월요일이용비중',
    'TUE_UE_CNT_RAT': '화요일이용비중',
    'WED_UE_CNT_RAT': '수요일이용비중',
    'THU_UE_CNT_RAT': '목요일이용비중',
    'FRI_UE_CNT_RAT': '금요일이용비중',
    'SAT_UE_CNT_RAT': '토요일이용비중',
    'SUN_UE_CNT_RAT': '일요일이용비중',
    'HR_5_11_UE_CNT_RAT': '5시~11시이용비중',
    'HR_12_13_UE_CNT_RAT': '12시~13시이용비중',
    'HR_14_17_UE_CNT_RAT': '14시~17시이용비중',
    'HR_18_22_UE_CNT_RAT': '18시~22시이용비중',
    'HR_23_4_UE_CNT_RAT': '23시~4시이용비중',
    'LOCAL_UE_CNT_RAT': '현지인이용비중',
    'RC_M12_MAL_CUS_CNT_RAT': '남성회원비중',
    'RC_M12_FME_CUS_CNT_RAT': '여성회원비중',
    'RC_M12_AGE_UND_20_CUS_CNT_RAT': '20대이하회원비중',
    'RC_M12_AGE_30_CUS_CNT_RAT': '30대회원비중',
    'RC_M12_AGE_40_CUS_CNT_RAT': '40대회원비중',
    'RC_M12_AGE_50_CUS_CNT_RAT': '50대회원비중',
    'RC_M12_AGE_OVR_60_CUS_CNT_RAT': '60대이상회원비중'
}

# 1. 컬럼명 변경
df.rename(columns=column_mapping, inplace=True)

# 2. MCT_NM을 기준으로 그룹화하여 필요한 값으로 통합
df = df.groupby('식당이름').agg({
    '기준연월': 'max',  # 기준연월은 가장 최신 값으로 선택
    '업종': 'first',  # 업종은 첫 번째 값으로 선택
    '주소': 'first',  # 주소는 첫 번째 값으로 선택
    '이용건수구간': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],  # 최빈값 또는 첫 번째 값
    '이용금액구간': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],  # 최빈값 또는 첫 번째 값
    '건당평균이용금액구간': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
    '월요일이용비중': 'mean', '화요일이용비중': 'mean', '수요일이용비중': 'mean',
    '목요일이용비중': 'mean', '금요일이용비중': 'mean', '토요일이용비중': 'mean',
    '일요일이용비중': 'mean', '5시~11시이용비중': 'mean', '12시~13시이용비중': 'mean',
    '14시~17시이용비중': 'mean', '18시~22시이용비중': 'mean', '23시~4시이용비중': 'mean',
    '현지인이용비중': 'mean', '남성회원비중': 'mean', '여성회원비중': 'mean',
    '20대이하회원비중': 'mean', '30대회원비중': 'mean', '40대회원비중': 'mean',
    '50대회원비중': 'mean', '60대이상회원비중': 'mean', '위도': 'first', '경도': 'first'
}).reset_index()

# 3. Label Encoding
le = LabelEncoder()
df['이용건수구간'] = le.fit_transform(df['이용건수구간'])
df['이용금액구간'] = le.fit_transform(df['이용금액구간'])
df['건당평균이용금액구간'] = le.fit_transform(df['건당평균이용금액구간'])
df['업종'] = le.fit_transform(df['업종'])  # 업종도 레이블 인코딩

# 4. 스케일링할 컬럼 정의
scaling_cols = [
    '월요일이용비중', '화요일이용비중', '수요일이용비중', '목요일이용비중',
    '금요일이용비중', '토요일이용비중', '일요일이용비중', '5시~11시이용비중',
    '12시~13시이용비중', '14시~17시이용비중', '18시~22시이용비중', '23시~4시이용비중',
    '남성회원비중', '여성회원비중', '20대이하회원비중',
    '30대회원비중', '40대회원비중', '50대회원비중', '60대이상회원비중', '위도', '경도'
]

# 5. 가중치 부여 (가중치 적용)
weights = {
    '이용건수구간': 2.0,  # 중요도 높은 항목에 가중치 부여
    '이용금액구간': 1.5,
    '건당평균이용금액구간': 1.5,
    '업종': 2.0
}

for col in scaling_cols + ['이용건수구간', '이용금액구간', '건당평균이용금액구간', '업종']:
    df[col] = df[col] * weights.get(col, 1.0)

# 6. StandardScaler를 사용한 스케일링
scaler = StandardScaler()
df[scaling_cols] = scaler.fit_transform(df[scaling_cols])

# 7. KMeans 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[scaling_cols])

'''# 7. DBSCAN 적용
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = dbscan.fit_predict(df[scaling_cols])'''

# 클러스터별로 그룹화하여 각 컬럼의 통계값 추출
cluster_summary = df.groupby('Cluster').agg({
    '월요일이용비중': ['mean', 'std'],
    '화요일이용비중': ['mean', 'std'],
    '수요일이용비중': ['mean', 'std'],
    '목요일이용비중': ['mean', 'std'],
    '금요일이용비중': ['mean', 'std'],
    '토요일이용비중': ['mean', 'std'],
    '일요일이용비중': ['mean', 'std'],
    '5시~11시이용비중': ['mean', 'std'],
    '12시~13시이용비중': ['mean', 'std'],
    '14시~17시이용비중': ['mean', 'std'],
    '18시~22시이용비중': ['mean', 'std'],
    '23시~4시이용비중': ['mean', 'std'],
    '남성회원비중': ['mean', 'std'],
    '여성회원비중': ['mean', 'std'],
    '20대이하회원비중': ['mean', 'std'],
    '30대회원비중': ['mean', 'std'],
    '40대회원비중': ['mean', 'std'],
    '50대회원비중': ['mean', 'std'],
    '60대이상회원비중': ['mean', 'std'],
    '이용건수구간': ['mean', 'std'],
    '이용금액구간': ['mean', 'std'],
    '건당평균이용금액구간': ['mean', 'std'],
    '위도': ['mean', 'std'],
    '경도': ['mean', 'std'],
})

# 클러스터별 통계 결과 출력
print(cluster_summary)

# PCA로 KMeans 클러스터 시각화 (2D)
pca_kmeans_data = pca.fit_transform(df[scaling_cols])

plt.figure(figsize=(8, 6))
plt.scatter(pca_kmeans_data[:, 0], pca_kmeans_data[:, 1], c=df['Cluster'], cmap='viridis', marker='o', s=50)
plt.title("KMeans Clustering (k=4) Visualization using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='KMeans Cluster')
plt.show()

# 각 클러스터별 주요 특징 요약
for cluster in df['Cluster'].unique():
    print(f"\nKMeans Cluster {cluster} 주요 특징:")
    cluster_data = df[df['Cluster'] == cluster]

    # 주요 특징 추출 (예: 평균값)
    print("이용금액구간 평균:", cluster_data['이용금액구간'].mean())
    print("건당평균이용금액구간 평균:", cluster_data['건당평균이용금액구간'].mean())
    print("남성회원비중 평균:", cluster_data['남성회원비중'].mean())
    print("여성회원비중 평균:", cluster_data['여성회원비중'].mean())
    print("위도 평균:", cluster_data['위도'].mean())
    print("경도 평균:", cluster_data['경도'].mean())

# 클러스터별로 그룹화하여 요일별 이용비중 평균 추출
weekday_visit_summary = df.groupby('Cluster').agg({
    '월요일이용비중': 'mean',
    '화요일이용비중': 'mean',
    '수요일이용비중': 'mean',
    '목요일이용비중': 'mean',
    '금요일이용비중': 'mean',
    '토요일이용비중': 'mean',
    '일요일이용비중': 'mean'
}).reset_index()

# 클러스터별 요일별 방문수 출력
print(weekday_visit_summary)

# 클러스터별로 그룹화하여 시간대별 통계값 추출
time_summary = df.groupby('Cluster').agg({
    '5시~11시이용비중': ['mean', 'std'],
    '12시~13시이용비중': ['mean', 'std'],
    '14시~17시이용비중': ['mean', 'std'],
    '18시~22시이용비중': ['mean', 'std'],
    '23시~4시이용비중': ['mean', 'std'],
})

# 클러스터별 시간대 통계 결과 출력
print(time_summary)

# 각 클러스터별 주요 시간대 특징 요약
for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster} 주요 시간대 특징:")
    cluster_data = df[df['Cluster'] == cluster]

    # 주요 시간대별 특징 추출 (예: 평균값)
    print("5시~11시 이용 비중 평균:", cluster_data['5시~11시이용비중'].mean())
    print("12시~13시 이용 비중 평균:", cluster_data['12시~13시이용비중'].mean())
    print("14시~17시 이용 비중 평균:", cluster_data['14시~17시이용비중'].mean())
    print("18시~22시 이용 비중 평균:", cluster_data['18시~22시이용비중'].mean())
    print("23시~4시 이용 비중 평균:", cluster_data['23시~4시이용비중'].mean())

df_cluster = df.copy()

# 원본 데이터 다시 불러오기
df = pd.read_csv('/content/drive/MyDrive/17. 빅콘테스트/data/jeju_data_lat_lng.csv')  # 실제 파일 경로로 변경

# 복사된 클러스터링 결과 데이터
df_cluster = df_cluster[['식당이름', 'Cluster']]  # '식당이름'과 'Cluster' 컬럼만 남김

# 'MCT_NM'을 기준으로 원본 데이터와 클러스터링 결과를 병합
df = pd.merge(df, df_cluster, left_on='MCT_NM', right_on='식당이름', how='left')

# 결과 확인
df.head()

## Cluster별 주요 특징 요약

### Cluster 0:
**주요 금액 관련 특징:**
- 이용금액구간 평균: 4.0756 (상위 구간에 속함)
- 건당평균이용금액구간 평균: 5.1558 (상위 구간에 속함)

**성별 비중:**
- 남성회원비중 평균: -0.8826
- 여성회원비중 평균: 0.8826 (여성회원 비중이 높음)

**위치 정보:**
- 위도 평균: 0.0177
- 경도 평균: 0.0215

**시간대 특징:**
- 5시~11시, 12시~13시, 14시~17시 시간대에 방문 비중이 높고, 특히 14시~17시에 매우 높음.
- 18시~22시, 23시~4시 시간대에는 방문 비중이 낮음.

---

### Cluster 1:
**주요 금액 관련 특징:**
- 이용금액구간 평균: 4.4367 (상위 구간에 속함)
- 건당평균이용금액구간 평균: 2.2048 (중간 구간에 속함)

**성별 비중:**
- 남성회원비중 평균: 0.5161
- 여성회원비중 평균: -0.5161 (남성회원 비중이 높음)

**위치 정보:**
- 위도 평균: -0.0266
- 경도 평균: -0.0176

**시간대 특징:**
- 18시~22시 시간대에 방문 비중이 매우 높고, 23시~4시에도 일부 방문이 있음.
- 5시~17시 시간대에는 방문 비중이 매우 낮음.

---

### Cluster 2:
**주요 금액 관련 특징:**
- 이용금액구간 평균: 3.8612 (중간 구간에 속함)
- 건당평균이용금액구간 평균: 2.5515 (중간 구간에 속함)

**성별 비중:**
- 남성회원비중 평균: 0.1338
- 여성회원비중 평균: -0.1338 (남성회원 비중이 조금 높음)

**위치 정보:**
- 위도 평균: 0.2110
- 경도 평균: 0.0131

**시간대 특징:**
- 23시~4시 시간대에 방문 비중이 매우 높고, 18시~22시에도 다소 방문이 있음.
- 5시~17시 시간대에는 방문 비중이 매우 낮음, 특히 12시~13시, 14시~17시에 매우 낮음.

---

### Cluster 3:
**주요 금액 관련 특징:**
- 이용금액구간 평균: 4.4944 (상위 구간에 속함)
- 건당평균이용금액구간 평균: 3.7153 (상위 구간에 속함)

**성별 비중:**
- 남성회원비중 평균: 0.4632
- 여성회원비중 평균: -0.4632 (남성회원 비중이 조금 높음)

**위치 정보:**
- 위도 평균: -0.0951
- 경도 평균: -0.0137

**시간대 특징:**
- 5시~13시 시간대에 방문 비중이 높음, 특히 12시~13시에 매우 높음.
- 14시~4시 시간대에는 방문 비중이 낮음, 특히 23시~4시 시간대는 매우 낮음.

---

## 종합 분석:
- **Cluster 0**: 주로 이른 시간대(5시~17시) 방문이 많고, 특히 14시~17시에 집중된 고객층. 여성 비중이 높으며, 높은 이용 금액을 보임.
- **Cluster 1**: 저녁(18시~22시) 방문이 두드러지며, 남성 비중이 높음. 상위 구간의 이용 금액과 평균 이용 금액을 보임.
- **Cluster 2**: 주로 야간(23시~4시)에 방문하는 고객층. 비교적 낮은 시간대 이용 비중을 보임.
- **Cluster 3**: 아침점심 시간대(5시~13시) 방문이 집중된 고객층. 특히 12시~13시 방문이 많으며, 남성 비중이 높음.






# Cluster 별 특징 정의
cluster_features = {
    0: "이용금액 높음, 여성비율 높음, 오후시간대 인기많음",
    1: "이용금액 높음, 남성비율 높음, 저녁시간대 인기많음",
    2: "이용금액 중간, 남성비율 약간 높음, 야간시간대 인기많음",
    3: "이용금액 높음, 남성비율 약간 높음, 아침~점심시간대 인기많음"
}

# Cluster 별 추천 대상 정의
cluster_recommendations = {
    0: "중고가 식당 선호 여성 관광객, 오후 방문자",
    1: "저녁시간대 선호 남성 관광객, 고급식당 선호자",
    2: "야간 활동 남성 관광객, 중간 가격대 선호자",
    3: "아침~점심 선호 남성 관광객, 고급식당 선호자"
}
column_mapping = {
    'YM': '기준연월',
    'MCT_NM': '식당이름',
    'OP_YMD': '오픈일자',
    'MCT_TYPE': '업종',
    'ADDR': '주소',
    'UE_CNT_GRP': '이용건수구간',
    'UE_AMT_GRP': '이용금액구간',
    'UE_AMT_PER_TRSN_GRP': '건당평균이용금액구간',
    'MON_UE_CNT_RAT': '월요일이용비중',
    'TUE_UE_CNT_RAT': '화요일이용비중',
    'WED_UE_CNT_RAT': '수요일이용비중',
    'THU_UE_CNT_RAT': '목요일이용비중',
    'FRI_UE_CNT_RAT': '금요일이용비중',
    'SAT_UE_CNT_RAT': '토요일이용비중',
    'SUN_UE_CNT_RAT': '일요일이용비중',
    'HR_5_11_UE_CNT_RAT': '5시~11시이용비중',
    'HR_12_13_UE_CNT_RAT': '12시~13시이용비중',
    'HR_14_17_UE_CNT_RAT': '14시~17시이용비중',
    'HR_18_22_UE_CNT_RAT': '18시~22시이용비중',
    'HR_23_4_UE_CNT_RAT': '23시~4시이용비중',
    'LOCAL_UE_CNT_RAT': '현지인이용비중',
    'RC_M12_MAL_CUS_CNT_RAT': '남성회원비중',
    'RC_M12_FME_CUS_CNT_RAT': '여성회원비중',
    'RC_M12_AGE_UND_20_CUS_CNT_RAT': '20대이하회원비중',
    'RC_M12_AGE_30_CUS_CNT_RAT': '30대회원비중',
    'RC_M12_AGE_40_CUS_CNT_RAT': '40대회원비중',
    'RC_M12_AGE_50_CUS_CNT_RAT': '50대회원비중',
    'RC_M12_AGE_OVR_60_CUS_CNT_RAT': '60대이상회원비중',
    'Cluster': '클러스터'
}

# 1. 컬럼명 변경
df.rename(columns=column_mapping, inplace=True)


# 1. 클러스터 별 특징 컬럼 추가
df['식당특징'] = df['클러스터'].map(cluster_features)

# 2. 클러스터 별 추천 대상 컬럼 추가
df['추천대상'] = df['클러스터'].map(cluster_recommendations)

# 중복된 '식당이름' 컬럼 제거
df = df.loc[:, ~df.columns.duplicated()]  # 중복된 컬럼을 제거하는 코드

# 결과 확인
df.head()


df.to_csv('jeju_data_clustered.csv', index=False)

df.head()
