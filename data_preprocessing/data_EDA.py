import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/17. 빅콘테스트/data/JEJU_MCT_DATA_v2.csv", encoding='cp949')
df.shape

columns_mapping = {
    'MON_UE_CNT_RAT': 'mon', 'TUE_UE_CNT_RAT': 'tue', 'WED_UE_CNT_RAT': 'wed',
    'THU_UE_CNT_RAT': 'thu', 'FRI_UE_CNT_RAT': 'fri', 'SAT_UE_CNT_RAT': 'sat', 'SUN_UE_CNT_RAT': 'sun',
    'HR_5_11_UE_CNT_RAT': '5to11', 'HR_12_13_UE_CNT_RAT': '12to13', 'HR_14_17_UE_CNT_RAT': '14to17',
    'HR_18_22_UE_CNT_RAT': '18to22', 'HR_23_4_UE_CNT_RAT': '23to4',
    'LOCAL_UE_CNT_RAT': 'local',
    'RC_M12_MAL_CUS_CNT_RAT': 'man', 'RC_M12_FME_CUS_CNT_RAT': 'woman',
    'RC_M12_AGE_UND_20_CUS_CNT_RAT': 'age_und20', 'RC_M12_AGE_30_CUS_CNT_RAT': 'age30',
    'RC_M12_AGE_40_CUS_CNT_RAT': 'age40', 'RC_M12_AGE_50_CUS_CNT_RAT': 'age50',
    'RC_M12_AGE_OVR_60_CUS_CNT_RAT': 'age_over60',
    'MCT_NM': 'name',
    'UE_CNT_GRP': 'use_cnt',
    'UE_AMT_GRP': 'use_pay',
    'UE_AMT_PER_TRSN_GRP': 'pay_per_cnt'
}

remaining_columns = {col: col.lower() for col in df.columns if col not in columns_mapping}

columns_mapping.update(remaining_columns)

df = df.rename(columns=columns_mapping)

df.head()

df['mct_type'].value_counts()

df.info()

df['ym'].value_counts()

df['mct_type'].value_counts()

df['use_cnt'].value_counts()

df['pay_per_cnt'].value_counts()

df.describe()

## 수치데이터 시각화

# 요일별 판매비율 비교

week_columns = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

# 평균과 중앙값 계산
mean_values = df[week_columns].mean()
median_values = df[week_columns].median()

# 시각화 (평균과 중앙값 비교)
plt.figure(figsize=(10, 6))
plt.plot(mean_values.index, mean_values, marker='o', label='Mean', color='b')
plt.plot(median_values.index, median_values, marker='s', label='Median', color='g')
plt.title('Mean vs Median of Weekday Usage Rates')
plt.ylabel('Usage Rate')
plt.xlabel('Day of the Week')
plt.legend()
plt.grid(True)
plt.show()

# 시간대별 판매비율 비교

time_columns = ['5to11', '12to13', '14to17', '18to22', '23to4']

mean_time = df[time_columns].mean()
median_time = df[time_columns].median()

plt.figure(figsize=(10, 6))
plt.plot(mean_time.index, mean_time, marker='o', label='Mean', color='b')
plt.plot(median_time.index, median_time, marker='s', label='Median', color='g')
plt.title('Mean vs Median of Time-based Usage Rates')
plt.ylabel('Usage Rate')
plt.xlabel('Time of Day')
plt.legend()
plt.grid(True)
plt.show()

# 성별별 비율
gender_columns = ['man', 'woman']

mean_gender = df[gender_columns].mean()

plt.figure(figsize=(10, 6))
plt.bar(mean_gender.index, mean_gender, label='Mean', color='b', alpha=0.6)

plt.title('Mean of Gender-based Usage Rates (Bar Graph)')
plt.ylabel('Usage Rate')
plt.xlabel('Gender')
plt.legend()
plt.grid(True)
plt.show()

# 연령별 비율

age_columns = ['age_und20', 'age30', 'age40', 'age50', 'age_over60']

mean_age = df[age_columns].mean()

plt.figure(figsize=(10, 6))
plt.bar(mean_age.index, mean_age, label='Mean', color='g', alpha=0.6)

plt.title('Mean of Age-based Usage Rates')
plt.ylabel('Usage Rate')
plt.xlabel('Age Group')
plt.xticks(rotation=20)
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import LabelEncoder

# 범주형 데이터 레이블 인코딩

label_encoding_columns = ['ym', 'mct_type', 'use_cnt', 'use_pay', 'pay_per_cnt']

label_encoder = LabelEncoder()

for col in label_encoding_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 범주형 데이터 시각화

categorical_columns = ['use_cnt', 'use_pay', 'pay_per_cnt', 'mct_type']

for col in categorical_columns:
    plt.figure(figsize=(8, 6))

    sns.countplot(data=df, x=col)

    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
