import pandas as pd
df = pd.read_csv("jeju_data_clustered.csv")

df.info()

# 기존 클러스터링 컬럼 제거
df = df.drop(['클러스터', '식당특징', '추천대상'], axis=1)

# 2단계: '식당이름' 컬럼을 기준으로 데이터 통합
# 최빈값으로 통합할 컬럼: '이용건수구간', '이용금액구간', '건당평균이용금액구간'
# 평균값으로 통합할 컬럼: '월요일이용비중', '화요일이용비중', '수요일이용비중', '목요일이용비중', '금요일이용비중', '토요일이용비중', '일요일이용비중',
#                        '5시~11시이용비중', '12시~13시이용비중', '14시~17시이용비중', '18시~22시이용비중', '23시~4시이용비중',
#                        '현지인이용비중', '남성회원비중', '여성회원비중', '20대이하회원비중', '30대회원비중', '40대회원비중', '50대회원비중', '60대이상회원비중'

# 1. 필요한 라이브러리 불러오기
import pandas as pd
from scipy import stats

def get_mode(series):
    return series.mode()[0]

df = df.groupby('식당이름').agg({
    '오픈일자': 'first',
    '업종': 'first',
    '주소': 'first',
    '이용건수구간': get_mode,
    '이용금액구간': get_mode,
    '건당평균이용금액구간': get_mode,
    '월요일이용비중': 'mean',
    '화요일이용비중': 'mean',
    '수요일이용비중': 'mean',
    '목요일이용비중': 'mean',
    '금요일이용비중': 'mean',
    '토요일이용비중': 'mean',
    '일요일이용비중': 'mean',
    '5시~11시이용비중': 'mean',
    '12시~13시이용비중': 'mean',
    '14시~17시이용비중': 'mean',
    '18시~22시이용비중': 'mean',
    '23시~4시이용비중': 'mean',
    '현지인이용비중': 'mean',
    '남성회원비중': 'mean',
    '여성회원비중': 'mean',
    '20대이하회원비중': 'mean',
    '30대회원비중': 'mean',
    '40대회원비중': 'mean',
    '50대회원비중': 'mean',
    '60대이상회원비중': 'mean',
    '위도': 'first',
    '경도': 'first'
}).reset_index()

df.head()

# 3단계: 시간대별 이용비중을 참고하여 영업시간 컬럼 생성
def generate_business_hours(row):
    hours = []
    # 시간대별 이용비중을 참고하여 영업시간 리스트를 만듦
    if row['5시~11시이용비중'] > 0:
        hours.extend(range(5, 12))  # 5시부터 11시까지
    if row['12시~13시이용비중'] > 0:
        hours.extend(range(12, 14))  # 12시부터 13시까지
    if row['14시~17시이용비중'] > 0:
        hours.extend(range(14, 18))  # 14시부터 17시까지
    if row['18시~22시이용비중'] > 0:
        hours.extend(range(18, 23))  # 18시부터 22시까지
    if row['23시~4시이용비중'] > 0:
        hours.extend(list(range(23, 25)) + list(range(1, 5)))  # 23시부터 4시까지

    return hours  # 중복 제거 및 정렬하지 않음

# 영업시간 컬럼 생성
df['영업시간'] = df.apply(generate_business_hours, axis=1)

# 3단계 완료된 데이터 확인 (상위 5개 데이터)
df[['식당이름', '영업시간']].head()

# 4단계: 요일별 이용비중을 참고하여 영업요일 컬럼 생성
def generate_business_days(row):
    days = []
    # 요일별 이용비중을 참고하여 영업요일 리스트를 만듦
    if row['월요일이용비중'] > 0:
        days.append('월요일')
    if row['화요일이용비중'] > 0:
        days.append('화요일')
    if row['수요일이용비중'] > 0:
        days.append('수요일')
    if row['목요일이용비중'] > 0:
        days.append('목요일')
    if row['금요일이용비중'] > 0:
        days.append('금요일')
    if row['토요일이용비중'] > 0:
        days.append('토요일')
    if row['일요일이용비중'] > 0:
        days.append('일요일')

    return days  # 중복 제거 및 정렬하지 않음

# 영업요일 컬럼 생성
df['영업요일'] = df.apply(generate_business_days, axis=1)

# 4단계 완료된 데이터 확인 (상위 5개 데이터)
df[['식당이름', '영업요일']].head()

df.info()

def generate_text(row):
    text = f"{row['식당이름']}의 업종은 {row['업종']}이고, 위치는 {row['주소']}입니다. "

    # 이용건수와 이용금액구간에서 숫자와 언더스코어 (_) 제거
    usage_range = row['이용건수구간'].split('_')[-1]  # '_' 이후 값만 추출
    amount_range = row['이용금액구간'].split('_')[-1]  # '_' 이후 값만 추출

    text += f"이용 건수는 동일 업종 내 상위 {usage_range}이고, "
    text += f"이용 금액 구간은 {amount_range}입니다. "

    # 시간대별 이용 비중에서 가장 높은 비중을 가진 시간대를 찾음
    time_columns = ['5시~11시이용비중', '12시~13시이용비중', '14시~17시이용비중', '18시~22시이용비중', '23시~4시이용비중']
    max_time_column = row[time_columns].idxmax()  # 가장 높은 비중을 가진 컬럼명
    time_period = max_time_column.replace("이용비중", "")  # "이용비중" 제거
    text += f"가장 이용이 많은 시간대는 {time_period}입니다. "

    # 요일별 이용 비중에서 가장 높은 비중을 가진 요일을 찾음
    day_columns = ['월요일이용비중', '화요일이용비중', '수요일이용비중', '목요일이용비중', '금요일이용비중', '토요일이용비중', '일요일이용비중']
    max_day_column = row[day_columns].idxmax()  # 가장 높은 비중을 가진 컬럼명
    day = max_day_column.replace("이용비중", "")  # "이용비중" 제거
    text += f"가장 이용이 많은 요일은 {day}입니다. "

    # 현지인 및 회원 비중 정보 추가
    text += f"현지인 이용 비중은 {row['현지인이용비중']:.2f}이고, 남성 회원 비중은 {row['남성회원비중']:.2f}, 여성 회원 비중은 {row['여성회원비중']:.2f}입니다."

    return text

df['text'] = df.apply(generate_text, axis=1)
df.head()

# 요일 리스트
weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

# 영업요일이 없는 경우 휴무일을 텍스트 컬럼에 추가하는 함수
def add_closing_days(row):
    # 영업요일이 있는지 확인
    operating_days = row['영업요일']
    
    # 휴무일 계산 (영업하지 않는 요일)
    closed_days = [day for day in weekdays if day not in operating_days]
    
    # 휴무일이 있는지 없는지에 따라 텍스트 추가
    if closed_days:
        closing_text = f"휴무일은 {', '.join(closed_days)}입니다."
    else:
        closing_text = "휴무일은 없습니다."
    
    # 원래 text 컬럼에 휴무일 텍스트 추가
    row['text'] += " " + closing_text
    return row

# 데이터프레임에 적용
df = df.apply(add_closing_days, axis=1)

df.rename(columns={'식당이름': '상호명'}, inplace=True)
df.head()

df = df[~df['상호명'].str.contains(r'\(유\)|\(주\)|\(사\)', regex=True)]

df.info()

df.head()

# 'check'는 Series 형태이므로, loc에 하나의 인덱스만 사용해야 함
check = df[df['상호명'] == '114화교반점']['text']

# 0번째 인덱스의 값을 가져옴
print(check.iloc[0])

df.to_csv("jeju_data_final.csv", index=False)
