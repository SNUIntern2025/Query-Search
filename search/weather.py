import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from config import API_URL, ENCODED_SERVICE_KEY


def translate_data(location, data, forcast_day):
    items = data["response"]["body"]["items"]["item"]
    text = f"{location} 지역 날씨 정보\n"
    for item in items:
        if item['fcstDate'] != forcast_day:
            continue
        category = item['category']
        value = item['fcstValue']
        if category == "PTY":
            if value=="0": text += "강수형태: 맑음\n"
            elif value=="1": text += "강수형태: 비\n"
            elif value=="2": text += "강수형태: 비/눈\n"
            elif value=="3": text += "강수형태: 눈\n"
            elif value=="4": text += "강수형태: 소나기\n"
            elif value=="5": text += "강수형태: 빗방울\n"
            elif value=="6": text += "강수형태: 빗방울눈 날림\n"
            elif value=="7": text += "강수형태: 눈 날림\n"
        elif category == "REH":
            text += f"습도: {value}%\n"
        elif category == "POP":
            text += f"강수확률: {value}%\n"
        elif category == "SKY":
            value = int(value)
            if value >= 0 and value <= 5: text += "하늘상태: 맑음\n"
            elif value >= 6 and value <= 8: text += "하늘상태: 구름많음\n"
            elif value == 9 or value == 10: text += "하늘상태: 흐림\n"
        elif category == "TMN":
            text += f"일 최저기온: {value}도\n"
        elif category == "TMX":
            text += f"일 최고기온: {value}도\n"
    return text

# ['내일', '현재', '모레', '글피', '어제', '오늘', '주말', '평일']
def get_weather_forecast(location, date='오늘'):
    weekday_list = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

    # 현재 시각이 23시 이전이면 어제 날짜
    if datetime.now().hour < 23:
        today = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        today = datetime.now().strftime("%Y%m%d")
    base_time = "2300"  # 발표시각 예시 (변경 가능)
    forcast_time = "2300"  # 예보시각 예시 (변경 가능)
    if date == '내일':
        forcast_day = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    elif date == '모레':
        forcast_day = (datetime.now() + timedelta(days=2)).strftime("%Y%m%d")
    elif date == '글피':
        forcast_day = (datetime.now() + timedelta(days=3)).strftime("%Y%m%d")
    elif date == '주말':
        # 현재 요일 조회
        weekday = datetime.now().weekday()
        if weekday >= 5:  # 이미 주말인 경우
            forcast_day = datetime.now().strftime("%Y%m%d")
        else:
            # 다음 주 토요일로 이동
            forcast_day = (datetime.now() + timedelta(days=(5-weekday))).strftime("%Y%m%d")
    elif date == '평일':
        # 현재 요일 조회
        weekday = datetime.now().weekday()
        if weekday < 5:
            forcast_day = datetime.now().strftime("%Y%m%d")
        else:
            # 다음 평일로 이동
            forcast_day = (datetime.now() + timedelta(days=(7-weekday))).strftime("%Y%m%d")
    elif date in weekday_list:
        # date가 weekday_list의 몇 번째 index인지 찾기
        target_weekday = weekday_list.index(date)
        # 현재 요일 조회
        weekday = datetime.now().weekday()
        # 다음 해당 요일로 이동
        if target_weekday < weekday:
            target_weekday += 7
        forcast_day = (datetime.now() + timedelta(days=(target_weekday-weekday))).strftime("%Y%m%d")
    else:
        forcast_day = today

    df = pd.read_csv('search/places.csv')   # df 로딩 시간 줄이기 위해 csv로 변경

    nx, ny = find_coordinates(location, df)

    # 요청 파라미터 설정
    params = {
        "serviceKey": ENCODED_SERVICE_KEY,
        "numOfRows": 1000,
        "pageNo": 1,
        "base_date": today,  # 발표일자 (예: 20240101)
        "base_time": base_time,   # 발표시각 (예: 0500, 0800 등)
        "nx": nx,                 # 지점의 x 좌표
        "ny": ny,                 # 지점의 y 좌표
        "dataType": "JSON"   # JSON 형식
    }

    try:
        # API 요청 수행
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # HTTP 요청 실패 시 예외 처리

        # JSON 응답 데이터 가져오기
        data = response.json()

        # API 응답 코드 확인
        result_code = data["response"]["header"]["resultCode"]
        result_msg = data["response"]["header"]["resultMsg"]

        if result_code == "00":
            result = translate_data(location, data, forcast_day)
            return f"{location} {date}날씨: {result}"
        else:
            print(f"API 오류 발생 [{result_code}]: {result_msg}")
            return None

    except requests.exceptions.RequestException as e:
        print("HTTP 요청 에러 발생:", e)
        return None
    
# 지명을 입력받아 격자 x, y 값을 찾는 함수
def find_coordinates(location, df):
    if location is None:
        return None, None
    for col in ['1단계', '2단계', '3단계']:
        matched_rows = df[df[col].astype(str).str.contains(location, na=False, case=False)]
        #matched_rows = df[df[col] == place_name]
        if not matched_rows.empty:
            # 첫 번째로 발견된 행의 격자 x, 격자 y 반환
            x = matched_rows.iloc[0]['격자 X']
            y = matched_rows.iloc[0]['격자 Y']
            return x, y
    return None, None  # 지명을 찾지 못한 경우

# 초단기실황
if __name__ == "__main__":
    location = input("지역명: ")
    forecast_items = get_weather_forecast(location)
    print(forecast_items)
