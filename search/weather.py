import requests
import json
from datetime import datetime
import pandas as pd

# 기상청 단기예보 조회 API 기본 URL
API_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# 사용자의 Encoding된 인증키
ENCODED_SERVICE_KEY = r"w9mjgQS/+xk/bQ8bBCBgr6ZNgAeqLQry7V/zW4bcWKtUpPR+PdbExY/3pWbiFFZnF6TTRtl7jIGduPI5XCoqcg=="

def translate_data(location, data):
    items = data["response"]["body"]["items"]["item"]
    text = f"{location} 지역 날씨 정보\n"
    for item in items:
        category = item['category']
        value = item['obsrValue']
        if category == "PTY":
            if value=="0": text += "강수형태: 없음\n"
            elif value=="1": text += "강수형태: 비\n"
            elif value=="2": text += "강수형태: 비/눈\n"
            elif value=="3": text += "강수형태: 눈\n"
            elif value=="4": text += "강수형태: 소나기\n"
            elif value=="5": text += "강수형태: 빗방울\n"
            elif value=="6": text += "강수형태: 빗방울눈 날림\n"
            elif value=="7": text += "강수형태: 눈 날림\n"
        elif category == "REH":
            text += f"습도: {value}%\n"
        elif category == "RN1":
            text += f"1시간 강수량: {value}mm\n"
        elif category == "T1H":
            text += f"기온: {value}도\n"
    return text


def get_weather_forecast(location):
    today = datetime.today().strftime("%Y%m%d")
    base_time = "0500"  # 발표시각 예시 (변경 가능)
    df = pd.read_excel('places.xlsx')

    nx, ny = find_coordinates(location, df)

    # 요청 파라미터 설정
    params = {
        "serviceKey": ENCODED_SERVICE_KEY,
        "numOfRows": 4,
        "pageNo": 1,
        "totalCount":10,
        "resultMsg":100,
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
            result = translate_data(location, data)
            return result
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
