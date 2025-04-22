import datetime
import os
from typing import Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from smolagents import tool

# 환경 변수 로드
load_dotenv()

# Google AI 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 설정 - 무료 버전에 맞게 조정
generation_config = {
    "temperature": 0.7,  # 더 안정적인 응답을 위해 낮춤
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,  # 토큰 수 제한
}

def get_coordinates(location: str) -> Tuple[float, float]:
    """
    위치 문자열을 좌표로 변환합니다.
    """
    try:
        geolocator = Nominatim(user_agent="weather_agent")
        location_data = geolocator.geocode(location)
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            raise ValueError(f"위치를 찾을 수 없습니다: {location}")
    except GeocoderTimedOut:
        raise TimeoutError("위치 검색 시간이 초과되었습니다.")

def get_weather_info(location: str) -> str:
    """
    Google Gemini를 사용하여 날씨 정보와 설명을 가져옵니다.
    """
    prompt = f"""
당신은 전문적인 날씨 정보 제공자입니다. 
{location}의 현재 날씨 정보를 검색하여 아래 형식으로 응답해주세요.

===날씨 데이터===
{{
    "temperature": "20°C",
    "feels_like": "19°C",
    "humidity": "60%",
    "condition": "맑음",
    "wind_speed": "2.5m/s"
}}

===날씨 설명===
1. 현재 날씨가 야외 활동하기에 적합한지 설명
2. 필요한 준비물이나 주의사항 안내
3. 체감온도와 실제 기온의 차이가 있다면 그에 대한 설명

위 형식을 정확히 지켜서 응답해주세요. 특히 JSON 형식과 구분자(===)를 정확하게 사용해주세요.
"""

    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-pro-latest",
        generation_config=generation_config
    )
    response = model.generate_content(prompt)
    return response.text

def get_weather_report(location: str) -> str:
    """
    날씨 정보를 조회하고 리포트를 생성합니다.
    """
    try:
        # 위치를 좌표로 변환 (참고용)
        lat, lon = get_coordinates(location)
        print(f"위치 '{location}'이(가) 좌표 ({lat}, {lon})로 변환되었습니다.")
        
        # Google AI로 날씨 정보와 설명 조회
        response = get_weather_info(location)
        
        # 응답 파싱
        sections = response.split("===")
        weather_data = ""
        weather_description = ""
        
        found_data_label = False
        found_desc_label = False

        for i, section in enumerate(sections):
            section_content = section.strip()
            
            # 이전 섹션에서 데이터 레이블을 찾았으면 현재 섹션은 데이터 내용임
            if found_data_label:
                # ```json 태그 제거
                if section_content.startswith("```json"):
                    section_content = section_content.split("```json", 1)[1]
                if section_content.endswith("```"):
                    section_content = section_content.rsplit("```", 1)[0]
                weather_data = section_content.strip()
                found_data_label = False # 처리 완료
                continue # 다음 섹션으로

            # 이전 섹션에서 설명 레이블을 찾았으면 현재 섹션은 설명 내용임
            if found_desc_label:
                 # 설명 부분에서 주의사항 제거
                description_lines = [line for line in section_content.split("\n")
                                  if not line.startswith("**") and line.strip()]
                weather_description = "\n".join(description_lines)
                found_desc_label = False # 처리 완료
                continue # 다음 섹션으로

            # 현재 섹션이 데이터 레이블인지 확인
            if section_content == "날씨 데이터":
                found_data_label = True
                continue

            # 현재 섹션이 설명 레이블인지 확인
            if section_content == "날씨 설명":
                found_desc_label = True
                continue
                
        # 파싱 결과 확인 (필요시 주석 해제)
        # print(f"--- 최종 파싱된 데이터 ---")
        # print(weather_data)
        # print("-------------------------")
        # print(f"--- 최종 파싱된 설명 ---")
        # print(weather_description)
        # print("------------------------")

        return f"""
날씨 리포트 - {location}

[날씨 데이터]
{weather_data}

[AI 날씨 설명]
{weather_description}
"""
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        raise e

def main():
    location = "Gangnam, Seoul, South Korea"
    print(f"\n{location}의 날씨 정보를 조회합니다...\n")
    
    try:
        weather_report = get_weather_report(location)
        print("\n최종 날씨 리포트:")
        print(weather_report)
    except Exception as e:
        print(f"날씨 정보 조회 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 