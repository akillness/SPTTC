from pytrends.request import TrendReq
import pandas as pd
import time
from requests.exceptions import RequestException
import random
from datetime import datetime, timedelta

def get_trends_data(pytrends, keyword, max_retries=5):
    for attempt in range(max_retries):
        try:
            # 랜덤 지연 추가 (15-30초)
            delay = random.randint(15, 30) + (attempt * 5)
            print(f"[{datetime.now()}] {keyword} 요청 대기: {delay}초")
            time.sleep(delay)
            
            pytrends.build_payload([keyword], timeframe='today 7-d', geo='KR')
            related_queries = pytrends.related_queries()
            return related_queries
        except RequestException as e:
            print(f"[{datetime.now()}] 시도 {attempt+1}/{max_retries} 실패: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(60 * (attempt + 1))  # 지수 백오프
    return None

try:
    # TrendReq 설정 개선
    pytrends = TrendReq(
        hl='ko-KR',
        tz=540,
        timeout=(60, 120),  # 타임아웃 증가
        retries=5,
        backoff_factor=0.5,
        requests_args={
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
            }
        }
    )

    # 인기 있는 주제어들
    keywords = ["아이유", "BTS", "손흥민", "블랙핑크"]
    print("\n=== 검색 키워드 관련 인기 검색어 ===")

    for keyword in keywords:
        try:
            # 연관 검색어 가져오기
            related_queries = get_trends_data(pytrends, keyword)
            
            if related_queries is None:
                print(f"\n[{keyword}] 데이터를 가져오는데 실패했습니다.")
                continue
                
            print(f"\n[{keyword}] 관련 상위 검색어:")
            # top 관련 검색어가 있는 경우에만 출력
            if keyword in related_queries and related_queries[keyword] is not None and 'top' in related_queries[keyword] and related_queries[keyword]['top'] is not None and not related_queries[keyword]['top'].empty:
                top_queries = related_queries[keyword]['top'].head(5)
                for idx, row in top_queries.iterrows():
                    print(f"{idx + 1}. {row['query']} (검색량: {row['value']})")
            else:
                print("관련 검색어 데이터가 없습니다.")
                
        except Exception as e:
            print(f"[{datetime.now()}] {keyword} 처리 중 치명적 오류: {str(e)}")
            time.sleep(120)  # 심각한 오류 발생 시 장시간 대기
            continue
        
        # 키워드 간 랜덤 지연 (30-60초)
        delay = random.randint(30, 60)
        print(f"[{datetime.now()}] 다음 키워드까지 대기: {delay}초")
        time.sleep(delay)

    # 전체 키워드들의 관심도 트렌드
    try:
        for attempt in range(5):
            try:
                pytrends.build_payload(keywords, timeframe='today 7-d', geo='KR')
                time.sleep(30)
                interest_over_time = pytrends.interest_over_time()
                
                if interest_over_time is not None and not interest_over_time.empty:
                    print("\n=== 키워드 관심도 ===")
                    print(interest_over_time)
                    break
                else:
                    print("\n관심도 데이터를 가져올 수 없습니다.")
                    break
            except Exception as e:
                print(f"관심도 요청 실패 ({attempt+1}/5): {str(e)}")
                if attempt == 4:
                    print("최대 재시도 횟수 도달")
                time.sleep(60 * (attempt + 1))
    except Exception as e:
        print(f"관심도 데이터 가져오기 실패: {str(e)}")

except Exception as e:
    print(f"[{datetime.now()}] 프로그램 실행 중 오류 발생: {str(e)}")