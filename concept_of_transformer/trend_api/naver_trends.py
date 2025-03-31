import os
import json
import urllib.request
from datetime import datetime, timedelta

class NaverDataLabAPI:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        
    def get_trending_keywords(self, start_date=None, end_date=None):
        """
        Get trending keywords from Naver DataLab
        
        Args:
            start_date (str): Start date in format 'yyyy-mm-dd' (default: 7 days ago)
            end_date (str): End date in format 'yyyy-mm-dd' (default: today)
            
        Returns:
            list: List of trending keywords with their relative ratios
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        url = "https://openapi.naver.com/v1/datalab/search"
        
        # 예시 키워드 (실제로는 더 많은 키워드를 추가할 수 있습니다)
        keywords = ["코로나", "주식", "부동산", "취업", "날씨"]
        
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "date",
            "keywordGroups": [
                {
                    "groupName": keyword,
                    "keywords": [keyword]
                } for keyword in keywords
            ]
        }
            
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        request.add_header("Content-Type", "application/json")
        
        try:
            response = urllib.request.urlopen(request, data=json.dumps(body).encode("utf-8"))
            rescode = response.getcode()
            response_body = response.read()
            
            if rescode == 200:
                data = json.loads(response_body.decode('utf-8'))
                return self._process_response(data)
            else:
                error_data = json.loads(response_body.decode('utf-8'))
                print(f"Error Code: {rescode}")
                print(f"Error Message: {error_data.get('errorMessage', '알 수 없는 오류가 발생했습니다.')}")
                print(f"Error Code: {error_data.get('errorCode', '알 수 없음')}")
                return []
        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            try:
                error_body = e.read().decode('utf-8')
                error_data = json.loads(error_body)
                print(f"Error Message: {error_data.get('errorMessage', '알 수 없는 오류가 발생했습니다.')}")
                print(f"Error Code: {error_data.get('errorCode', '알 수 없음')}")
            except:
                print("추가 오류 정보를 가져올 수 없습니다.")
            return []
        except Exception as e:
            print(f"Error getting trending keywords: {e}")
            return []
            
    def _process_response(self, data):
        """Process the response from Naver DataLab API"""
        results = []
        if 'results' in data:
            for result in data['results']:
                keyword = result['title']
                # Get the most recent ratio
                latest_ratio = result['data'][-1]['ratio']
                results.append({
                    'keyword': keyword,
                    'ratio': latest_ratio
                })
        return sorted(results, key=lambda x: x['ratio'], reverse=True)

if __name__ == "__main__":
    # Naver API 인증정보 설정
    CLIENT_ID = ""
    CLIENT_SECRET = ""
    
    print("=== Naver 트렌드 키워드 ===")
    print("주의: API 사용을 위해서는 네이버 개발자 센터(https://developers.naver.com)에서")
    print("애플리케이션을 등록하고 클라이언트 ID와 시크릿을 발급받아야 합니다.")
    print("\n실제 사용을 위해서는 코드의 CLIENT_ID와 CLIENT_SECRET을 발급받은 값으로 교체해주세요.")
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("\n오류: CLIENT_ID와 CLIENT_SECRET이 설정되지 않았습니다.")
        print("네이버 개발자 센터에서 발급받은 인증정보를 입력해주세요.")
    else:
        api = NaverDataLabAPI(CLIENT_ID, CLIENT_SECRET)
        trends = api.get_trending_keywords()
        
        if trends:
            print("\n현재 인기 검색어:")
            for i, trend in enumerate(trends, 1):
                print(f"{i}. {trend['keyword']} (상대 검색량: {trend['ratio']})")
        else:
            print("\n트렌드 데이터를 가져오는데 실패했습니다.")
            print("다음 사항을 확인해주세요:")
            print("1. 네이버 개발자 센터에서 애플리케이션이 정상적으로 등록되어 있는지")
            print("2. 데이터랩 API 사용 권한이 설정되어 있는지")
            print("3. Client ID와 Client Secret이 올바른지")
            print("4. API 호출 횟수 제한에 도달하지 않았는지") 