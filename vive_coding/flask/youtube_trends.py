from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
from collections import Counter
import re
from flask_cors import CORS

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)  # CORS 설정 추가
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///youtube_trends.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

def extract_keywords(title):
    try:
        # 한글과 영어를 분리하여 처리
        korean_words = re.findall(r'[가-힣]{2,}', title)
        english_words = re.findall(r'[a-zA-Z0-9]{2,}', title)
        all_words = korean_words + english_words
        
        # 변경: 모든 유효한 키워드 리스트 반환 (중복 포함)
        return all_words if all_words else []

    except Exception as e:
        print(f"키워드 추출 중 오류 발생: {str(e)}")
        return []

class YouTubeTrend(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String(100), nullable=False)
    keyword_type = db.Column(db.String(50))  # 인기 검색어, 추천 검색어 등
    timestamp = db.Column(db.DateTime, nullable=False)
    views = db.Column(db.Integer)
    likes = db.Column(db.Integer)
    dislikes = db.Column(db.Integer)
    frequency = db.Column(db.Integer)  # 검색 빈도
    country_code = db.Column(db.String(2)) # 국가 코드 컬럼 추가

with app.app_context():
    db.create_all()

def get_youtube_service():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key or len(api_key) < 30:
        raise ValueError("유효하지 않은 YouTube API 키 형식")
    return build('youtube', 'v3', developerKey=api_key)

@app.route('/')
def index():
    return render_template('youtube_trends.html')

@app.route('/collect', methods=['POST'])
def collect_trends():
    try:
        youtube = get_youtube_service()
        print(f"API 키 사용 시작: {os.getenv('YOUTUBE_API_KEY')[:5]}...")  # 키 확인

        # 인기 검색어 수집
        search_request = youtube.search().list(
            part='snippet',
            type='video',
            order='viewCount',
            maxResults=50,
            regionCode='KR'
        )
        print(f"검색 요청 파라미터: {search_request.uri}")

        response = search_request.execute()
        print(f"응답 항목 수: {len(response.get('items', []))}")

        if 'items' not in response:
            return jsonify({'error': 'YouTube API 응답 형식이 변경되었습니다'})

        current_country_code = 'KR' # 현재는 한국 데이터만 수집

        successful_saves = 0
        for item in response.get('items', []):
            try:
                # 비디오 ID 검증 강화
                if not item.get('id', {}).get('videoId'):
                    print(f"무효한 비디오 ID 구조: {item.get('id')}")
                    continue

                video_id = item['id']['videoId']
                print(f"처리 시작: 비디오 ID {video_id}")

                # 비디오 상세 정보 요청
                video_request = youtube.videos().list(
                    part='statistics,snippet',
                    id=video_id
                )
                video_response = video_request.execute()

                # 응답 데이터 검증
                if not video_response.get('items'):
                    print(f"경고: 비디오 {video_id} 데이터 없음 - 응답: {video_response}")
                    continue

                video_data = video_response['items'][0]
                stats = video_data.get('statistics', {})
                snippet = video_data.get('snippet', {})

                # 필수 필드 검증
                if not snippet.get('title'):
                    print(f"경고: 비디오 {video_id} 제목 없음")
                    continue

                # 데이터 변환 안전 처리
                def safe_int_convert(value, default=0):
                    try:
                        return int(value) if value else default
                    except (TypeError, ValueError):
                        return default

                # 데이터베이스 객체 생성 시 country_code 추가
                trend = YouTubeTrend(
                    keyword=snippet['title'][:100],
                    keyword_type='인기 검색어',
                    timestamp=datetime.now(),
                    views=safe_int_convert(stats.get('viewCount')),
                    likes=safe_int_convert(stats.get('likeCount')),
                    dislikes=0,
                    frequency=1,
                    country_code=current_country_code # 국가 코드 저장
                )
                
                # 데이터베이스 유효성 검증
                try:
                    db.session.add(trend)
                    db.session.flush()  # 즉시 유효성 검증
                    successful_saves += 1
                    print(f"성공: 비디오 {video_id} 추가됨")
                except Exception as e:
                    db.session.rollback()
                    print(f"데이터베이스 유효성 오류: {str(e)}")
                    continue

            except HttpError as e:
                print(f"API 오류: 비디오 {video_id} - {str(e)}")
                continue
            except Exception as e:
                print(f"비디오 {video_id} 처리 중 예상치 못한 오류: {str(e)}")
                continue

        # 모든 비디오 처리 후 커밋
        if successful_saves > 0:
            try:
                db.session.commit()
                print(f"성공적으로 {successful_saves}개 항목 저장 완료")
            except Exception as e:
                print(f"데이터베이스 커밋 실패: {str(e)}")
                db.session.rollback()
                return jsonify({'error': '데이터베이스 저장 실패'}), 500
        else:
            print("저장할 유효한 데이터가 없습니다.")
            return jsonify({'error': '수집된 데이터가 없습니다'}), 404

        # 수집 후 데이터 정렬
        try:
             trends = YouTubeTrend.query.filter_by(country_code=current_country_code).order_by(
                 db.desc(YouTubeTrend.likes).nullslast(),
                 db.desc(YouTubeTrend.timestamp)
             ).limit(50).all()
        except Exception as e:
             print(f"데이터 정렬 중 오류 (likes): {str(e)}")
             trends = YouTubeTrend.query.filter_by(country_code=current_country_code).order_by(db.desc(YouTubeTrend.timestamp)).limit(50).all()

        # items_data 생성 시 country_code 포함
        items_data = [{
            'keyword': trend.keyword,
            'views': trend.views,
            'likes': trend.likes,
            'dislikes': trend.dislikes,
            'keywords': extract_keywords(trend.keyword),
            'country_code': trend.country_code # 국가 코드 추가
        } for trend in trends]

        return jsonify({'items': items_data})

    except HttpError as e:
        error_detail = f"YouTube API 오류 [{e.resp.status}]: {e._get_reason()}"
        print(error_detail)
        return jsonify({'error': error_detail}), e.resp.status
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()  # 상세 오류 추적 정보
        print(f"심각한 오류 발생:\n{error_trace}")
        return jsonify({
            'error': '데이터 수집 실패',
            'detail': str(e),
            'trace': error_trace.split('\n')  # 개발 환경에서만 노출
        }), 500

@app.route('/trends')
def get_trends():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    # 좋아요 순 정렬 및 국가 필터링 (예: 한국만) - 필요시 국가 선택 기능 추가
    trends_query = YouTubeTrend.query.filter_by(country_code='KR').order_by(
        db.desc(YouTubeTrend.likes).nullslast(),
        db.desc(YouTubeTrend.timestamp)
    )
    trends_page = trends_query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'items': [{
            'keyword': trend.keyword,
            'keyword_type': trend.keyword_type,
            'timestamp': trend.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'views': trend.views,
            'likes': trend.likes,
            'dislikes': trend.dislikes,
            'frequency': trend.frequency,
            'keywords': extract_keywords(trend.keyword),
            'country_code': trend.country_code # 국가 코드 추가
        } for trend in trends_page.items],
        'total_pages': trends_page.pages,
        'current_page': trends_page.page
    })

@app.route('/search')
def search_trends():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    # 검색 쿼리 생성 (좋아요 순 정렬 및 국가 필터링)
    trends_query = YouTubeTrend.query.filter(
        YouTubeTrend.country_code == 'KR', # 국가 필터링
        YouTubeTrend.keyword.ilike(f'%{query}%')
    ).order_by(
        db.desc(YouTubeTrend.likes).nullslast(),
        db.desc(YouTubeTrend.timestamp)
    )

    # 페이지네이션 적용
    trends_page = trends_query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'items': [{
            'keyword': trend.keyword,
            'keyword_type': trend.keyword_type,
            'timestamp': trend.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'views': trend.views,
            'likes': trend.likes,
            'dislikes': trend.dislikes,
            'frequency': trend.frequency,
            'keywords': extract_keywords(trend.keyword),
            'country_code': trend.country_code # 국가 코드 추가
        } for trend in trends_page.items],
        'total_pages': trends_page.pages,
        'current_page': trends_page.page
    })

if __name__ == '__main__':
    app.run(debug=True) 