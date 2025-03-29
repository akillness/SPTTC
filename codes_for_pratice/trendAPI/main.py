import feedparser  # pip install feedparser

def get_rss_trends(geo="KR"):
    """Google Trends RSS 피드 파싱"""
    url = f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"
    feed = feedparser.parse(url)
    trends = [entry.title for entry in feed.entries]
    return trends

# 실행 예시
if __name__ == "__main__":
    trends = get_rss_trends()
    print("🇰🇷 한국 일일 검색 트렌드:")
    for i, trend in enumerate(trends[:20], 1):
        print(f"{i}. {trend}")