class YoutubeData {
  final String title;
  final String videoId;
  final int likes;
  final int views;
  final List<String> keywords;
  final String timestamp;

  YoutubeData({
    required this.title,
    required this.videoId,
    required this.likes,
    required this.views,
    required this.keywords,
    required this.timestamp,
  });

  static YoutubeData fromVideoItem(Map<String, dynamic> item) {
    final snippet = item['snippet'] as Map<String, dynamic>;
    final statistics = item['statistics'] as Map<String, dynamic>;
    final title = snippet['title'] as String;
    final videoId = item['id'] as String;
    
    return YoutubeData(
      title: title,
      videoId: videoId,
      likes: int.tryParse(statistics['likeCount']?.toString() ?? '0') ?? 0,
      views: int.tryParse(statistics['viewCount']?.toString() ?? '0') ?? 0,
      keywords: _extractKeywords(title),
      timestamp: snippet['publishedAt'] as String,
    );
  }

  static List<String> _extractKeywords(String title) {
    // 1. 기본 전처리
    final normalizedTitle = title.toLowerCase()
        .replaceAll(RegExp(r'[^\w\s가-힣]'), ' ') // 특수문자 제거
        .replaceAll(RegExp(r'\s+'), ' ') // 연속된 공백 제거
        .trim();

    // 2. 한국어 키워드 추출 (2글자 이상)
    final koreanKeywords = RegExp(r'[가-힣]{2,}')
        .allMatches(normalizedTitle)
        .map((m) => m.group(0)!)
        .where((word) => !_isStopWord(word))
        .toList();

    // 3. 영어 키워드 추출 (3글자 이상)
    final englishKeywords = RegExp(r'\b[a-zA-Z]{3,}\b')
        .allMatches(normalizedTitle)
        .map((m) => m.group(0)!.toLowerCase())
        .where((word) => !_isStopWord(word))
        .toList();

    // 4. 복합 키워드 추출 (한글 + 영어)
    final compoundKeywords = _extractCompoundKeywords(normalizedTitle);

    // 5. 키워드 중요도 점수 계산 및 정렬
    final allKeywords = [...koreanKeywords, ...englishKeywords, ...compoundKeywords];
    final keywordScores = _calculateKeywordScores(allKeywords, normalizedTitle);
    
    final sortedEntries = keywordScores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    
    return sortedEntries
        .take(5)
        .map((e) => e.key)
        .toList();
  }

  static bool _isStopWord(String word) {
    final stopWords = {
      // 영어 불용어
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
      'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
      'before', 'during', 'through', 'throughout', 'within', 'without',
      'above', 'below', 'under', 'again', 'further', 'then', 'once',
      'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
      'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
      'very', 'can', 'will', 'just', 'should', 'now',
      
      // 한국어 불용어
      '이', '그', '저', '것', '등', '들', '및', '에서', '으로', '에게', '에게서',
      '으로서', '으로써', '으로부터', '으로', '에서', '에게', '에게서',
      '이런', '저런', '그런', '어떤', '무슨', '어느', '이것', '저것', '그것',
      '여기', '저기', '거기', '언제', '어디', '누구', '무엇', '어떻게',
      '왜', '어째서', '어찌', '어째', '어찌나', '어찌하여', '어찌해서',
      '이렇게', '저렇게', '그렇게', '어떻게', '이리', '저리', '그리',
      '이만큼', '저만큼', '그만큼', '얼마나', '얼마', '몇', '몇 개',
      '몇 명', '몇 번', '몇 번째', '몇 시', '몇 분', '몇 초',
      '오늘', '어제', '내일', '모레', '글피', '작년', '작작년', '내년',
      '아침', '점심', '저녁', '새벽', '낮', '밤', '주말', '평일',
      '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일',
      '1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월',
      '봄', '여름', '가을', '겨울',
    };
    return stopWords.contains(word.toLowerCase());
  }

  static List<String> _extractCompoundKeywords(String text) {
    final compoundPattern = RegExp(r'[가-힣]+[a-zA-Z]+|[a-zA-Z]+[가-힣]+');
    return compoundPattern
        .allMatches(text)
        .map((m) => m.group(0)!)
        .where((word) => word.length >= 4)
        .toList();
  }

  static Map<String, double> _calculateKeywordScores(List<String> keywords, String title) {
    final scores = <String, double>{};
    final words = title.split(' ');
    final titleLength = title.length;
    
    for (final keyword in keywords) {
      double score = 0;
      
      // 1. 길이 점수 (키워드가 길수록 더 중요)
      score += keyword.length * 0.2;
      
      // 2. 위치 점수
      if (title.startsWith(keyword)) score += 3; // 제목 시작
      if (title.endsWith(keyword)) score += 2;   // 제목 끝
      
      // 3. 빈도 점수 (키워드가 자주 등장할수록 더 중요)
      final frequency = words.where((w) => w.contains(keyword)).length;
      score += frequency * 0.8;
      
      // 4. 복합어 점수 (한글+영어 조합은 더 중요)
      if (keyword.contains(RegExp(r'[가-힣]')) && keyword.contains(RegExp(r'[a-zA-Z]'))) {
        score += 2.0;
      }
      
      // 5. 제목 내 비중 점수 (전체 제목에서 차지하는 비중)
      final keywordRatio = keyword.length / titleLength;
      score += keywordRatio * 5;
      
      // 6. 대문자 포함 점수 (영문 키워드의 경우)
      if (keyword.contains(RegExp(r'[A-Z]'))) {
        score += 1.5;
      }
      
      // 7. 숫자 포함 점수 (숫자가 포함된 키워드는 더 중요할 수 있음)
      if (keyword.contains(RegExp(r'[0-9]'))) {
        score += 1.0;
      }
      
      scores[keyword] = score;
    }
    
    return scores;
  }
} 