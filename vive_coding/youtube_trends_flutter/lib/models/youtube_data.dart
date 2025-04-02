class YoutubeData {
  final String title;
  final int likes;
  final int views;
  final List<String> keywords;
  final String timestamp;

  YoutubeData({
    required this.title,
    required this.likes,
    required this.views,
    required this.keywords,
    required this.timestamp,
  });

  static YoutubeData fromVideoItem(Map<String, dynamic> item) {
    final snippet = item['snippet'] as Map<String, dynamic>;
    final statistics = item['statistics'] as Map<String, dynamic>;
    final title = snippet['title'] as String;
    
    return YoutubeData(
      title: title,
      likes: int.tryParse(statistics['likeCount']?.toString() ?? '0') ?? 0,
      views: int.tryParse(statistics['viewCount']?.toString() ?? '0') ?? 0,
      keywords: _extractKeywords(title),
      timestamp: snippet['publishedAt'] as String,
    );
  }

  static List<String> _extractKeywords(String title) {
    final koreanRegExp = RegExp(r'[가-힣]{2,}');
    final englishRegExp = RegExp(r'\b[a-zA-Z]{3,}\b');
    
    return [
      ...koreanRegExp.allMatches(title).map((m) => m.group(0)!),
      ...englishRegExp.allMatches(title).map((m) => m.group(0)!.toLowerCase()),
    ];
  }
} 