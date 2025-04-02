class Trend {
  final String keyword;
  final String keywordType;
  final String timestamp;
  final int views;
  final int likes;
  final int dislikes;
  final int frequency;
  final List<String> keywords;
  final String countryCode;

  Trend({
    required this.keyword,
    required this.keywordType,
    required this.timestamp,
    required this.views,
    required this.likes,
    required this.dislikes,
    required this.frequency,
    required this.keywords,
    required this.countryCode,
  });

  factory Trend.fromJson(Map<String, dynamic> json) {
    return Trend(
      keyword: json['keyword'] ?? '',
      keywordType: json['keyword_type'] ?? '',
      timestamp: json['timestamp'] ?? '',
      views: json['views'] ?? 0,
      likes: json['likes'] ?? 0,
      dislikes: json['dislikes'] ?? 0,
      frequency: json['frequency'] ?? 0,
      keywords: json['keywords'] != null 
          ? List<String>.from(json['keywords']) 
          : [],
      countryCode: json['country_code'] ?? 'KR',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'keyword': keyword,
      'keyword_type': keywordType,
      'timestamp': timestamp,
      'views': views,
      'likes': likes,
      'dislikes': dislikes,
      'frequency': frequency,
      'keywords': keywords,
      'country_code': countryCode,
    };
  }
} 