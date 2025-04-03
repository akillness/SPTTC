import 'dart:html' as html;
import 'dart:convert';
import '../models/youtube_data.dart';

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static const String TRENDS_KEY = 'youtube_trends';
  static const String KEYWORDS_KEY = 'youtube_keywords';
  static const String STATS_KEY = 'youtube_stats';

  DatabaseHelper._init();

  Future<void> insertTrend(YoutubeData trend) async {
    try {
      // 기존 데이터 로드
      final trendsJson = html.window.localStorage[TRENDS_KEY] ?? '[]';
      final keywordsJson = html.window.localStorage[KEYWORDS_KEY] ?? '[]';
      final statsJson = html.window.localStorage[STATS_KEY] ?? '{}';

      List<Map<String, dynamic>> trends = List<Map<String, dynamic>>.from(json.decode(trendsJson));
      List<Map<String, dynamic>> keywords = List<Map<String, dynamic>>.from(json.decode(keywordsJson));
      Map<String, dynamic> stats = Map<String, dynamic>.from(json.decode(statsJson));

      // 중복 체크
      if (!trends.any((t) => t['video_id'] == trend.videoId)) {
        // 트렌드 데이터 추가
        trends.add({
          'video_id': trend.videoId,
          'title': trend.title,
          'views': trend.views,
          'likes': trend.likes,
          'timestamp': trend.timestamp,
          'collected_at': DateTime.now().toIso8601String(),
        });

        // 키워드 데이터 추가
        for (final keyword in trend.keywords) {
          keywords.add({
            'video_id': trend.videoId,
            'keyword': keyword,
          });

          // 통계 업데이트
          if (!stats.containsKey(keyword)) {
            stats[keyword] = {
              'total_count': 1,
              'last_seen': DateTime.now().toIso8601String(),
              'avg_views': trend.views,
              'avg_likes': trend.likes,
            };
          } else {
            final stat = stats[keyword];
            stat['total_count'] = (stat['total_count'] as int) + 1;
            stat['last_seen'] = DateTime.now().toIso8601String();
            stat['avg_views'] = ((stat['avg_views'] as num) + trend.views) / 2;
            stat['avg_likes'] = ((stat['avg_likes'] as num) + trend.likes) / 2;
          }
        }

        // 데이터 저장
        html.window.localStorage[TRENDS_KEY] = json.encode(trends);
        html.window.localStorage[KEYWORDS_KEY] = json.encode(keywords);
        html.window.localStorage[STATS_KEY] = json.encode(stats);
      }
    } catch (e) {
      print('Error saving data: $e');
    }
  }

  Future<List<Map<String, dynamic>>> getKeywordStats() async {
    try {
      final statsJson = html.window.localStorage[STATS_KEY] ?? '{}';
      final Map<String, dynamic> stats = Map<String, dynamic>.from(json.decode(statsJson));
      
      return stats.entries.map((entry) => <String, dynamic>{
        'keyword': entry.key,
        ...Map<String, dynamic>.from(entry.value as Map),
      }).toList()
        ..sort((a, b) => (b['total_count'] as int).compareTo(a['total_count'] as int));
    } catch (e) {
      print('Error getting keyword stats: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getTrendsByKeyword(String keyword) async {
    try {
      final trendsJson = html.window.localStorage[TRENDS_KEY] ?? '[]';
      final keywordsJson = html.window.localStorage[KEYWORDS_KEY] ?? '[]';

      final List<Map<String, dynamic>> trends = List<Map<String, dynamic>>.from(
        (json.decode(trendsJson) as List).map((item) => Map<String, dynamic>.from(item)).toList()
      );
      final List<Map<String, dynamic>> keywords = List<Map<String, dynamic>>.from(
        (json.decode(keywordsJson) as List).map((item) => Map<String, dynamic>.from(item)).toList()
      );

      final videoIds = keywords
          .where((k) => k['keyword'] == keyword)
          .map((k) => k['video_id'] as String)
          .toSet();

      return trends
          .where((t) => videoIds.contains(t['video_id']))
          .toList()
        ..sort((a, b) => (b['views'] as int).compareTo(a['views'] as int));
    } catch (e) {
      print('Error getting trends by keyword: $e');
      return [];
    }
  }

  Future<List<Map<String, dynamic>>> getTopTrends({int limit = 10}) async {
    try {
      final trendsJson = html.window.localStorage[TRENDS_KEY] ?? '[]';
      final List<Map<String, dynamic>> trends = List<Map<String, dynamic>>.from(
        (json.decode(trendsJson) as List).map((item) => Map<String, dynamic>.from(item)).toList()
      );

      return trends
        ..sort((a, b) => (b['views'] as int).compareTo(a['views'] as int))
        ..take(limit)
        .toList();
    } catch (e) {
      print('Error getting top trends: $e');
      return [];
    }
  }

  Future<Map<String, dynamic>> getKeywordTrends() async {
    try {
      final statsJson = html.window.localStorage[STATS_KEY] ?? '{}';
      final stats = Map<String, dynamic>.from(json.decode(statsJson));

      final results = stats.entries.map((entry) => {
        'keyword': entry.key,
        'count': entry.value['total_count'],
        'avg_views': entry.value['avg_views'],
        'avg_likes': entry.value['avg_likes'],
      }).toList()
        ..sort((a, b) => (b['count'] as int).compareTo(a['count'] as int))
        ..take(20);

      return {
        'keywords': results.map((r) => r['keyword']).toList(),
        'counts': results.map((r) => r['count']).toList(),
        'avgViews': results.map((r) => r['avg_views']).toList(),
        'avgLikes': results.map((r) => r['avg_likes']).toList(),
      };
    } catch (e) {
      print('Error getting keyword trends: $e');
      return {
        'keywords': [],
        'counts': [],
        'avgViews': [],
        'avgLikes': [],
      };
    }
  }
} 