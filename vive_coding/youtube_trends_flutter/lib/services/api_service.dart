import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:googleapis/youtube/v3.dart';
import 'package:googleapis_auth/auth_io.dart';
import '../models/trend.dart';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

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

  static YoutubeData fromVideoItem(Video video) {
    final keywords = _extractKeywords(video.snippet?.title ?? '');
    return YoutubeData(
      title: video.snippet?.title ?? '제목 없음',
      likes: int.tryParse(video.statistics?.likeCount ?? '0') ?? 0,
      views: int.tryParse(video.statistics?.viewCount ?? '0') ?? 0,
      keywords: keywords,
      timestamp: video.snippet?.publishedAt?.toIso8601String() ?? '',
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

class ApiResponse {
  final List<YoutubeData> items;
  final int total;
  final int currentPage;
  final int totalPages;

  ApiResponse({
    required this.items,
    required this.total,
    required this.currentPage,
    required this.totalPages,
  });
}

class ApiService {
  static const String _apiKey = 'AIzaSyCE9hxzOP0uYI0YbrPNJFG1I_w-PRW9Usc';  
  
  late final YouTubeApi _youtubeApi;
  bool _isInitialized = false;

  Future<void> _initialize() async {
    if (_isInitialized) return;
    final authClient = await clientViaApiKey(_apiKey);
    _youtubeApi = YouTubeApi(authClient);
    _isInitialized = true;
  }

  Future<ApiResponse> getTrends({int page = 1, int perPage = 10}) async {
    await _initialize();
    
    final searchResponse = await _youtubeApi.search.list(
      ['snippet'],
      type: ['video'],
      order: 'viewCount',
      maxResults: perPage,
      regionCode: 'KR',
      relevanceLanguage: 'ko',
    );

    if (searchResponse.items == null) {
      return ApiResponse(
        items: [],
        total: 0,
        currentPage: page,
        totalPages: 0,
      );
    }

    final videoIds = searchResponse.items!
        .map((item) => item.id?.videoId ?? '')
        .where((id) => id.isNotEmpty)
        .toList();

    final videosResponse = await _youtubeApi.videos.list(
      ['snippet', 'statistics'],
      id: videoIds,
    );

    final items = videosResponse.items?.map(YoutubeData.fromVideoItem).toList() ?? [];
    final totalResults = searchResponse.pageInfo?.totalResults ?? 0;
    final totalPages = (totalResults / perPage).ceil();

    return ApiResponse(
      items: items,
      total: totalResults,
      currentPage: page,
      totalPages: totalPages,
    );
  }

  Future<ApiResponse> collectTrends() async {
    return getTrends(perPage: 50);
  }

  Future<Map<String, dynamic>> searchTrends(String query, {int page = 1, int perPage = 10}) async {
    await _initialize();
    
    try {
      final searchResponse = await _youtubeApi.search.list(
        ['snippet'],
        type: ['video'],
        q: query,
        order: 'viewCount',
        maxResults: perPage,
        regionCode: 'KR',
      );

      if (searchResponse.items == null || searchResponse.items!.isEmpty) {
        return {'items': [], 'total': 0};
      }

      final videoIds = searchResponse.items!
          .map((item) => item.id?.videoId ?? '')
          .where((id) => id.isNotEmpty)
          .toList();

      final videosResponse = await _youtubeApi.videos.list(
        ['snippet', 'statistics'],
        id: videoIds,
      );

      final items = videosResponse.items?.map((video) {
        final keywords = _extractKeywords(video.snippet?.title ?? '');
        return Trend(
          keyword: video.snippet?.title ?? '제목 없음',
          likes: int.tryParse(video.statistics?.likeCount ?? '0') ?? 0,
          views: int.tryParse(video.statistics?.viewCount ?? '0') ?? 0,
          dislikes: int.tryParse(video.statistics?.dislikeCount ?? '0') ?? 0,
          frequency: keywords.length,
          keywords: keywords,
          countryCode: 'KR',
          timestamp: video.snippet?.publishedAt?.toIso8601String() ?? '',
          keywordType: '검색결과',
        ).toJson();
      }).toList() ?? [];

      return {
        'items': items,
        'total': searchResponse.pageInfo?.totalResults ?? 0,
        'current_page': page,
        'total_pages': (searchResponse.pageInfo?.totalResults ?? 0) ~/ perPage + 1,
      };
    } catch (e) {
      print('검색 오류: $e');
      rethrow;
    }
  }

  List<String> _extractKeywords(String title) {
    final RegExp koreanRegExp = RegExp(r'[가-힣]{2,}');
    final RegExp englishRegExp = RegExp(r'\b[a-zA-Z]{3,}\b');
    
    return [
      ...koreanRegExp.allMatches(title).map((m) => m.group(0)!),
      ...englishRegExp.allMatches(title).map((m) => m.group(0)!.toLowerCase()),
    ];
  }

  int _getInt(dynamic value) => (value is num?) ? value?.toInt() ?? 0 : 0;
  String _getString(dynamic value) => (value is String?) ? value ?? '' : '';
  List<String> _getStringList(dynamic value) => 
      (value is List<dynamic>) ? value.map((e) => e.toString()).toList() : [];
}

class DataVisualization {
  static List<BarChartGroupData> buildBarChart(List<Map<String, dynamic>> trends) {
    return trends.asMap().entries.map((entry) {
      final index = entry.key;
      final trend = entry.value;
      final likes = (trend['likes'] as num).toDouble();

      return BarChartGroupData(
        x: index,
        barRods: [BarChartRodData(toY: likes, color: Colors.blue, width: 16)],
      );
    }).toList();
  }

  static Widget buildDataTable(List<Map<String, dynamic>> trends) {
    final format = NumberFormat('#,###');
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columns: const [
          DataColumn(label: Text('제목')),
          DataColumn(label: Text('조회수')),
          DataColumn(label: Text('좋아요')),
          DataColumn(label: Text('키워드')),
          DataColumn(label: Text('시간')),
        ],
        rows: trends.map((trend) => DataRow(
          cells: [
            DataCell(Text(trend['keyword'].toString())),
            DataCell(Text(format.format(trend['views']))),
            DataCell(Text(format.format(trend['likes']))),
            DataCell(Text((trend['keywords'] as List).join(', '))),
            DataCell(Text(trend['timestamp'].toString())),
          ],
        )).toList(),
      ),
    );
  }
}

class ChartColors {
  static const List<Color> gradientColors = [
    Color(0xFF3498DB), // 파란색
    Color(0xFF2ECC71), // 녹색
    // ... 기존 색상 목록
  ];

  static Color getGradientColor(int index) => 
      gradientColors[index % gradientColors.length];
  
  static Color getKeywordColor(String keyword) => 
      Colors.accents[keyword.hashCode % Colors.accents.length];
}

class PaginationControls extends StatelessWidget {
  final int currentPage;
  final int totalPages;
  final Function(int) onPageChanged;

  const PaginationControls({
    super.key,
    required this.currentPage,
    required this.totalPages,
    required this.onPageChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: currentPage > 1 ? () => onPageChanged(-1) : null,
        ),
        Text('$currentPage / $totalPages'),
        IconButton(
          icon: const Icon(Icons.arrow_forward),
          onPressed: currentPage < totalPages ? () => onPageChanged(1) : null,
        ),
      ],
    );
  }
} 