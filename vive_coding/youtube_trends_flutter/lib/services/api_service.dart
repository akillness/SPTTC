import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/youtube_data.dart';
import '../models/api_response.dart';

class ApiService {
  static const String _apiKey = 'AIzaSyCE9hxzOP0uYI0YbrPNJFG1I_w-PRW9Usc';
  static const String _baseUrl = 'https://www.googleapis.com/youtube/v3';
  
  Future<ApiResponse> getTrends() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/videos?part=snippet,statistics&chart=mostPopular&regionCode=KR&maxResults=50&key=$_apiKey'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final items = (data['items'] as List)
            .map((item) => YoutubeData.fromVideoItem(item))
            .toList();
            
        return ApiResponse(
          items: items,
          total: items.length,
          currentPage: 1,
          totalPages: 1
        );
      } else {
        print('Error fetching trends: ${response.statusCode}');
        return ApiResponse(
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1
        );
      }
    } catch (e) {
      print('Error fetching trends: $e');
      return ApiResponse(
        items: [],
        total: 0,
        currentPage: 1,
        totalPages: 1
      );
    }
  }
  
  Future<ApiResponse> collectTrends() async {
    return getTrends();
  }
  
  Future<ApiResponse> searchTrends(String query) async {
    try {
      final searchResponse = await http.get(
        Uri.parse('$_baseUrl/search?part=snippet&q=$query&type=video&regionCode=KR&maxResults=50&key=$_apiKey'),
      );

      if (searchResponse.statusCode != 200) {
        print('Error searching trends: ${searchResponse.statusCode}');
        return ApiResponse(
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1
        );
      }

      final searchData = json.decode(searchResponse.body);
      final videoIds = (searchData['items'] as List)
          .map((item) => item['id']['videoId'])
          .join(',');

      if (videoIds.isEmpty) {
        return ApiResponse(
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1
        );
      }

      final videosResponse = await http.get(
        Uri.parse('$_baseUrl/videos?part=snippet,statistics&id=$videoIds&key=$_apiKey'),
      );

      if (videosResponse.statusCode == 200) {
        final videosData = json.decode(videosResponse.body);
        final items = (videosData['items'] as List)
            .map((item) => YoutubeData.fromVideoItem(item))
            .toList();
            
        return ApiResponse(
          items: items,
          total: items.length,
          currentPage: 1,
          totalPages: 1
        );
      } else {
        print('Error fetching video details: ${videosResponse.statusCode}');
        return ApiResponse(
          items: [],
          total: 0,
          currentPage: 1,
          totalPages: 1
        );
      }
    } catch (e) {
      print('Error searching trends: $e');
      return ApiResponse(
        items: [],
        total: 0,
        currentPage: 1,
        totalPages: 1
      );
    }
  }
} 