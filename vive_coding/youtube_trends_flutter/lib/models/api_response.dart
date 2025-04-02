import 'youtube_data.dart';

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