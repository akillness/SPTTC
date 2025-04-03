import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../models/youtube_data.dart';
import '../widgets/charts/trends_pie_chart.dart';
import '../widgets/charts/keyword_typography.dart';
import '../widgets/pagination_controls.dart';

class TrendsScreen extends StatefulWidget {
  const TrendsScreen({super.key});

  @override
  State<TrendsScreen> createState() => _TrendsScreenState();
}

class _TrendsScreenState extends State<TrendsScreen> {
  final ApiService _apiService = ApiService();
  List<YoutubeData> _trends = [];
  bool _isLoading = false;
  String _searchQuery = '';
  int _currentPage = 1;
  int _totalPages = 1;
  static const int _itemsPerPage = 10;

  @override
  void initState() {
    super.initState();
    _loadTrends();
  }

  Future<void> _loadTrends() async {
    setState(() => _isLoading = true);
    try {
      final response = await _apiService.getTrends();
      setState(() {
        _trends = response.items;
        _trends.sort((a, b) => b.views.compareTo(a.views));
        _totalPages = (_trends.length / _itemsPerPage).ceil();
        _currentPage = 1;
      });
    } catch (e) {
      print('Error loading trends: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _searchTrends() async {
    if (_searchQuery.isEmpty) return;
    setState(() => _isLoading = true);
    try {
      final response = await _apiService.searchTrends(_searchQuery);
      setState(() {
        _trends = response.items;
        _trends.sort((a, b) => b.views.compareTo(a.views));
        _totalPages = (_trends.length / _itemsPerPage).ceil();
        _currentPage = 1;
      });
    } catch (e) {
      print('Error searching trends: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _handlePageChange(int newPage) {
    setState(() => _currentPage = newPage);
  }

  List<YoutubeData> get _currentPageItems {
    final startIndex = (_currentPage - 1) * _itemsPerPage;
    final endIndex = startIndex + _itemsPerPage;
    return _trends.sublist(
      startIndex,
      endIndex > _trends.length ? _trends.length : endIndex,
    );
  }

  Widget _buildDataTable() {
    return Column(
      children: [
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: DataTable(
            columns: const [
              DataColumn(label: Text('순위')),
              DataColumn(label: Text('제목')),
              DataColumn(label: Text('조회수')),
              DataColumn(label: Text('좋아요')),
              DataColumn(label: Text('키워드')),
              DataColumn(label: Text('업로드 시간')),
            ],
            rows: _currentPageItems.asMap().entries.map((entry) {
              final index = entry.key;
              final item = entry.value;
              final rank = (_currentPage - 1) * _itemsPerPage + index + 1;
              
              Color getRankColor() {
                if (rank <= 3) return Colors.amber;
                if (rank <= 10) return Colors.blue;
                if (rank <= 20) return Colors.green;
                return Colors.grey;
              }

              return DataRow(
                color: MaterialStateProperty.resolveWith<Color?>(
                  (Set<MaterialState> states) {
                    if (states.contains(MaterialState.hovered)) {
                      return Colors.grey.withOpacity(0.1);
                    }
                    return rank <= 10 ? getRankColor().withOpacity(0.05) : null;
                  },
                ),
                cells: [
                  DataCell(
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: getRankColor().withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: getRankColor().withOpacity(0.3),
                          width: 1,
                        ),
                      ),
                      child: Text(
                        rank.toString(),
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: getRankColor(),
                        ),
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      item.title,
                      style: TextStyle(
                        fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      NumberFormat.compact().format(item.views),
                      style: TextStyle(
                        color: rank <= 10 ? Colors.blue : null,
                        fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                  ),
                  DataCell(
                    Text(
                      NumberFormat.compact().format(item.likes),
                      style: TextStyle(
                        color: rank <= 10 ? Colors.red : null,
                        fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                  ),
                  DataCell(
                    Wrap(
                      spacing: 4,
                      children: item.keywords.map((keyword) {
                        return Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.grey[200],
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            keyword,
                            style: const TextStyle(fontSize: 12),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                  DataCell(Text(item.timestamp)),
                ],
              );
            }).toList(),
          ),
        ),
        PaginationControls(
          currentPage: _currentPage,
          totalPages: _totalPages,
          onPageChanged: _handlePageChange,
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YouTube 트렌드 분석'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadTrends,
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(
                      hintText: '검색어를 입력하세요',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) => _searchQuery = value,
                    onSubmitted: (_) => _searchTrends(),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _searchTrends,
                  child: const Text('검색'),
                ),
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: _loadTrends,
                  icon: const Icon(Icons.download),
                  label: const Text('트렌드 데이터 로드'),
                ),
              ],
            ),
          ),
          if (_isLoading)
            const Center(child: CircularProgressIndicator())
          else
            Expanded(
              child: DefaultTabController(
                length: 3,
                child: Column(
                  children: [
                    const TabBar(
                      tabs: [
                        Tab(text: '데이터 테이블'),
                        Tab(text: '파이 차트'),
                        Tab(text: '키워드 차트'),
                      ],
                    ),
                    Expanded(
                      child: TabBarView(
                        children: [
                          _buildDataTable(),
                          TrendsPieChart(trends: _trends),
                          KeywordTypography(trends: _trends),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
} 