import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/youtube_data.dart';
import '../widgets/trends_chart.dart';
import '../widgets/trends_table.dart';
import '../widgets/pagination_controls.dart';

class TrendsScreen extends StatefulWidget {
  const TrendsScreen({super.key});

  @override
  State<TrendsScreen> createState() => _TrendsScreenState();
}

class _TrendsScreenState extends State<TrendsScreen> {
  final ApiService _api = ApiService();
  List<YoutubeData> _trends = [];
  int _currentPage = 1;
  int _totalPages = 1;
  bool _isLoading = false;
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    
    try {
      final response = _searchQuery.isEmpty
          ? await _api.getTrends()
          : await _api.searchTrends(_searchQuery);
          
      setState(() {
        _trends = response.items;
        _totalPages = response.totalPages;
        _currentPage = response.currentPage;
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _handlePageChange(int delta) {
    setState(() => _currentPage += delta);
    _loadData();
  }

  void _handleSearch(String query) {
    setState(() {
      _searchQuery = query;
      _currentPage = 1;
    });
    _loadData();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YouTube 트렌드 분석'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadData,
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              decoration: const InputDecoration(
                labelText: '검색어',
                prefixIcon: Icon(Icons.search),
                border: OutlineInputBorder(),
              ),
              onSubmitted: _handleSearch,
            ),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _buildContent(),
          ),
          PaginationControls(
            currentPage: _currentPage,
            totalPages: _totalPages,
            onPageChanged: _handlePageChange,
          ),
        ],
      ),
    );
  }
  
  Widget _buildContent() {
    if (_trends.isEmpty) {
      return const Center(
        child: Text('데이터가 없습니다. 검색하거나 새로고침을 해보세요.'),
      );
    }
    
    return DefaultTabController(
      length: 2,
      child: Column(
        children: [
          const TabBar(
            tabs: [
              Tab(text: '차트'),
              Tab(text: '테이블'),
            ],
          ),
          Expanded(
            child: TabBarView(
              children: [
                TrendsChart(trends: _trends),
                TrendsTable(trends: _trends),
              ],
            ),
          ),
        ],
      ),
    );
  }
} 