import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import 'package:url_launcher/url_launcher.dart';
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

  Future<void> _launchYoutubeVideo(String videoId) async {
    final Uri url = Uri.parse('https://www.youtube.com/watch?v=$videoId');
    if (await canLaunchUrl(url)) {
      await launchUrl(url);
    }
  }

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
        Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(24),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 20,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: ConstrainedBox(
                  constraints: BoxConstraints(
                    minWidth: MediaQuery.of(context).size.width - 48,
                  ),
                  child: SingleChildScrollView(
                    child: Theme(
                      data: Theme.of(context).copyWith(
                        dataTableTheme: DataTableThemeData(
                          headingTextStyle: const TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                          headingRowColor: MaterialStateProperty.all(
                            const Color(0xFF4A6FFF),
                          ),
                        ),
                      ),
                      child: DataTable(
                        columnSpacing: 24,
                        horizontalMargin: 24,
                        headingRowHeight: 56,
                        dataRowMinHeight: 64,
                        dataRowMaxHeight: 84,
                        showBottomBorder: true,
                        dividerThickness: 0.5,
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
                            if (rank <= 3) return const Color(0xFFFFB800);
                            if (rank <= 10) return const Color(0xFF4A6FFF);
                            if (rank <= 20) return const Color(0xFF00C48C);
                            return Colors.grey;
                          }

                          return DataRow(
                            color: MaterialStateProperty.resolveWith<Color?>(
                              (Set<MaterialState> states) {
                                if (states.contains(MaterialState.hovered)) {
                                  return const Color(0xFF4A6FFF).withOpacity(0.05);
                                }
                                return null;
                              },
                            ),
                            cells: [
                              DataCell(
                                Container(
                                  width: 48,
                                  height: 48,
                                  decoration: BoxDecoration(
                                    color: getRankColor().withOpacity(0.1),
                                    shape: BoxShape.circle,
                                  ),
                                  child: Center(
                                    child: Text(
                                      rank.toString(),
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: getRankColor(),
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              DataCell(
                                Container(
                                  constraints: BoxConstraints(
                                    maxWidth: MediaQuery.of(context).size.width * 0.3,
                                  ),
                                  child: InkWell(
                                    onTap: () => _launchYoutubeVideo(item.videoId),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Flexible(
                                          child: Text(
                                            item.title,
                                            style: TextStyle(
                                              fontSize: 14,
                                              fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                                              color: const Color(0xFF4A6FFF),
                                              decoration: TextDecoration.underline,
                                            ),
                                            overflow: TextOverflow.ellipsis,
                                            maxLines: 2,
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        Container(
                                          padding: const EdgeInsets.all(4),
                                          decoration: BoxDecoration(
                                            color: const Color(0xFF4A6FFF).withOpacity(0.1),
                                            borderRadius: BorderRadius.circular(4),
                                          ),
                                          child: const Icon(
                                            Icons.launch,
                                            size: 16,
                                            color: Color(0xFF4A6FFF),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              DataCell(
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFF4A6FFF).withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: const Icon(
                                        Icons.visibility,
                                        size: 16,
                                        color: Color(0xFF4A6FFF),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      NumberFormat.compact().format(item.views),
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                                        color: const Color(0xFF4A6FFF),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              DataCell(
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFFF4A4A).withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: const Icon(
                                        Icons.favorite,
                                        size: 16,
                                        color: Color(0xFFFF4A4A),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Text(
                                      NumberFormat.compact().format(item.likes),
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: rank <= 10 ? FontWeight.bold : FontWeight.normal,
                                        color: const Color(0xFFFF4A4A),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              DataCell(
                                Container(
                                  constraints: BoxConstraints(
                                    maxWidth: MediaQuery.of(context).size.width * 0.2,
                                  ),
                                  child: Wrap(
                                    spacing: 4,
                                    runSpacing: 4,
                                    children: item.keywords.map((keyword) {
                                      return Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 12,
                                          vertical: 6,
                                        ),
                                        decoration: BoxDecoration(
                                          gradient: const LinearGradient(
                                            colors: [
                                              Color(0xFF4A6FFF),
                                              Color(0xFF6B8AFF),
                                            ],
                                          ),
                                          borderRadius: BorderRadius.circular(16),
                                          boxShadow: [
                                            BoxShadow(
                                              color: const Color(0xFF4A6FFF).withOpacity(0.2),
                                              blurRadius: 4,
                                              offset: const Offset(0, 2),
                                            ),
                                          ],
                                        ),
                                        child: Text(
                                          keyword,
                                          style: const TextStyle(
                                            fontSize: 12,
                                            fontWeight: FontWeight.w500,
                                            color: Colors.white,
                                          ),
                                        ),
                                      );
                                    }).toList(),
                                  ),
                                ),
                              ),
                              DataCell(
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 12,
                                    vertical: 6,
                                  ),
                                  decoration: BoxDecoration(
                                    color: Colors.grey.shade100,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.access_time,
                                        size: 16,
                                        color: Colors.grey.shade600,
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        item.timestamp,
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: Colors.grey.shade600,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          );
                        }).toList(),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
        Container(
          margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: PaginationControls(
            currentPage: _currentPage,
            totalPages: _totalPages,
            onPageChanged: _handlePageChange,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFFF8F9FF),
              Color(0xFFFFFFFF),
            ],
          ),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(24, 48, 24, 24),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Color(0xFF4A6FFF),
                    Color(0xFF6B8AFF),
                  ],
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Text(
                        'YouTube 트렌드 분석',
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      const Spacer(),
                      IconButton(
                        icon: const Icon(Icons.refresh, color: Colors.white),
                        onPressed: _loadTrends,
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  style: const TextStyle(color: Colors.white),
                                  decoration: InputDecoration(
                                    hintText: '검색어를 입력하세요',
                                    hintStyle: TextStyle(
                                      color: Colors.white.withOpacity(0.7),
                                    ),
                                    border: InputBorder.none,
                                    contentPadding: const EdgeInsets.symmetric(
                                      horizontal: 16,
                                      vertical: 12,
                                    ),
                                    prefixIcon: Icon(
                                      Icons.search,
                                      color: Colors.white.withOpacity(0.7),
                                    ),
                                  ),
                                  onChanged: (value) => _searchQuery = value,
                                  onSubmitted: (_) => _searchTrends(),
                                ),
                              ),
                              Container(
                                margin: const EdgeInsets.only(right: 4),
                                child: ElevatedButton.icon(
                                  onPressed: _searchTrends,
                                  icon: const Icon(Icons.search),
                                  label: const Text('검색'),
                                  style: ElevatedButton.styleFrom(
                                    foregroundColor: const Color(0xFF4A6FFF),
                                    backgroundColor: Colors.white,
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 20,
                                      vertical: 12,
                                    ),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      ElevatedButton.icon(
                        onPressed: _loadTrends,
                        icon: const Icon(Icons.trending_up),
                        label: const Text('트렌드 검색'),
                        style: ElevatedButton.styleFrom(
                          foregroundColor: const Color(0xFF4A6FFF),
                          backgroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 12,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            if (_isLoading)
              const Expanded(
                child: Center(
                  child: CircularProgressIndicator(
                    color: Color(0xFF4A6FFF),
                  ),
                ),
              )
            else
              Expanded(
                child: DefaultTabController(
                  length: 2,
                  child: Column(
                    children: [
                      Container(
                        margin: const EdgeInsets.fromLTRB(24, 0, 24, 16),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.05),
                              blurRadius: 10,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                        child: TabBar(
                          indicator: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [
                                Color(0xFF4A6FFF),
                                Color(0xFF6B8AFF),
                              ],
                            ),
                            borderRadius: BorderRadius.circular(16),
                          ),
                          labelColor: Colors.white,
                          unselectedLabelColor: Colors.grey.shade600,
                          labelStyle: const TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                          tabs: const [
                            Tab(
                              icon: Icon(Icons.table_chart),
                              text: '데이터 테이블',
                            ),
                            Tab(
                              icon: Icon(Icons.analytics),
                              text: '차트 분석',
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _buildDataTable(),
                            Column(
                              children: [
                                Expanded(
                                  flex: 3,
                                  child: Container(
                                    margin: const EdgeInsets.all(16),
                                    decoration: BoxDecoration(
                                      color: Colors.white,
                                      borderRadius: BorderRadius.circular(24),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.05),
                                          blurRadius: 20,
                                          offset: const Offset(0, 4),
                                        ),
                                      ],
                                    ),
                                    child: TrendsPieChart(trends: _trends),
                                  ),
                                ),
                                Expanded(
                                  flex: 2,
                                  child: Container(
                                    margin: const EdgeInsets.all(16),
                                    decoration: BoxDecoration(
                                      color: Colors.white,
                                      borderRadius: BorderRadius.circular(24),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.05),
                                          blurRadius: 20,
                                          offset: const Offset(0, 4),
                                        ),
                                      ],
                                    ),
                                    child: KeywordTypography(trends: _trends),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
} 