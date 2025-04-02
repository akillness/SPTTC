import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../models/trend.dart';

// Badge 클래스
class _Badge extends StatelessWidget {
  final String text;
  final double size;
  final Color borderColor;

  const _Badge(
    this.text, {
    required this.size,
    required this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: PieChart.defaultDuration,
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Colors.white,
        shape: BoxShape.circle,
        border: Border.all(
          color: borderColor,
          width: 2,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(.5),
            offset: const Offset(3, 3),
            blurRadius: 3,
          ),
        ],
      ),
      padding: EdgeInsets.all(size * .15),
      child: Center(
        child: Text(
          text.length > 5 ? text.substring(0, 3) + '...' : text,
          style: TextStyle(
            fontSize: size * .3,
            fontWeight: FontWeight.bold,
            color: borderColor,
          ),
          textAlign: TextAlign.center,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
    );
  }
}

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

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    try {
      final response = await _api.collectTrends();
      setState(() {
        _trends = response.items;
        _totalPages = response.totalPages;
        _currentPage = response.currentPage;
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildTrendsChart() {
    if (_trends.isEmpty) {
      return const Center(child: Text('데이터가 없습니다'));
    }

    final sortedTrends = List<YoutubeData>.from(_trends)
      ..sort((a, b) => b.likes.compareTo(a.likes));
    final displayTrends = sortedTrends.take(10).toList();

    return Column(
      children: [
        const Padding(
          padding: EdgeInsets.all(16.0),
          child: Text(
            "좋아요 상위 10개 트렌드",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: BarChart(
              BarChartData(
                alignment: BarChartAlignment.spaceAround,
                maxY: displayTrends.isEmpty ? 100 : displayTrends.first.likes * 1.1,
                titlesData: FlTitlesData(
                  show: true,
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      getTitlesWidget: (value, meta) {
                        if (value < 0 || value >= displayTrends.length) {
                          return const SizedBox.shrink();
                        }
                        final text = displayTrends[value.toInt()].title;
                        return Transform.rotate(
                          angle: 0.3,
                          child: Text(
                            text.length > 15 ? '${text.substring(0, 12)}...' : text,
                            style: const TextStyle(fontSize: 10),
                          ),
                        );
                      },
                      reservedSize: 40,
                    ),
                  ),
                  leftTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 60,
                      getTitlesWidget: (value, meta) {
                        return Text(NumberFormat.compact().format(value));
                      },
                    ),
                  ),
                  topTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: const AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                ),
                gridData: FlGridData(show: true),
                borderData: FlBorderData(show: false),
                barGroups: displayTrends.asMap().entries.map((entry) {
                  return BarChartGroupData(
                    x: entry.key,
                    barRods: [
                      BarChartRodData(
                        toY: entry.value.likes.toDouble(),
                        color: Colors.blue,
                        width: 16,
                      ),
                    ],
                  );
                }).toList(),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDataTable() {
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
        rows: _trends.map((trend) {
          final format = NumberFormat('#,###');
          return DataRow(
            cells: [
              DataCell(Text(trend.title)),
              DataCell(Text(format.format(trend.views))),
              DataCell(Text(format.format(trend.likes))),
              DataCell(Text(trend.keywords.join(', '))),
              DataCell(Text(trend.timestamp)),
            ],
          );
        }).toList(),
      ),
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
            onPressed: _loadData,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: DefaultTabController(
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
                              _buildTrendsChart(),
                              _buildDataTable(),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.arrow_back),
                        onPressed: _currentPage > 1
                            ? () {
                                setState(() => _currentPage--);
                                _loadData();
                              }
                            : null,
                      ),
                      Text('$_currentPage / $_totalPages'),
                      IconButton(
                        icon: const Icon(Icons.arrow_forward),
                        onPressed: _currentPage < _totalPages
                            ? () {
                                setState(() => _currentPage++);
                                _loadData();
                              }
                            : null,
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }
} 