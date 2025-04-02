import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import '../models/youtube_data.dart';

class TrendsChart extends StatelessWidget {
  final List<YoutubeData> trends;
  
  const TrendsChart({super.key, required this.trends});
  
  @override
  Widget build(BuildContext context) {
    if (trends.isEmpty) {
      return const Center(child: Text('데이터가 없습니다'));
    }

    final sortedTrends = List<YoutubeData>.from(trends)
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
                titlesData: _getTitlesData(displayTrends),
                gridData: FlGridData(show: true),
                borderData: FlBorderData(show: false),
                barGroups: _createBarGroups(displayTrends),
              ),
            ),
          ),
        ),
      ],
    );
  }
  
  FlTitlesData _getTitlesData(List<YoutubeData> displayTrends) {
    return FlTitlesData(
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
    );
  }
  
  List<BarChartGroupData> _createBarGroups(List<YoutubeData> displayTrends) {
    return displayTrends.asMap().entries.map((entry) {
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
    }).toList();
  }
} 