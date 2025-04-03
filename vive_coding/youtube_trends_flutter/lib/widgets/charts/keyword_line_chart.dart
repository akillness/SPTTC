import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../models/youtube_data.dart';

class KeywordLineChart extends StatelessWidget {
  final List<YoutubeData> trends;
  final int itemCount;

  const KeywordLineChart({
    super.key,
    required this.trends,
    this.itemCount = 10,
  });

  Map<String, int> _getKeywordFrequency() {
    final Map<String, int> frequency = {};
    for (final trend in trends.take(itemCount)) {
      for (final keyword in trend.keywords) {
        frequency[keyword] = (frequency[keyword] ?? 0) + 1;
      }
    }
    return Map.fromEntries(
      frequency.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value))
        ..take(10),
    );
  }

  @override
  Widget build(BuildContext context) {
    final keywordFrequency = _getKeywordFrequency();
    final keywords = keywordFrequency.keys.toList();
    final maxFrequency = keywordFrequency.values.reduce((a, b) => a > b ? a : b).toDouble();

    return SizedBox(
      height: 300,
      child: LineChart(
        LineChartData(
          gridData: FlGridData(show: false),
          titlesData: FlTitlesData(
            show: true,
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  final index = value.toInt();
                  if (index >= 0 && index < keywords.length) {
                    return Padding(
                      padding: const EdgeInsets.only(top: 8.0),
                      child: Text(
                        keywords[index],
                        style: const TextStyle(fontSize: 10),
                      ),
                    );
                  }
                  return const Text('');
                },
              ),
            ),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  return Text(
                    value.toInt().toString(),
                    style: const TextStyle(fontSize: 10),
                  );
                },
              ),
            ),
          ),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: keywords.asMap().entries.map((entry) {
                return FlSpot(
                  entry.key.toDouble(),
                  keywordFrequency[entry.value]!.toDouble(),
                );
              }).toList(),
              isCurved: true,
              color: Colors.blue,
              barWidth: 3,
              dotData: FlDotData(show: true),
            ),
          ],
        ),
      ),
    );
  }
} 