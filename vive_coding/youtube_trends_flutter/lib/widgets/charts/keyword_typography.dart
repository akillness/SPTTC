import 'package:flutter/material.dart';
import '../../models/youtube_data.dart';

class KeywordTypography extends StatelessWidget {
  final List<YoutubeData> trends;
  final int itemCount;

  const KeywordTypography({
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
        ..take(20),
    );
  }

  Color _getColorForFrequency(int frequency, int maxFrequency) {
    final ratio = frequency / maxFrequency;
    if (ratio > 0.8) {
      return Colors.red;
    } else if (ratio > 0.6) {
      return Colors.orange;
    } else if (ratio > 0.4) {
      return Colors.blue;
    } else if (ratio > 0.2) {
      return Colors.green;
    } else {
      return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    final keywordFrequency = _getKeywordFrequency();
    final maxFrequency = keywordFrequency.values.reduce((a, b) => a > b ? a : b).toDouble();

    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Wrap(
        spacing: 8.0,
        runSpacing: 8.0,
        children: keywordFrequency.entries.map((entry) {
          final fontSize = 10.0 + (entry.value / maxFrequency * 8.0);
          final color = _getColorForFrequency(entry.value, maxFrequency.toInt());
          
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: color.withOpacity(0.2),
                width: 0.5,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  entry.key,
                  style: TextStyle(
                    fontSize: fontSize,
                    fontWeight: FontWeight.w500,
                    color: color.withOpacity(0.8),
                  ),
                ),
                const SizedBox(width: 3),
                Text(
                  '${entry.value}',
                  style: TextStyle(
                    fontSize: fontSize * 0.8,
                    color: color.withOpacity(0.6),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }
} 