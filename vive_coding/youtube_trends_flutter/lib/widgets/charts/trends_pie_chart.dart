import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../models/youtube_data.dart';
import 'package:intl/intl.dart';

class TrendsPieChart extends StatefulWidget {
  final List<YoutubeData> trends;
  final int itemCount;

  const TrendsPieChart({
    super.key,
    required this.trends,
    this.itemCount = 5,
  });

  @override
  State<TrendsPieChart> createState() => _TrendsPieChartState();
}

class _TrendsPieChartState extends State<TrendsPieChart> with SingleTickerProviderStateMixin {
  int? touchedIndex;
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 1.1).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  static const List<Color> _colors = [
    Colors.blue,
    Colors.red,
    Colors.green,
    Colors.orange,
    Colors.purple,
    Colors.teal,
    Colors.pink,
    Colors.indigo,
    Colors.amber,
    Colors.cyan,
  ];

  @override
  Widget build(BuildContext context) {
    final topItems = widget.trends.take(widget.itemCount).toList();
    final total = topItems.fold<int>(0, (sum, item) => sum + item.views);
    
    return SizedBox(
      height: 300,
      child: Stack(
        children: [
          AnimatedBuilder(
            animation: _scaleAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: touchedIndex != null ? _scaleAnimation.value : 1.0,
                child: PieChart(
                  PieChartData(
                    sections: topItems.asMap().entries.map((entry) {
                      final percentage = (entry.value.views / total) * 100;
                      final isTouched = entry.key == touchedIndex;
                      final double radius = isTouched ? 110 : 100;
                      
                      return PieChartSectionData(
                        value: entry.value.views.toDouble(),
                        title: '${percentage.toStringAsFixed(1)}%',
                        radius: radius,
                        titleStyle: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                        color: _colors[entry.key % _colors.length],
                        borderSide: isTouched
                            ? const BorderSide(color: Colors.white, width: 2)
                            : BorderSide.none,
                        titlePositionPercentageOffset: 0.6,
                        badgeWidget: isTouched ? _buildBadge(entry.value) : null,
                        badgePositionPercentageOffset: 1.2,
                      );
                    }).toList(),
                    sectionsSpace: 2,
                    centerSpaceRadius: 40,
                    pieTouchData: PieTouchData(
                      touchCallback: (FlTouchEvent event, pieTouchResponse) {
                        setState(() {
                          if (event is FlPointerHoverEvent) {
                            touchedIndex = pieTouchResponse?.touchedSection?.touchedSectionIndex;
                            if (touchedIndex != null) {
                              _controller.forward();
                            } else {
                              _controller.reverse();
                            }
                          }
                        });
                      },
                    ),
                  ),
                ),
              );
            },
          ),
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Column(
              children: topItems.asMap().entries.map((entry) {
                final isTouched = entry.key == touchedIndex;
                return AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  decoration: BoxDecoration(
                    color: isTouched ? Colors.grey.withOpacity(0.1) : Colors.transparent,
                    borderRadius: BorderRadius.circular(4),
                    border: isTouched
                        ? Border.all(color: _colors[entry.key % _colors.length], width: 1)
                        : null,
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 16,
                        height: 16,
                        decoration: BoxDecoration(
                          color: _colors[entry.key % _colors.length],
                          shape: BoxShape.circle,
                          boxShadow: isTouched
                              ? [
                                  BoxShadow(
                                    color: _colors[entry.key % _colors.length].withOpacity(0.3),
                                    blurRadius: 4,
                                    spreadRadius: 1,
                                  )
                                ]
                              : null,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              entry.value.title,
                              style: TextStyle(
                                fontSize: isTouched ? 14 : 12,
                                fontWeight: isTouched ? FontWeight.bold : FontWeight.normal,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                            if (isTouched) ...[
                              const SizedBox(height: 4),
                              Text(
                                '키워드: ${entry.value.keywords.join(", ")}',
                                style: TextStyle(
                                  fontSize: 10,
                                  color: Colors.grey[600],
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBadge(YoutubeData data) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 4,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '조회수: ${NumberFormat.compact().format(data.views)}',
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(
            '좋아요: ${NumberFormat.compact().format(data.likes)}',
            style: const TextStyle(fontSize: 12),
          ),
        ],
      ),
    );
  }
} 