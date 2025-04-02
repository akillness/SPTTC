import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/youtube_data.dart';

class TrendsTable extends StatelessWidget {
  final List<YoutubeData> trends;
  final _formatter = NumberFormat('#,###');
  
  TrendsTable({super.key, required this.trends});
  
  @override
  Widget build(BuildContext context) {
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
        rows: trends.map(_buildDataRow).toList(),
      ),
    );
  }
  
  DataRow _buildDataRow(YoutubeData trend) {
    return DataRow(
      cells: [
        DataCell(Text(trend.title)),
        DataCell(Text(_formatter.format(trend.views))),
        DataCell(Text(_formatter.format(trend.likes))),
        DataCell(Text(trend.keywords.join(', '))),
        DataCell(Text(trend.timestamp)),
      ],
    );
  }
} 