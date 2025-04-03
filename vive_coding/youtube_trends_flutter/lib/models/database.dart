import 'package:drift/drift.dart';
import 'package:youtube_trends_flutter/models/database/connection/shared.dart' as connection;

part 'database.g.dart';

class VideoTrends extends Table {
  TextColumn get videoId => text()();
  TextColumn get title => text()();
  TextColumn get channelTitle => text()();
  IntColumn get viewCount => integer()();
  IntColumn get likeCount => integer()();
  IntColumn get commentCount => integer()();
  DateTimeColumn get publishedAt => dateTime()();
  TextColumn get thumbnailUrl => text()();
  TextColumn get categoryId => text()();
}

@DriftDatabase(tables: [VideoTrends])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(connection.openConnection());

  @override
  int get schemaVersion => 1;

  Future<int> insertTrend(VideoTrendsCompanion entry) =>
      into(videoTrends).insert(entry);
      
  Future<List<VideoTrend>> getAllTrends() => select(videoTrends).get();

  Future<void> insertMultipleTrends(List<VideoTrendsCompanion> trends) async {
    await batch((batch) {
      batch.insertAll(videoTrends, trends);
    });
  }
} 