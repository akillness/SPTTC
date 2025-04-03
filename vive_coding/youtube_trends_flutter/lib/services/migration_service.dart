import 'package:idb_shim/idb_shim.dart';
import 'package:idb_shim/idb_browser.dart';
import '../models/database.dart';
import 'package:drift/drift.dart';

class MigrationService {
  final AppDatabase driftDb;
  static const String dbName = 'youtube_trends_db';
  static const int dbVersion = 1;

  MigrationService(this.driftDb);

  Future<void> migrateFromIndexedDB() async {
    final idbFactory = getIdbFactory();
    if (idbFactory == null) {
      throw Exception('IndexedDB is not supported in this environment');
    }
    
    final db = await idbFactory.open(dbName,
        version: dbVersion,
        onUpgradeNeeded: (VersionChangeEvent event) {
          final db = event.database;
          if (!db.objectStoreNames.contains('videoTrends')) {
            db.createObjectStore('videoTrends', autoIncrement: true);
          }
        });

    final txn = db.transaction('videoTrends', 'readonly');
    final store = txn.objectStore('videoTrends');
    
    final records = await store.getAll();
    
    if (records.isNotEmpty) {
      final trends = records.map((dynamic record) {
        final Map<String, dynamic> data = record as Map<String, dynamic>;
        return VideoTrendsCompanion(
          videoId: Value(data['videoId'] as String),
          title: Value(data['title'] as String),
          channelTitle: Value(data['channelTitle'] as String),
          viewCount: Value(int.parse(data['viewCount'].toString())),
          likeCount: Value(int.parse(data['likeCount'].toString())),
          commentCount: Value(int.parse(data['commentCount'].toString())),
          publishedAt: Value(DateTime.parse(data['publishedAt'] as String)),
          thumbnailUrl: Value(data['thumbnailUrl'] as String),
          categoryId: Value(data['categoryId'] as String),
        );
      }).toList();

      await driftDb.insertMultipleTrends(trends);
    }

    await txn.completed;
    db.close();
  }
} 