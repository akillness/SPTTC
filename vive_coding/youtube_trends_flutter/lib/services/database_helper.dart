import 'package:idb_shim/idb_client.dart';
import 'package:idb_shim/idb_browser.dart';
import '../models/youtube_data.dart';

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static Database? _database;
  static const String dbName = 'youtube_trends.db';
  static const int dbVersion = 1;

  DatabaseHelper._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB();
    return _database!;
  }

  Future<Database> _initDB() async {
    final idbFactory = getIdbFactory()!;
    final db = await idbFactory.open(dbName,
        version: dbVersion,
        onUpgradeNeeded: (VersionChangeEvent event) {
          final db = event.database;
          // Create object stores
          db.createObjectStore('trends', keyPath: 'video_id');
          db.createObjectStore('keywords', keyPath: 'id', autoIncrement: true);
          final statsStore = db.createObjectStore('stats', keyPath: 'keyword');
          statsStore.createIndex('total_count_idx', 'total_count');
        });
    return db;
  }

  Future<void> insertTrend(YoutubeData trend) async {
    final db = await database;
    final txn = db.transaction(['trends', 'keywords', 'stats'], 'readwrite');
    
    try {
      final trendsStore = txn.objectStore('trends');
      final keywordsStore = txn.objectStore('keywords');
      final statsStore = txn.objectStore('stats');

      // Check if video already exists
      final existing = await trendsStore.getObject(trend.videoId);
      
      if (existing == null) {
        // Insert trend data
        await trendsStore.put({
          'video_id': trend.videoId,
          'title': trend.title,
          'views': trend.views,
          'likes': trend.likes,
          'timestamp': trend.timestamp,
          'collected_at': DateTime.now().toIso8601String(),
        });

        // Insert keywords and update stats
        for (final keyword in trend.keywords) {
          await keywordsStore.put({
            'video_id': trend.videoId,
            'keyword': keyword,
          });

          // Update stats
          final stat = await statsStore.getObject(keyword) as Map<String, dynamic>?;
          if (stat == null) {
            await statsStore.put({
              'keyword': keyword,
              'total_count': 1,
              'last_seen': DateTime.now().toIso8601String(),
              'avg_views': trend.views,
              'avg_likes': trend.likes,
            });
          } else {
            await statsStore.put({
              'keyword': keyword,
              'total_count': (stat['total_count'] as int) + 1,
              'last_seen': DateTime.now().toIso8601String(),
              'avg_views': ((stat['avg_views'] as num) + trend.views) / 2,
              'avg_likes': ((stat['avg_likes'] as num) + trend.likes) / 2,
            });
          }
        }
      }
      
      await txn.completed;
    } catch (e) {
      print('Error inserting trend: $e');
      rethrow;
    }
  }

  Future<List<Map<String, dynamic>>> getKeywordStats() async {
    final db = await database;
    final txn = db.transaction(['stats'], 'readonly');
    final statsStore = txn.objectStore('stats');
    final stats = await statsStore.getAll();
    await txn.completed;
    return List<Map<String, dynamic>>.from(stats)
      ..sort((a, b) => (b['total_count'] as int).compareTo(a['total_count'] as int));
  }

  Future<List<Map<String, dynamic>>> getTrendsByKeyword(String keyword) async {
    final db = await database;
    final txn = db.transaction(['trends', 'keywords'], 'readonly');
    final keywordsStore = txn.objectStore('keywords');
    final trendsStore = txn.objectStore('trends');

    final keywordEntries = await keywordsStore.getAll();
    final videoIds = (keywordEntries as List<Map<String, dynamic>>)
        .where((k) => k['keyword'] == keyword)
        .map((k) => k['video_id'] as String)
        .toSet();

    final trends = await Future.wait(
        videoIds.map((id) => trendsStore.getObject(id)));

    await txn.completed;
    return List<Map<String, dynamic>>.from(
        trends.whereType<Map<String, dynamic>>())
      ..sort((a, b) => (b['views'] as int).compareTo(a['views'] as int));
  }

  Future<List<Map<String, dynamic>>> getTopTrends({int limit = 10}) async {
    final db = await database;
    final txn = db.transaction(['trends'], 'readonly');
    final trendsStore = txn.objectStore('trends');
    final trends = await trendsStore.getAll();
    await txn.completed;
    return List<Map<String, dynamic>>.from(trends)
      ..sort((a, b) => (b['views'] as int).compareTo(a['views'] as int))
      ..take(limit)
      .toList();
  }

  Future<Map<String, dynamic>> getKeywordTrends() async {
    final db = await database;
    final txn = db.transaction(['stats'], 'readonly');
    final statsStore = txn.objectStore('stats');
    final stats = await statsStore.getAll();
    await txn.completed;

    final results = List<Map<String, dynamic>>.from(stats)
      ..sort((a, b) => (b['total_count'] as int).compareTo(a['total_count'] as int))
      ..take(20);

    return {
      'keywords': results.map((r) => r['keyword']).toList(),
      'counts': results.map((r) => r['total_count']).toList(),
      'avgViews': results.map((r) => r['avg_views']).toList(),
      'avgLikes': results.map((r) => r['avg_likes']).toList(),
    };
  }
} 