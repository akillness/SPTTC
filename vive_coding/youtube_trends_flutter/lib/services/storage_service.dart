import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:idb_shim/idb_shim.dart';
import 'package:idb_shim/idb_browser.dart';
import '../models/database.dart';
import 'package:drift/drift.dart';

abstract class StorageService {
  Future<void> saveVideoTrend({
    required String videoId,
    required String title,
    required String channelTitle,
    required int viewCount,
    required int likeCount,
    required int commentCount,
    required DateTime publishedAt,
    required String thumbnailUrl,
    required String categoryId,
  });
  
  Future<List<Map<String, dynamic>>> getAllTrends();
  
  static StorageService create() {
    if (kIsWeb) {
      return WebStorageService();
    } else {
      return NativeStorageService();
    }
  }
}

class WebStorageService implements StorageService {
  static const String dbName = 'youtube_trends_db';
  static const int dbVersion = 1;
  static const String storeName = 'videoTrends';
  
  Future<Database> _openDb() async {
    final idbFactory = getIdbFactory();
    if (idbFactory == null) {
      throw Exception('IndexedDB is not supported in this environment');
    }
    
    return await idbFactory.open(dbName,
      version: dbVersion,
      onUpgradeNeeded: (VersionChangeEvent event) {
        final db = event.database;
        if (!db.objectStoreNames.contains(storeName)) {
          db.createObjectStore(storeName, autoIncrement: true);
        }
      });
  }

  @override
  Future<void> saveVideoTrend({
    required String videoId,
    required String title,
    required String channelTitle,
    required int viewCount,
    required int likeCount,
    required int commentCount,
    required DateTime publishedAt,
    required String thumbnailUrl,
    required String categoryId,
  }) async {
    final db = await _openDb();
    final txn = db.transaction(storeName, 'readwrite');
    final store = txn.objectStore(storeName);
    
    await store.put({
      'videoId': videoId,
      'title': title,
      'channelTitle': channelTitle,
      'viewCount': viewCount,
      'likeCount': likeCount,
      'commentCount': commentCount,
      'publishedAt': publishedAt.toIso8601String(),
      'thumbnailUrl': thumbnailUrl,
      'categoryId': categoryId,
    });
    
    await txn.completed;
    db.close();
  }

  @override
  Future<List<Map<String, dynamic>>> getAllTrends() async {
    final db = await _openDb();
    final txn = db.transaction(storeName, 'readonly');
    final store = txn.objectStore(storeName);
    
    final records = await store.getAll();
    
    await txn.completed;
    db.close();
    
    return records.map((record) {
      final Map<String, dynamic> data = record as Map<String, dynamic>;
      return {
        'videoId': data['videoId'],
        'title': data['title'],
        'channelTitle': data['channelTitle'],
        'viewCount': data['viewCount'],
        'likeCount': data['likeCount'],
        'commentCount': data['commentCount'],
        'publishedAt': DateTime.parse(data['publishedAt']),
        'thumbnailUrl': data['thumbnailUrl'],
        'categoryId': data['categoryId'],
      };
    }).toList() ?? [];
  }
}

class NativeStorageService implements StorageService {
  final AppDatabase _db;
  
  NativeStorageService() : _db = AppDatabase();
  
  @override
  Future<void> saveVideoTrend({
    required String videoId,
    required String title,
    required String channelTitle,
    required int viewCount,
    required int likeCount,
    required int commentCount,
    required DateTime publishedAt,
    required String thumbnailUrl,
    required String categoryId,
  }) async {
    await _db.insertTrend(
      VideoTrendsCompanion(
        videoId: Value(videoId),
        title: Value(title),
        channelTitle: Value(channelTitle),
        viewCount: Value(viewCount),
        likeCount: Value(likeCount),
        commentCount: Value(commentCount),
        publishedAt: Value(publishedAt),
        thumbnailUrl: Value(thumbnailUrl),
        categoryId: Value(categoryId),
      ),
    );
  }

  @override
  Future<List<Map<String, dynamic>>> getAllTrends() async {
    final trends = await _db.getAllTrends();
    return trends.map((trend) => {
      'videoId': trend.videoId,
      'title': trend.title,
      'channelTitle': trend.channelTitle,
      'viewCount': trend.viewCount,
      'likeCount': trend.likeCount,
      'commentCount': trend.commentCount,
      'publishedAt': trend.publishedAt,
      'thumbnailUrl': trend.thumbnailUrl,
      'categoryId': trend.categoryId,
    }).toList();
  }
} 