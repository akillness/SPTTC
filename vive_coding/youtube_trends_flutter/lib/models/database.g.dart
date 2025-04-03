// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $VideoTrendsTable extends VideoTrends
    with TableInfo<$VideoTrendsTable, VideoTrend> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $VideoTrendsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _videoIdMeta =
      const VerificationMeta('videoId');
  @override
  late final GeneratedColumn<String> videoId = GeneratedColumn<String>(
      'video_id', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _titleMeta = const VerificationMeta('title');
  @override
  late final GeneratedColumn<String> title = GeneratedColumn<String>(
      'title', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _channelTitleMeta =
      const VerificationMeta('channelTitle');
  @override
  late final GeneratedColumn<String> channelTitle = GeneratedColumn<String>(
      'channel_title', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _viewCountMeta =
      const VerificationMeta('viewCount');
  @override
  late final GeneratedColumn<int> viewCount = GeneratedColumn<int>(
      'view_count', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _likeCountMeta =
      const VerificationMeta('likeCount');
  @override
  late final GeneratedColumn<int> likeCount = GeneratedColumn<int>(
      'like_count', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _commentCountMeta =
      const VerificationMeta('commentCount');
  @override
  late final GeneratedColumn<int> commentCount = GeneratedColumn<int>(
      'comment_count', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _publishedAtMeta =
      const VerificationMeta('publishedAt');
  @override
  late final GeneratedColumn<DateTime> publishedAt = GeneratedColumn<DateTime>(
      'published_at', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  static const VerificationMeta _thumbnailUrlMeta =
      const VerificationMeta('thumbnailUrl');
  @override
  late final GeneratedColumn<String> thumbnailUrl = GeneratedColumn<String>(
      'thumbnail_url', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _categoryIdMeta =
      const VerificationMeta('categoryId');
  @override
  late final GeneratedColumn<String> categoryId = GeneratedColumn<String>(
      'category_id', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  @override
  List<GeneratedColumn> get $columns => [
        videoId,
        title,
        channelTitle,
        viewCount,
        likeCount,
        commentCount,
        publishedAt,
        thumbnailUrl,
        categoryId
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'video_trends';
  @override
  VerificationContext validateIntegrity(Insertable<VideoTrend> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('video_id')) {
      context.handle(_videoIdMeta,
          videoId.isAcceptableOrUnknown(data['video_id']!, _videoIdMeta));
    } else if (isInserting) {
      context.missing(_videoIdMeta);
    }
    if (data.containsKey('title')) {
      context.handle(
          _titleMeta, title.isAcceptableOrUnknown(data['title']!, _titleMeta));
    } else if (isInserting) {
      context.missing(_titleMeta);
    }
    if (data.containsKey('channel_title')) {
      context.handle(
          _channelTitleMeta,
          channelTitle.isAcceptableOrUnknown(
              data['channel_title']!, _channelTitleMeta));
    } else if (isInserting) {
      context.missing(_channelTitleMeta);
    }
    if (data.containsKey('view_count')) {
      context.handle(_viewCountMeta,
          viewCount.isAcceptableOrUnknown(data['view_count']!, _viewCountMeta));
    } else if (isInserting) {
      context.missing(_viewCountMeta);
    }
    if (data.containsKey('like_count')) {
      context.handle(_likeCountMeta,
          likeCount.isAcceptableOrUnknown(data['like_count']!, _likeCountMeta));
    } else if (isInserting) {
      context.missing(_likeCountMeta);
    }
    if (data.containsKey('comment_count')) {
      context.handle(
          _commentCountMeta,
          commentCount.isAcceptableOrUnknown(
              data['comment_count']!, _commentCountMeta));
    } else if (isInserting) {
      context.missing(_commentCountMeta);
    }
    if (data.containsKey('published_at')) {
      context.handle(
          _publishedAtMeta,
          publishedAt.isAcceptableOrUnknown(
              data['published_at']!, _publishedAtMeta));
    } else if (isInserting) {
      context.missing(_publishedAtMeta);
    }
    if (data.containsKey('thumbnail_url')) {
      context.handle(
          _thumbnailUrlMeta,
          thumbnailUrl.isAcceptableOrUnknown(
              data['thumbnail_url']!, _thumbnailUrlMeta));
    } else if (isInserting) {
      context.missing(_thumbnailUrlMeta);
    }
    if (data.containsKey('category_id')) {
      context.handle(
          _categoryIdMeta,
          categoryId.isAcceptableOrUnknown(
              data['category_id']!, _categoryIdMeta));
    } else if (isInserting) {
      context.missing(_categoryIdMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => const {};
  @override
  VideoTrend map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return VideoTrend(
      videoId: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}video_id'])!,
      title: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}title'])!,
      channelTitle: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}channel_title'])!,
      viewCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}view_count'])!,
      likeCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}like_count'])!,
      commentCount: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}comment_count'])!,
      publishedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}published_at'])!,
      thumbnailUrl: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}thumbnail_url'])!,
      categoryId: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}category_id'])!,
    );
  }

  @override
  $VideoTrendsTable createAlias(String alias) {
    return $VideoTrendsTable(attachedDatabase, alias);
  }
}

class VideoTrend extends DataClass implements Insertable<VideoTrend> {
  final String videoId;
  final String title;
  final String channelTitle;
  final int viewCount;
  final int likeCount;
  final int commentCount;
  final DateTime publishedAt;
  final String thumbnailUrl;
  final String categoryId;
  const VideoTrend(
      {required this.videoId,
      required this.title,
      required this.channelTitle,
      required this.viewCount,
      required this.likeCount,
      required this.commentCount,
      required this.publishedAt,
      required this.thumbnailUrl,
      required this.categoryId});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['video_id'] = Variable<String>(videoId);
    map['title'] = Variable<String>(title);
    map['channel_title'] = Variable<String>(channelTitle);
    map['view_count'] = Variable<int>(viewCount);
    map['like_count'] = Variable<int>(likeCount);
    map['comment_count'] = Variable<int>(commentCount);
    map['published_at'] = Variable<DateTime>(publishedAt);
    map['thumbnail_url'] = Variable<String>(thumbnailUrl);
    map['category_id'] = Variable<String>(categoryId);
    return map;
  }

  VideoTrendsCompanion toCompanion(bool nullToAbsent) {
    return VideoTrendsCompanion(
      videoId: Value(videoId),
      title: Value(title),
      channelTitle: Value(channelTitle),
      viewCount: Value(viewCount),
      likeCount: Value(likeCount),
      commentCount: Value(commentCount),
      publishedAt: Value(publishedAt),
      thumbnailUrl: Value(thumbnailUrl),
      categoryId: Value(categoryId),
    );
  }

  factory VideoTrend.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return VideoTrend(
      videoId: serializer.fromJson<String>(json['videoId']),
      title: serializer.fromJson<String>(json['title']),
      channelTitle: serializer.fromJson<String>(json['channelTitle']),
      viewCount: serializer.fromJson<int>(json['viewCount']),
      likeCount: serializer.fromJson<int>(json['likeCount']),
      commentCount: serializer.fromJson<int>(json['commentCount']),
      publishedAt: serializer.fromJson<DateTime>(json['publishedAt']),
      thumbnailUrl: serializer.fromJson<String>(json['thumbnailUrl']),
      categoryId: serializer.fromJson<String>(json['categoryId']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'videoId': serializer.toJson<String>(videoId),
      'title': serializer.toJson<String>(title),
      'channelTitle': serializer.toJson<String>(channelTitle),
      'viewCount': serializer.toJson<int>(viewCount),
      'likeCount': serializer.toJson<int>(likeCount),
      'commentCount': serializer.toJson<int>(commentCount),
      'publishedAt': serializer.toJson<DateTime>(publishedAt),
      'thumbnailUrl': serializer.toJson<String>(thumbnailUrl),
      'categoryId': serializer.toJson<String>(categoryId),
    };
  }

  VideoTrend copyWith(
          {String? videoId,
          String? title,
          String? channelTitle,
          int? viewCount,
          int? likeCount,
          int? commentCount,
          DateTime? publishedAt,
          String? thumbnailUrl,
          String? categoryId}) =>
      VideoTrend(
        videoId: videoId ?? this.videoId,
        title: title ?? this.title,
        channelTitle: channelTitle ?? this.channelTitle,
        viewCount: viewCount ?? this.viewCount,
        likeCount: likeCount ?? this.likeCount,
        commentCount: commentCount ?? this.commentCount,
        publishedAt: publishedAt ?? this.publishedAt,
        thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
        categoryId: categoryId ?? this.categoryId,
      );
  VideoTrend copyWithCompanion(VideoTrendsCompanion data) {
    return VideoTrend(
      videoId: data.videoId.present ? data.videoId.value : this.videoId,
      title: data.title.present ? data.title.value : this.title,
      channelTitle: data.channelTitle.present
          ? data.channelTitle.value
          : this.channelTitle,
      viewCount: data.viewCount.present ? data.viewCount.value : this.viewCount,
      likeCount: data.likeCount.present ? data.likeCount.value : this.likeCount,
      commentCount: data.commentCount.present
          ? data.commentCount.value
          : this.commentCount,
      publishedAt:
          data.publishedAt.present ? data.publishedAt.value : this.publishedAt,
      thumbnailUrl: data.thumbnailUrl.present
          ? data.thumbnailUrl.value
          : this.thumbnailUrl,
      categoryId:
          data.categoryId.present ? data.categoryId.value : this.categoryId,
    );
  }

  @override
  String toString() {
    return (StringBuffer('VideoTrend(')
          ..write('videoId: $videoId, ')
          ..write('title: $title, ')
          ..write('channelTitle: $channelTitle, ')
          ..write('viewCount: $viewCount, ')
          ..write('likeCount: $likeCount, ')
          ..write('commentCount: $commentCount, ')
          ..write('publishedAt: $publishedAt, ')
          ..write('thumbnailUrl: $thumbnailUrl, ')
          ..write('categoryId: $categoryId')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(videoId, title, channelTitle, viewCount,
      likeCount, commentCount, publishedAt, thumbnailUrl, categoryId);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is VideoTrend &&
          other.videoId == this.videoId &&
          other.title == this.title &&
          other.channelTitle == this.channelTitle &&
          other.viewCount == this.viewCount &&
          other.likeCount == this.likeCount &&
          other.commentCount == this.commentCount &&
          other.publishedAt == this.publishedAt &&
          other.thumbnailUrl == this.thumbnailUrl &&
          other.categoryId == this.categoryId);
}

class VideoTrendsCompanion extends UpdateCompanion<VideoTrend> {
  final Value<String> videoId;
  final Value<String> title;
  final Value<String> channelTitle;
  final Value<int> viewCount;
  final Value<int> likeCount;
  final Value<int> commentCount;
  final Value<DateTime> publishedAt;
  final Value<String> thumbnailUrl;
  final Value<String> categoryId;
  final Value<int> rowid;
  const VideoTrendsCompanion({
    this.videoId = const Value.absent(),
    this.title = const Value.absent(),
    this.channelTitle = const Value.absent(),
    this.viewCount = const Value.absent(),
    this.likeCount = const Value.absent(),
    this.commentCount = const Value.absent(),
    this.publishedAt = const Value.absent(),
    this.thumbnailUrl = const Value.absent(),
    this.categoryId = const Value.absent(),
    this.rowid = const Value.absent(),
  });
  VideoTrendsCompanion.insert({
    required String videoId,
    required String title,
    required String channelTitle,
    required int viewCount,
    required int likeCount,
    required int commentCount,
    required DateTime publishedAt,
    required String thumbnailUrl,
    required String categoryId,
    this.rowid = const Value.absent(),
  })  : videoId = Value(videoId),
        title = Value(title),
        channelTitle = Value(channelTitle),
        viewCount = Value(viewCount),
        likeCount = Value(likeCount),
        commentCount = Value(commentCount),
        publishedAt = Value(publishedAt),
        thumbnailUrl = Value(thumbnailUrl),
        categoryId = Value(categoryId);
  static Insertable<VideoTrend> custom({
    Expression<String>? videoId,
    Expression<String>? title,
    Expression<String>? channelTitle,
    Expression<int>? viewCount,
    Expression<int>? likeCount,
    Expression<int>? commentCount,
    Expression<DateTime>? publishedAt,
    Expression<String>? thumbnailUrl,
    Expression<String>? categoryId,
    Expression<int>? rowid,
  }) {
    return RawValuesInsertable({
      if (videoId != null) 'video_id': videoId,
      if (title != null) 'title': title,
      if (channelTitle != null) 'channel_title': channelTitle,
      if (viewCount != null) 'view_count': viewCount,
      if (likeCount != null) 'like_count': likeCount,
      if (commentCount != null) 'comment_count': commentCount,
      if (publishedAt != null) 'published_at': publishedAt,
      if (thumbnailUrl != null) 'thumbnail_url': thumbnailUrl,
      if (categoryId != null) 'category_id': categoryId,
      if (rowid != null) 'rowid': rowid,
    });
  }

  VideoTrendsCompanion copyWith(
      {Value<String>? videoId,
      Value<String>? title,
      Value<String>? channelTitle,
      Value<int>? viewCount,
      Value<int>? likeCount,
      Value<int>? commentCount,
      Value<DateTime>? publishedAt,
      Value<String>? thumbnailUrl,
      Value<String>? categoryId,
      Value<int>? rowid}) {
    return VideoTrendsCompanion(
      videoId: videoId ?? this.videoId,
      title: title ?? this.title,
      channelTitle: channelTitle ?? this.channelTitle,
      viewCount: viewCount ?? this.viewCount,
      likeCount: likeCount ?? this.likeCount,
      commentCount: commentCount ?? this.commentCount,
      publishedAt: publishedAt ?? this.publishedAt,
      thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
      categoryId: categoryId ?? this.categoryId,
      rowid: rowid ?? this.rowid,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (videoId.present) {
      map['video_id'] = Variable<String>(videoId.value);
    }
    if (title.present) {
      map['title'] = Variable<String>(title.value);
    }
    if (channelTitle.present) {
      map['channel_title'] = Variable<String>(channelTitle.value);
    }
    if (viewCount.present) {
      map['view_count'] = Variable<int>(viewCount.value);
    }
    if (likeCount.present) {
      map['like_count'] = Variable<int>(likeCount.value);
    }
    if (commentCount.present) {
      map['comment_count'] = Variable<int>(commentCount.value);
    }
    if (publishedAt.present) {
      map['published_at'] = Variable<DateTime>(publishedAt.value);
    }
    if (thumbnailUrl.present) {
      map['thumbnail_url'] = Variable<String>(thumbnailUrl.value);
    }
    if (categoryId.present) {
      map['category_id'] = Variable<String>(categoryId.value);
    }
    if (rowid.present) {
      map['rowid'] = Variable<int>(rowid.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('VideoTrendsCompanion(')
          ..write('videoId: $videoId, ')
          ..write('title: $title, ')
          ..write('channelTitle: $channelTitle, ')
          ..write('viewCount: $viewCount, ')
          ..write('likeCount: $likeCount, ')
          ..write('commentCount: $commentCount, ')
          ..write('publishedAt: $publishedAt, ')
          ..write('thumbnailUrl: $thumbnailUrl, ')
          ..write('categoryId: $categoryId, ')
          ..write('rowid: $rowid')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $VideoTrendsTable videoTrends = $VideoTrendsTable(this);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [videoTrends];
}

typedef $$VideoTrendsTableCreateCompanionBuilder = VideoTrendsCompanion
    Function({
  required String videoId,
  required String title,
  required String channelTitle,
  required int viewCount,
  required int likeCount,
  required int commentCount,
  required DateTime publishedAt,
  required String thumbnailUrl,
  required String categoryId,
  Value<int> rowid,
});
typedef $$VideoTrendsTableUpdateCompanionBuilder = VideoTrendsCompanion
    Function({
  Value<String> videoId,
  Value<String> title,
  Value<String> channelTitle,
  Value<int> viewCount,
  Value<int> likeCount,
  Value<int> commentCount,
  Value<DateTime> publishedAt,
  Value<String> thumbnailUrl,
  Value<String> categoryId,
  Value<int> rowid,
});

class $$VideoTrendsTableFilterComposer
    extends Composer<_$AppDatabase, $VideoTrendsTable> {
  $$VideoTrendsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<String> get videoId => $composableBuilder(
      column: $table.videoId, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get title => $composableBuilder(
      column: $table.title, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get channelTitle => $composableBuilder(
      column: $table.channelTitle, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get viewCount => $composableBuilder(
      column: $table.viewCount, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get likeCount => $composableBuilder(
      column: $table.likeCount, builder: (column) => ColumnFilters(column));

  ColumnFilters<int> get commentCount => $composableBuilder(
      column: $table.commentCount, builder: (column) => ColumnFilters(column));

  ColumnFilters<DateTime> get publishedAt => $composableBuilder(
      column: $table.publishedAt, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get thumbnailUrl => $composableBuilder(
      column: $table.thumbnailUrl, builder: (column) => ColumnFilters(column));

  ColumnFilters<String> get categoryId => $composableBuilder(
      column: $table.categoryId, builder: (column) => ColumnFilters(column));
}

class $$VideoTrendsTableOrderingComposer
    extends Composer<_$AppDatabase, $VideoTrendsTable> {
  $$VideoTrendsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<String> get videoId => $composableBuilder(
      column: $table.videoId, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get title => $composableBuilder(
      column: $table.title, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get channelTitle => $composableBuilder(
      column: $table.channelTitle,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get viewCount => $composableBuilder(
      column: $table.viewCount, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get likeCount => $composableBuilder(
      column: $table.likeCount, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<int> get commentCount => $composableBuilder(
      column: $table.commentCount,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<DateTime> get publishedAt => $composableBuilder(
      column: $table.publishedAt, builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get thumbnailUrl => $composableBuilder(
      column: $table.thumbnailUrl,
      builder: (column) => ColumnOrderings(column));

  ColumnOrderings<String> get categoryId => $composableBuilder(
      column: $table.categoryId, builder: (column) => ColumnOrderings(column));
}

class $$VideoTrendsTableAnnotationComposer
    extends Composer<_$AppDatabase, $VideoTrendsTable> {
  $$VideoTrendsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<String> get videoId =>
      $composableBuilder(column: $table.videoId, builder: (column) => column);

  GeneratedColumn<String> get title =>
      $composableBuilder(column: $table.title, builder: (column) => column);

  GeneratedColumn<String> get channelTitle => $composableBuilder(
      column: $table.channelTitle, builder: (column) => column);

  GeneratedColumn<int> get viewCount =>
      $composableBuilder(column: $table.viewCount, builder: (column) => column);

  GeneratedColumn<int> get likeCount =>
      $composableBuilder(column: $table.likeCount, builder: (column) => column);

  GeneratedColumn<int> get commentCount => $composableBuilder(
      column: $table.commentCount, builder: (column) => column);

  GeneratedColumn<DateTime> get publishedAt => $composableBuilder(
      column: $table.publishedAt, builder: (column) => column);

  GeneratedColumn<String> get thumbnailUrl => $composableBuilder(
      column: $table.thumbnailUrl, builder: (column) => column);

  GeneratedColumn<String> get categoryId => $composableBuilder(
      column: $table.categoryId, builder: (column) => column);
}

class $$VideoTrendsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $VideoTrendsTable,
    VideoTrend,
    $$VideoTrendsTableFilterComposer,
    $$VideoTrendsTableOrderingComposer,
    $$VideoTrendsTableAnnotationComposer,
    $$VideoTrendsTableCreateCompanionBuilder,
    $$VideoTrendsTableUpdateCompanionBuilder,
    (VideoTrend, BaseReferences<_$AppDatabase, $VideoTrendsTable, VideoTrend>),
    VideoTrend,
    PrefetchHooks Function()> {
  $$VideoTrendsTableTableManager(_$AppDatabase db, $VideoTrendsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$VideoTrendsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$VideoTrendsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$VideoTrendsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback: ({
            Value<String> videoId = const Value.absent(),
            Value<String> title = const Value.absent(),
            Value<String> channelTitle = const Value.absent(),
            Value<int> viewCount = const Value.absent(),
            Value<int> likeCount = const Value.absent(),
            Value<int> commentCount = const Value.absent(),
            Value<DateTime> publishedAt = const Value.absent(),
            Value<String> thumbnailUrl = const Value.absent(),
            Value<String> categoryId = const Value.absent(),
            Value<int> rowid = const Value.absent(),
          }) =>
              VideoTrendsCompanion(
            videoId: videoId,
            title: title,
            channelTitle: channelTitle,
            viewCount: viewCount,
            likeCount: likeCount,
            commentCount: commentCount,
            publishedAt: publishedAt,
            thumbnailUrl: thumbnailUrl,
            categoryId: categoryId,
            rowid: rowid,
          ),
          createCompanionCallback: ({
            required String videoId,
            required String title,
            required String channelTitle,
            required int viewCount,
            required int likeCount,
            required int commentCount,
            required DateTime publishedAt,
            required String thumbnailUrl,
            required String categoryId,
            Value<int> rowid = const Value.absent(),
          }) =>
              VideoTrendsCompanion.insert(
            videoId: videoId,
            title: title,
            channelTitle: channelTitle,
            viewCount: viewCount,
            likeCount: likeCount,
            commentCount: commentCount,
            publishedAt: publishedAt,
            thumbnailUrl: thumbnailUrl,
            categoryId: categoryId,
            rowid: rowid,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ));
}

typedef $$VideoTrendsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $VideoTrendsTable,
    VideoTrend,
    $$VideoTrendsTableFilterComposer,
    $$VideoTrendsTableOrderingComposer,
    $$VideoTrendsTableAnnotationComposer,
    $$VideoTrendsTableCreateCompanionBuilder,
    $$VideoTrendsTableUpdateCompanionBuilder,
    (VideoTrend, BaseReferences<_$AppDatabase, $VideoTrendsTable, VideoTrend>),
    VideoTrend,
    PrefetchHooks Function()>;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$VideoTrendsTableTableManager get videoTrends =>
      $$VideoTrendsTableTableManager(_db, _db.videoTrends);
}
