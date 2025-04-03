import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/trends_screen.dart';
import 'models/database.dart';
import 'services/migration_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  final database = AppDatabase();
  final migrationService = MigrationService(database);
  
  try {
    await migrationService.migrateFromIndexedDB();
    print('Migration completed successfully');
  } catch (e) {
    print('Migration failed: $e');
  }
  
  runApp(MyApp(database: database));
}

class MyApp extends StatelessWidget {
  final AppDatabase database;
  
  const MyApp({super.key, required this.database});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YouTube 트렌드 분석',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4A6FFF),
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.notoSansKrTextTheme(),
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
        ),
        cardTheme: CardTheme(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        tabBarTheme: const TabBarTheme(
          labelColor: Color(0xFF4A6FFF),
          unselectedLabelColor: Color(0xFF94A3B8),
          indicatorSize: TabBarIndicatorSize.tab,
        ),
      ),
      home: const TrendsScreen(),
    );
  }
}
