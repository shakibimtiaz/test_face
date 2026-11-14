import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/home_screen.dart';
import 'services/database_service.dart';
import 'services/face_recognition_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Portrait only: camera-frame rotation math assumes it.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  // Everything is local: SQLite DB + bundled TFLite / ML Kit models.
  await DatabaseService.instance.database;
  await FaceRecognitionService.instance.load();
  runApp(const FaceAttendanceApp());
}

class FaceAttendanceApp extends StatelessWidget {
  const FaceAttendanceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Attendance',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
