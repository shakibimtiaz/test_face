import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

/// A registered face: the user's name, the MobileFaceNet embedding and a
/// small JPEG of the cropped face (for display only).
class RegisteredFace {
  final int? id;
  final String name;
  final Float32List embedding;
  final Uint8List faceJpeg;
  final DateTime createdAt;

  RegisteredFace({
    this.id,
    required this.name,
    required this.embedding,
    required this.faceJpeg,
    required this.createdAt,
  });

  Map<String, Object?> toMap() => {
        if (id != null) 'id': id,
        'name': name,
        // Store the float vector as raw bytes (BLOB).
        'embedding': embedding.buffer.asUint8List(
          embedding.offsetInBytes,
          embedding.lengthInBytes,
        ),
        'face_image': faceJpeg,
        'created_at': createdAt.toIso8601String(),
      };

  factory RegisteredFace.fromMap(Map<String, Object?> m) {
    final raw = m['embedding'] as Uint8List;
    // Copy into an aligned buffer before viewing it as Float32.
    final aligned = Uint8List.fromList(raw);
    return RegisteredFace(
      id: m['id'] as int,
      name: m['name'] as String,
      embedding: aligned.buffer.asFloat32List(),
      faceJpeg: m['face_image'] as Uint8List,
      createdAt: DateTime.parse(m['created_at'] as String),
    );
  }
}

class AttendanceRecord {
  final int id;
  final int userId;
  final String name;
  final double score;
  final DateTime timestamp;

  AttendanceRecord({
    required this.id,
    required this.userId,
    required this.name,
    required this.score,
    required this.timestamp,
  });
}

/// Local SQLite storage. Everything stays on the device — no network needed.
class DatabaseService {
  DatabaseService._();
  static final DatabaseService instance = DatabaseService._();

  Database? _db;

  Future<Database> get database async {
    if (_db != null) return _db!;
    final dir = await getDatabasesPath();
    _db = await openDatabase(
      p.join(dir, 'face_attendance.db'),
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            face_image BLOB NOT NULL,
            created_at TEXT NOT NULL
          )
        ''');
        await db.execute('''
          CREATE TABLE attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            score REAL NOT NULL,
            timestamp TEXT NOT NULL
          )
        ''');
      },
    );
    return _db!;
  }

  Future<int> insertFace(RegisteredFace face) async {
    final db = await database;
    return db.insert('users', face.toMap());
  }

  Future<List<RegisteredFace>> getAllFaces() async {
    final db = await database;
    final rows = await db.query('users', orderBy: 'created_at DESC');
    return rows.map(RegisteredFace.fromMap).toList();
  }

  Future<void> deleteFace(int id) async {
    final db = await database;
    await db.delete('users', where: 'id = ?', whereArgs: [id]);
  }

  Future<void> insertAttendance(int userId, String name, double score) async {
    final db = await database;
    await db.insert('attendance', {
      'user_id': userId,
      'name': name,
      'score': score,
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  Future<List<AttendanceRecord>> getAttendance() async {
    final db = await database;
    final rows = await db.query('attendance', orderBy: 'timestamp DESC');
    return rows
        .map((m) => AttendanceRecord(
              id: m['id'] as int,
              userId: m['user_id'] as int,
              name: m['name'] as String,
              score: (m['score'] as num).toDouble(),
              timestamp: DateTime.parse(m['timestamp'] as String),
            ))
        .toList();
  }
}
