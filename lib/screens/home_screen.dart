import 'package:flutter/material.dart';

import '../services/database_service.dart';
import '../services/face_recognition_service.dart';
import 'attendance_screen.dart';
import 'authenticate_screen.dart';
import 'register_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<RegisteredFace> _faces = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final faces = await DatabaseService.instance.getAllFaces();
    if (mounted) setState(() => _faces = faces);
  }

  Future<void> _open(Widget screen) async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
    _load();
  }

  Future<void> _delete(RegisteredFace f) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete face?'),
        content: Text('Remove "${f.name}" from the database?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (ok == true) {
      await DatabaseService.instance.deleteFace(f.id!);
      _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Attendance'),
        actions: [
          IconButton(
            tooltip: 'Attendance log',
            icon: const Icon(Icons.history),
            onPressed: () => _open(const AttendanceScreen()),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: FilledButton.icon(
                onPressed: () => _open(const RegisterScreen()),
                icon: const Icon(Icons.person_add),
                label: const Text('Register Face'),
                style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 18)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: FilledButton.tonalIcon(
                onPressed: () => _open(const AuthenticateScreen()),
                icon: const Icon(Icons.face_retouching_natural),
                label: const Text('Authenticate'),
                style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 18)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 24, 16, 8),
              child: Text('Registered faces (${_faces.length})',
                  style: Theme.of(context).textTheme.titleMedium),
            ),
            Expanded(
              child: _faces.isEmpty
                  ? const Center(child: Text('No faces registered yet'))
                  : ListView.builder(
                      itemCount: _faces.length,
                      itemBuilder: (_, i) {
                        final f = _faces[i];
                        final outdated = f.embeddingVersion !=
                            FaceRecognitionService.embeddingVersion;
                        return ListTile(
                          leading: CircleAvatar(
                              backgroundImage: MemoryImage(f.faceJpeg)),
                          title: Text(f.name),
                          subtitle: outdated
                              ? Text(
                                  'Re-register needed (old format, '
                                  'ignored when authenticating)',
                                  style: TextStyle(
                                      color:
                                          Theme.of(context).colorScheme.error),
                                )
                              : Text(
                                  'Registered ${f.createdAt.toLocal().toString().substring(0, 16)}'),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete_outline),
                            onPressed: () => _delete(f),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
