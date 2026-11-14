import 'package:flutter/material.dart';

import '../services/database_service.dart';

class AttendanceScreen extends StatelessWidget {
  const AttendanceScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Attendance Log')),
      body: FutureBuilder<List<AttendanceRecord>>(
        future: DatabaseService.instance.getAttendance(),
        builder: (context, snap) {
          if (!snap.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final records = snap.data!;
          if (records.isEmpty) {
            return const Center(child: Text('No attendance recorded yet'));
          }
          return ListView.separated(
            itemCount: records.length,
            separatorBuilder: (_, _) => const Divider(height: 1),
            itemBuilder: (_, i) {
              final r = records[i];
              return ListTile(
                leading: const Icon(Icons.check_circle, color: Colors.green),
                title: Text(r.name),
                subtitle: Text(r.timestamp.toLocal().toString().substring(0, 19)),
                trailing: Text(r.score.toStringAsFixed(2)),
              );
            },
          );
        },
      ),
    );
  }
}
