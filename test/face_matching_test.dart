import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:test_face/services/database_service.dart';
import 'package:test_face/services/face_recognition_service.dart';
import 'package:test_face/utils/face_align.dart';

Face _face(Map<FaceLandmarkType, Point<int>> pts) => Face(
      boundingBox: const Rect.fromLTWH(0, 0, 200, 200),
      landmarks: {
        for (final e in pts.entries)
          e.key: FaceLandmark(type: e.key, position: e.value),
      },
      contours: const {},
    );

RegisteredFace _reg(int id, String name, List<double> v) => RegisteredFace(
      id: id,
      name: name,
      embedding: _norm(v),
      faceJpeg: Uint8List(0),
      createdAt: DateTime(2026),
      embeddingVersion: FaceRecognitionService.embeddingVersion,
    );

Float32List _norm(List<double> v) {
  final n = sqrt(v.fold<double>(0, (s, x) => s + x * x));
  return Float32List.fromList(v.map((x) => x / n).toList());
}

void main() {
  group('FaceAlignment', () {
    test('maps the template back onto the landmarks (tilted, scaled face)', () {
      // Template points rotated 20°, scaled ×2.5 and shifted — like a tilted
      // face further from the camera.
      const tpl = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
      ];
      const ang = 20 * pi / 180, sc = 2.5, ox = 300.0, oy = 150.0;
      Point<int> tr(List<double> p) => Point(
            (sc * (p[0] * cos(ang) - p[1] * sin(ang)) + ox).round(),
            (sc * (p[0] * sin(ang) + p[1] * cos(ang)) + oy).round(),
          );
      // ML Kit names are from the subject's view (left eye on image right);
      // the aligner sorts by x, so naming must not matter.
      final face = _face({
        FaceLandmarkType.rightEye: tr(tpl[0]),
        FaceLandmarkType.leftEye: tr(tpl[1]),
        FaceLandmarkType.noseBase: tr(tpl[2]),
        FaceLandmarkType.rightMouth: tr(tpl[3]),
        FaceLandmarkType.leftMouth: tr(tpl[4]),
      });

      final al = FaceAlignment.fromFace(face, 112)!;
      for (final p in tpl) {
        final want = tr(p);
        expect(al.srcX(p[0], p[1]), closeTo(want.x, 1.0));
        expect(al.srcY(p[0], p[1]), closeTo(want.y, 1.0));
      }
    });

    test('returns null without eye landmarks', () {
      expect(
        FaceAlignment.fromFace(
            _face({FaceLandmarkType.noseBase: const Point(10, 10)}), 112),
        isNull,
      );
    });
  });

  group('identify', () {
    test('accepts a clear winner', () {
      final r = FaceRecognitionService.identify(_norm([1, 0, 0]), [
        _reg(1, 'Rafid', [1, 0.05, 0]),
        _reg(2, 'Sadman', [0, 1, 0]),
      ]);
      expect(r.accepted, isTrue);
      expect(r.best!.face.name, 'Rafid');
    });

    test('rejects lookalikes whose scores are too close', () {
      final r = FaceRecognitionService.identify(_norm([1, 0, 0]), [
        _reg(1, 'Rafid', [1, 0.30, 0]), // ≈0.958
        _reg(2, 'Sadman', [1, 0.42, 0]), // ≈0.922 → gap < minMargin
      ]);
      expect(r.accepted, isFalse);
      expect(r.reason, contains('Unsure'));
    });

    test('two photos of the SAME person do not count as a competitor', () {
      final r = FaceRecognitionService.identify(_norm([1, 0, 0]), [
        _reg(1, 'Rafid', [1, 0.02, 0]),
        _reg(2, 'rafid ', [1, 0.03, 0]), // same person, second photo
        _reg(3, 'Sadman', [0, 1, 0]),
      ]);
      expect(r.accepted, isTrue);
      expect(r.runnerUp!.face.name, 'Sadman');
    });

    test('rejects below threshold', () {
      final r = FaceRecognitionService.identify(_norm([1, 0, 0]), [
        _reg(1, 'Rafid', [0.5, 1, 0]),
      ]);
      expect(r.accepted, isFalse);
    });
  });
}
