import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:test_face/services/face_recognition_service.dart';

void main() {
  test('cosine similarity of identical normalised vectors is 1', () {
    final v = Float32List.fromList([0.6, 0.8]);
    expect(FaceRecognitionService.similarity(v, v), closeTo(1.0, 1e-6));
  });
}
