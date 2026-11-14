import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'database_service.dart';

class FaceMatch {
  final RegisteredFace face;
  final double similarity;
  FaceMatch(this.face, this.similarity);
}

/// Runs MobileFaceNet (assets/mobilefacenet.tflite) fully on-device to turn a
/// cropped face into an embedding vector, and compares embeddings.
class FaceRecognitionService {
  FaceRecognitionService._();
  static final FaceRecognitionService instance = FaceRecognitionService._();

  /// Cosine-similarity threshold. Two L2-normalised MobileFaceNet embeddings
  /// of the same person usually score above ~0.6; different people usually
  /// score below ~0.45. Raise it to be stricter, lower it if genuine users
  /// are being rejected (the live score is shown on the Authenticate screen).
  static const double matchThreshold = 0.60;

  Interpreter? _interpreter;
  late List<int> _inputShape; // e.g. [1, 112, 112, 3]
  late List<int> _outputShape; // e.g. [1, 192]

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    if (_interpreter != null) return;
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(
      'assets/mobilefacenet.tflite',
      options: options,
    );
    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;
    debugPrint('MobileFaceNet input=$_inputShape output=$_outputShape');
  }

  /// Produces an L2-normalised embedding for an already-cropped face image.
  Float32List getEmbedding(img.Image face) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('FaceRecognitionService.load() was not called');
    }

    final batch = _inputShape[0];
    final h = _inputShape[1];
    final w = _inputShape[2];
    final resized = img.copyResize(
      face,
      width: w,
      height: h,
      interpolation: img.Interpolation.linear,
    );

    // Normalise pixels to [-1, 1] as MobileFaceNet expects.
    final perImage = w * h * 3;
    final input = Float32List(batch * perImage);
    var i = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = resized.getPixel(x, y);
        input[i++] = (px.r - 127.5) / 128.0;
        input[i++] = (px.g - 127.5) / 128.0;
        input[i++] = (px.b - 127.5) / 128.0;
      }
    }
    // Some MobileFaceNet exports use a fixed batch of 2 — duplicate the face.
    for (var b = 1; b < batch; b++) {
      input.setRange(b * perImage, (b + 1) * perImage, input);
    }

    final outLen = _outputShape.reduce((a, b) => a * b);
    final output = Float32List(outLen);
    interpreter.run(input.buffer, output.buffer);

    final dim = outLen ~/ _outputShape[0];
    return _l2Normalize(Float32List.sublistView(output, 0, dim));
  }

  static Float32List _l2Normalize(Float32List v) {
    var sum = 0.0;
    for (final x in v) {
      sum += x * x;
    }
    final norm = math.sqrt(sum);
    final out = Float32List(v.length);
    if (norm == 0) return out;
    for (var i = 0; i < v.length; i++) {
      out[i] = v[i] / norm;
    }
    return out;
  }

  /// Cosine similarity of two L2-normalised vectors (range -1..1).
  static double similarity(Float32List a, Float32List b) {
    if (a.length != b.length) return -1;
    var dot = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  /// Returns the closest registered face, or null if the database is empty.
  static FaceMatch? findBestMatch(
    Float32List embedding,
    List<RegisteredFace> registered,
  ) {
    FaceMatch? best;
    for (final r in registered) {
      final s = similarity(embedding, r.embedding);
      if (best == null || s > best.similarity) best = FaceMatch(r, s);
    }
    return best;
  }
}
