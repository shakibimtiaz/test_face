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

/// Result of comparing one live face against everyone registered.
class IdentifyResult {
  /// Highest-scoring person (best of all their registered photos).
  final FaceMatch? best;

  /// Highest-scoring DIFFERENT person (not just another photo of [best]).
  final FaceMatch? runnerUp;

  /// True only if [best] passes the threshold AND clearly beats [runnerUp].
  final bool accepted;

  /// Why it was not accepted (for the on-screen hint), or null.
  final String? reason;

  const IdentifyResult(this.best, this.runnerUp, this.accepted, this.reason);

  double get margin =>
      best == null ? 0 : best!.similarity - (runnerUp?.similarity ?? -1);
}

/// Runs MobileFaceNet (assets/mobilefacenet.tflite) fully on-device to turn a
/// cropped face into an embedding vector, and compares embeddings.
class FaceRecognitionService {
  FaceRecognitionService._();
  static final FaceRecognitionService instance = FaceRecognitionService._();

  /// Version of the face pre-processing used to create embeddings.
  ///   1 = raw bounding-box crop (old, not comparable with 2)
  ///   2 = 5-point landmark alignment (current)
  /// Faces stored with an older version must be registered again.
  static const int embeddingVersion = 2;

  /// Cosine-similarity threshold for ALIGNED faces. Same person typically
  /// scores ~0.70–0.90, different people ~0.10–0.50. Lookalikes can land in
  /// between — that is what [minMargin] and the multi-frame agreement on the
  /// Authenticate screen protect against. Tune with the live scores shown on
  /// the Authenticate screen (best + runner-up).
  static const double matchThreshold = 0.65;

  /// The best person must beat the next DIFFERENT person by at least this
  /// much; otherwise the result is "ambiguous" and nobody is authenticated.
  static const double minMargin = 0.08;

  Interpreter? _interpreter;
  late List<int> _inputShape; // e.g. [1, 112, 112, 3]
  late List<int> _outputShape; // e.g. [1, 192]

  bool get isLoaded => _interpreter != null;

  /// Width/height the model expects (112 for MobileFaceNet).
  int get inputSize => _inputShape[1];

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
    // Real-time crops are already sampled at the model size — skip resizing.
    final resized = (face.width == w && face.height == h)
        ? face
        : img.copyResize(
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

  /// People are grouped by name, so several registered photos of the same
  /// person never compete with each other in the margin check.
  static String personKey(RegisteredFace f) => f.name.trim().toLowerCase();

  /// Decides who this face is — or that it is nobody / ambiguous.
  static IdentifyResult identify(
    Float32List embedding,
    List<RegisteredFace> registered,
  ) {
    // Best score per person (over all of that person's registered photos).
    final perPerson = <String, FaceMatch>{};
    for (final r in registered) {
      final s = similarity(embedding, r.embedding);
      final key = personKey(r);
      final cur = perPerson[key];
      if (cur == null || s > cur.similarity) perPerson[key] = FaceMatch(r, s);
    }
    if (perPerson.isEmpty) {
      return const IdentifyResult(null, null, false, 'No faces registered');
    }

    final ranked = perPerson.values.toList()
      ..sort((x, y) => y.similarity.compareTo(x.similarity));
    final best = ranked.first;
    final runnerUp = ranked.length > 1 ? ranked[1] : null;

    if (best.similarity < matchThreshold) {
      return IdentifyResult(best, runnerUp, false, 'Face not recognized');
    }
    if (runnerUp != null &&
        best.similarity - runnerUp.similarity < minMargin) {
      return IdentifyResult(best, runnerUp, false,
          'Unsure: ${best.face.name} or ${runnerUp.face.name}?');
    }
    return IdentifyResult(best, runnerUp, true, null);
  }
}
