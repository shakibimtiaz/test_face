import 'dart:math' as math;
import 'dart:ui' show Offset;

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'image_utils.dart';

/// Standard ArcFace / InsightFace 5-point template for a 112×112 face.
/// MobileFaceNet was trained on faces warped so that these five points land
/// exactly here. Feeding faces cropped any other way (e.g. a raw bounding box)
/// moves eyes/nose/mouth off this grid and weakens the embedding — which is
/// what lets two similar-looking people (e.g. both with glasses) get confused.
const List<Offset> _template112 = [
  Offset(38.2946, 51.6963), // eye on the image's left side
  Offset(73.5318, 51.5014), // eye on the image's right side
  Offset(56.0252, 71.7366), // nose tip
  Offset(41.5493, 92.3655), // mouth corner, image left
  Offset(70.7299, 92.2041), // mouth corner, image right
];

/// A 2-D similarity transform (rotation + uniform scale + translation) that
/// maps a pixel of the aligned output face back to the source image:
///   src.x = a·x − b·y + tx
///   src.y = b·x + a·y + ty
class FaceAlignment {
  final double a, b, tx, ty;
  const FaceAlignment(this.a, this.b, this.tx, this.ty);

  double srcX(double x, double y) => a * x - b * y + tx;
  double srcY(double x, double y) => b * x + a * y + ty;

  /// Builds the alignment from ML Kit landmarks (upright image coordinates).
  /// Needs at least both eyes; nose and mouth corners are used when present.
  /// Returns null if landmarks are missing or implausible.
  static FaceAlignment? fromFace(Face face, int size) {
    Offset? p(FaceLandmarkType t) {
      final q = face.landmarks[t]?.position;
      return q == null ? null : Offset(q.x.toDouble(), q.y.toDouble());
    }

    final s = size / 112.0;
    final t = _template112.map((o) => o * s).toList();

    final e1 = p(FaceLandmarkType.leftEye);
    final e2 = p(FaceLandmarkType.rightEye);
    if (e1 == null || e2 == null) return null;
    // Sort by x instead of trusting "left/right" names, so it works for
    // mirrored and non-mirrored images alike.
    final eyes = [e1, e2]..sort((u, v) => u.dx.compareTo(v.dx));
    if ((eyes[1] - eyes[0]).distance < 8) return null; // too small / bogus

    final src = <Offset>[eyes[0], eyes[1]];
    final dst = <Offset>[t[0], t[1]];

    final nose = p(FaceLandmarkType.noseBase);
    if (nose != null) {
      src.add(nose);
      dst.add(t[2]);
    }
    final m1 = p(FaceLandmarkType.leftMouth);
    final m2 = p(FaceLandmarkType.rightMouth);
    if (m1 != null && m2 != null) {
      final mouth = [m1, m2]..sort((u, v) => u.dx.compareTo(v.dx));
      src.addAll(mouth);
      dst.addAll([t[3], t[4]]);
    }
    // Least-squares fit of output (template) → source (image).
    return _fit(dst, src);
  }

  static FaceAlignment? _fit(List<Offset> from, List<Offset> to) {
    final n = from.length;
    var fx = 0.0, fy = 0.0, gx = 0.0, gy = 0.0;
    for (var i = 0; i < n; i++) {
      fx += from[i].dx;
      fy += from[i].dy;
      gx += to[i].dx;
      gy += to[i].dy;
    }
    fx /= n;
    fy /= n;
    gx /= n;
    gy /= n;

    var num1 = 0.0, num2 = 0.0, den = 0.0;
    for (var i = 0; i < n; i++) {
      final x = from[i].dx - fx, y = from[i].dy - fy;
      final u = to[i].dx - gx, v = to[i].dy - gy;
      num1 += x * u + y * v;
      num2 += x * v - y * u;
      den += x * x + y * y;
    }
    if (den == 0) return null;
    final a = num1 / den, b = num2 / den;
    if (math.sqrt(a * a + b * b) < 1e-3) return null;
    return FaceAlignment(a, b, gx - (a * fx - b * fy), gy - (b * fx + a * fy));
  }
}

/// Aligned [size]×[size] face from a decoded, upright image (registration).
img.Image alignedFaceFromImage(img.Image src, FaceAlignment al, int size) {
  final out = img.Image(width: size, height: size);
  for (var j = 0; j < size; j++) {
    for (var i = 0; i < size; i++) {
      final x = al.srcX(i + 0.5, j + 0.5) - 0.5;
      final y = al.srcY(i + 0.5, j + 0.5) - 0.5;
      if (x < 0 || y < 0 || x > src.width - 1 || y > src.height - 1) {
        continue; // outside the photo → black, like the training data
      }
      final px = src.getPixelInterpolate(x, y,
          interpolation: img.Interpolation.linear);
      out.setPixelRgb(i, j, px.r, px.g, px.b);
    }
  }
  return out;
}

/// Aligned [size]×[size] face sampled straight from a raw camera frame
/// (real-time path; only ~12k pixels are touched, no full-frame conversion).
img.Image alignedFaceFromFrame(FrameData f, FaceAlignment al, int size) {
  final out = img.Image(width: size, height: size);
  final rot = f.rotation % 360;
  final swap = rot == 90 || rot == 270;
  final uprightW = swap ? f.height : f.width;
  final uprightH = swap ? f.width : f.height;
  final w = f.width, h = f.height, b = f.bytes;
  final frameSize = w * h;

  for (var j = 0; j < size; j++) {
    for (var i = 0; i < size; i++) {
      final u = al.srcX(i + 0.5, j + 0.5);
      final v = al.srcY(i + 0.5, j + 0.5);
      if (u < 0 || v < 0 || u > uprightW - 1 || v > uprightH - 1) continue;

      // Upright (u, v) -> raw frame (x, y): inverse of a clockwise rotation.
      final double x, y;
      switch (rot) {
        case 90:
          x = v;
          y = h - 1 - u;
        case 180:
          x = w - 1 - u;
          y = h - 1 - v;
        case 270:
          x = w - 1 - v;
          y = u;
        default:
          x = u;
          y = v;
      }

      final x0 = x.floor().clamp(0, w - 1);
      final y0 = y.floor().clamp(0, h - 1);
      if (f.isNv21) {
        // Bilinear luma, nearest chroma (chroma is half-resolution anyway).
        final x1 = math.min(x0 + 1, w - 1);
        final y1 = math.min(y0 + 1, h - 1);
        final fx = (x - x0).clamp(0.0, 1.0);
        final fy = (y - y0).clamp(0.0, 1.0);
        final yy =
            (b[y0 * w + x0] * (1 - fx) + b[y0 * w + x1] * fx) * (1 - fy) +
                (b[y1 * w + x0] * (1 - fx) + b[y1 * w + x1] * fx) * fy;
        final uvIndex = frameSize + (y0 >> 1) * w + (x0 & ~1);
        final cv = b[uvIndex] - 128;
        final cu = b[uvIndex + 1] - 128;
        out.setPixelRgb(
          i,
          j,
          (yy + 1.370705 * cv).round().clamp(0, 255),
          (yy - 0.337633 * cu - 0.698001 * cv).round().clamp(0, 255),
          (yy + 1.732446 * cu).round().clamp(0, 255),
        );
      } else {
        final p = y0 * f.rowStride + x0 * 4; // iOS BGRA8888
        out.setPixelRgb(i, j, b[p + 2], b[p + 1], b[p]);
      }
    }
  }
  return out;
}
