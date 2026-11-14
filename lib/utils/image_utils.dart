import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect, Size;

import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

/// Raw camera frame copied out of a [CameraImage] so it can be sent to an
/// isolate and converted to RGB.
class FrameData {
  final Uint8List bytes;
  final int width;
  final int height;
  final int rowStride;

  /// Clockwise rotation (degrees) that makes the frame upright.
  final int rotation;
  final bool isNv21; // true = Android NV21, false = iOS BGRA8888

  FrameData({
    required this.bytes,
    required this.width,
    required this.height,
    required this.rowStride,
    required this.rotation,
    required this.isNv21,
  });
}

class ImageUtils {
  /// Packs a camera frame as NV21 (Android) or returns the BGRA bytes (iOS).
  static FrameData? frameFromCameraImage(
    CameraImage image,
    CameraDescription camera,
  ) {
    if (Platform.isAndroid) {
      final Uint8List nv21;
      if (image.planes.length == 1) {
        nv21 = image.planes.first.bytes; // camera plugin already gave NV21
      } else if (image.planes.length == 3) {
        nv21 = _yuv420ToNv21(image);
      } else {
        return null;
      }
      return FrameData(
        bytes: nv21,
        width: image.width,
        height: image.height,
        rowStride: image.width,
        // App is locked to portrait, so compensation == sensor orientation.
        rotation: camera.sensorOrientation,
        isNv21: true,
      );
    } else {
      if (image.planes.length != 1) return null;
      final plane = image.planes.first;
      return FrameData(
        bytes: plane.bytes,
        width: image.width,
        height: image.height,
        rowStride: plane.bytesPerRow,
        rotation: 0,
        isNv21: false,
      );
    }
  }

  /// Builds the ML Kit [InputImage] for a frame.
  static InputImage inputImageFromFrame(FrameData f) {
    return InputImage.fromBytes(
      bytes: f.bytes,
      metadata: InputImageMetadata(
        size: Size(f.width.toDouble(), f.height.toDouble()),
        rotation: InputImageRotationValue.fromRawValue(f.rotation) ??
            InputImageRotation.rotation0deg,
        format: f.isNv21 ? InputImageFormat.nv21 : InputImageFormat.bgra8888,
        bytesPerRow: f.rowStride,
      ),
    );
  }

  /// Converts a 3-plane YUV_420_888 image into a tightly packed NV21 buffer.
  static Uint8List _yuv420ToNv21(CameraImage image) {
    final w = image.width, h = image.height;
    final yPlane = image.planes[0], uPlane = image.planes[1], vPlane = image.planes[2];
    final out = Uint8List(w * h + 2 * (w ~/ 2) * (h ~/ 2));
    var o = 0;
    for (var y = 0; y < h; y++) {
      out.setRange(o, o + w, yPlane.bytes, y * yPlane.bytesPerRow);
      o += w;
    }
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;
    for (var y = 0; y < h ~/ 2; y++) {
      for (var x = 0; x < w ~/ 2; x++) {
        final idx = y * uPlane.bytesPerRow + x * uvPixelStride;
        out[o++] = vPlane.bytes[idx];
        out[o++] = uPlane.bytes[idx];
      }
    }
    return out;
  }

  /// Converts a frame to an upright RGB image (run inside an isolate).
  static img.Image frameToImage(FrameData f) {
    img.Image out;
    if (f.isNv21) {
      out = img.Image(width: f.width, height: f.height);
      final frameSize = f.width * f.height;
      final b = f.bytes;
      for (var y = 0; y < f.height; y++) {
        final yRow = y * f.width;
        final uvRow = frameSize + (y >> 1) * f.width;
        for (var x = 0; x < f.width; x++) {
          final yy = b[yRow + x];
          final uvIndex = uvRow + (x & ~1);
          final v = b[uvIndex] - 128;
          final u = b[uvIndex + 1] - 128;
          final r = (yy + 1.370705 * v).round().clamp(0, 255);
          final g = (yy - 0.337633 * u - 0.698001 * v).round().clamp(0, 255);
          final bl = (yy + 1.732446 * u).round().clamp(0, 255);
          out.setPixelRgb(x, y, r, g, bl);
        }
      }
    } else {
      out = img.Image.fromBytes(
        width: f.width,
        height: f.height,
        bytes: f.bytes.buffer,
        bytesOffset: f.bytes.offsetInBytes,
        rowStride: f.rowStride,
        numChannels: 4,
        order: img.ChannelOrder.bgra,
      );
    }
    if (f.rotation != 0) {
      out = img.copyRotate(out, angle: f.rotation);
    }
    return out;
  }

  /// Crops a square region around the face bounding box.
  static img.Image cropFace(img.Image image, Rect box) {
    final side = math.max(box.width, box.height);
    final cx = box.center.dx, cy = box.center.dy;
    var left = (cx - side / 2).round();
    var top = (cy - side / 2).round();
    var size = side.round();
    left = left.clamp(0, image.width - 1);
    top = top.clamp(0, image.height - 1);
    size = math.min(size, math.min(image.width - left, image.height - top));
    return img.copyCrop(image, x: left, y: top, width: size, height: size);
  }
}

/// FAST path for real-time use: samples only the face region straight out of
/// the raw camera frame at the model's input size ([outSize]×[outSize]).
///
/// [box] is ML Kit's bounding box, which is in *upright* (rotated) image
/// coordinates. Instead of converting + rotating the whole frame (~300k
/// pixels), we map each output pixel back to the raw frame (~12k pixels),
/// which is fast enough to run on every frame.
img.Image cropFaceFromFrameFast(FrameData f, Rect box, int outSize) {
  final rot = f.rotation % 360;
  // Size of the upright image the box refers to.
  final uprightW = (rot == 90 || rot == 270) ? f.height : f.width;
  final uprightH = (rot == 90 || rot == 270) ? f.width : f.height;

  // Square crop around the face, clamped to the image.
  var side = math.max(box.width, box.height);
  side = math.min(side, math.min(uprightW, uprightH).toDouble());
  final left = (box.center.dx - side / 2).clamp(0.0, uprightW - side);
  final top = (box.center.dy - side / 2).clamp(0.0, uprightH - side);
  final step = side / outSize;

  final out = img.Image(width: outSize, height: outSize);
  final b = f.bytes;
  final w = f.width, h = f.height;
  final frameSize = w * h;

  for (var j = 0; j < outSize; j++) {
    final v = top + (j + 0.5) * step;
    for (var i = 0; i < outSize; i++) {
      final u = left + (i + 0.5) * step;
      // Upright (u, v) -> raw (x, y): inverse of a clockwise rotation.
      double x, y;
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

      if (f.isNv21) {
        // Bilinear luma, nearest chroma (chroma is half-resolution anyway).
        final x0 = x.floor().clamp(0, w - 1);
        final y0 = y.floor().clamp(0, h - 1);
        final x1 = math.min(x0 + 1, w - 1);
        final y1 = math.min(y0 + 1, h - 1);
        final fx = (x - x0).clamp(0.0, 1.0);
        final fy = (y - y0).clamp(0.0, 1.0);
        final yy = (b[y0 * w + x0] * (1 - fx) + b[y0 * w + x1] * fx) * (1 - fy) +
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
        // iOS BGRA8888
        final xi = x.round().clamp(0, w - 1);
        final yi = y.round().clamp(0, h - 1);
        final p = yi * f.rowStride + xi * 4;
        out.setPixelRgb(i, j, b[p + 2], b[p + 1], b[p]);
      }
    }
  }
  return out;
}
