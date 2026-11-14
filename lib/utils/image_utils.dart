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

/// Arguments for [cropFaceFromFrame] (top-level so `compute` can run it).
class CropRequest {
  final FrameData frame;
  final double left, top, right, bottom;
  CropRequest(this.frame, Rect r)
      : left = r.left,
        top = r.top,
        right = r.right,
        bottom = r.bottom;
}

/// Converts the frame to RGB and returns the cropped face. Runs in an isolate.
img.Image cropFaceFromFrame(CropRequest req) {
  final full = ImageUtils.frameToImage(req.frame);
  return ImageUtils.cropFace(
    full,
    Rect.fromLTRB(req.left, req.top, req.right, req.bottom),
  );
}
