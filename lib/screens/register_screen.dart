import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../services/database_service.dart';
import '../services/face_recognition_service.dart';
import '../utils/image_utils.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

/// Decodes the picked file and applies its EXIF orientation (isolate).
img.Image? _decodeUpright(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return null;
  return img.bakeOrientation(decoded);
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _nameController = TextEditingController();
  final _picker = ImagePicker();
  final _detector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.accurate,
      enableLandmarks: true,
      minFaceSize: 0.15,
    ),
  );

  bool _processing = false;
  String? _error;
  Uint8List? _faceJpeg;
  Float32List? _embedding;

  @override
  void dispose() {
    _nameController.dispose();
    _detector.close();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(
      source: source,
      preferredCameraDevice: CameraDevice.front,
      maxWidth: 1280,
      maxHeight: 1280,
      imageQuality: 95,
    );
    if (picked == null) return;

    setState(() {
      _processing = true;
      _error = null;
      _faceJpeg = null;
      _embedding = null;
    });

    try {
      final bytes = await picked.readAsBytes();
      final upright = await compute(_decodeUpright, bytes);
      if (upright == null) throw 'Could not read this image.';

      // Write the upright image to a temp file so ML Kit's coordinates match
      // exactly the pixels we crop from.
      final tmpDir = await getTemporaryDirectory();
      final tmpPath = p.join(
        tmpDir.path,
        'reg_${DateTime.now().millisecondsSinceEpoch}.jpg',
      );
      await File(tmpPath).writeAsBytes(img.encodeJpg(upright, quality: 95));

      final faces =
          await _detector.processImage(InputImage.fromFilePath(tmpPath));
      File(tmpPath).delete().ignore();

      if (faces.isEmpty) throw 'No face found. Use a clear, front-facing photo.';
      if (faces.length > 1) {
        throw 'More than one face found. Use a photo with only one person.';
      }

      final face = faces.first;
      final crop = ImageUtils.cropFace(upright, face.boundingBox);
      if (crop.width < 60) {
        throw 'Face is too small. Use a closer photo.';
      }

      final embedding = FaceRecognitionService.instance.getEmbedding(crop);
      final thumb = img.copyResize(crop, width: 160, height: 160);

      setState(() {
        _embedding = embedding;
        _faceJpeg = Uint8List.fromList(img.encodeJpg(thumb, quality: 90));
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _processing = false);
    }
  }

  Future<void> _save() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      setState(() => _error = 'Please enter a user name.');
      return;
    }
    if (_embedding == null || _faceJpeg == null) return;

    // Warn if this face already belongs to a different registered name.
    final existing = await DatabaseService.instance.getAllFaces();
    final best = FaceRecognitionService.findBestMatch(_embedding!, existing);
    if (best != null &&
        best.similarity >= FaceRecognitionService.matchThreshold &&
        best.face.name.toLowerCase() != name.toLowerCase() &&
        mounted) {
      final proceed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Face already registered'),
          content: Text(
            'This face looks like "${best.face.name}" '
            '(score ${best.similarity.toStringAsFixed(2)}). Save anyway?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Save'),
            ),
          ],
        ),
      );
      if (proceed != true) return;
    }

    await DatabaseService.instance.insertFace(
      RegisteredFace(
        name: name,
        embedding: _embedding!,
        faceJpeg: _faceJpeg!,
        createdAt: DateTime.now(),
      ),
    );
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('"$name" registered successfully')),
    );
    Navigator.pop(context, true);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Register Face')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            TextField(
              controller: _nameController,
              textCapitalization: TextCapitalization.words,
              decoration: const InputDecoration(
                labelText: 'User name',
                prefixIcon: Icon(Icons.person),
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 24),
            Center(
              child: Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(16),
                ),
                clipBehavior: Clip.antiAlias,
                child: _processing
                    ? const Center(child: CircularProgressIndicator())
                    : _faceJpeg != null
                        ? Image.memory(_faceJpeg!, fit: BoxFit.cover)
                        : Icon(Icons.face,
                            size: 96, color: theme.colorScheme.outline),
              ),
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed:
                        _processing ? null : () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed:
                        _processing ? null : () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
            if (_error != null) ...[
              const SizedBox(height: 16),
              Text(_error!,
                  style: TextStyle(color: theme.colorScheme.error),
                  textAlign: TextAlign.center),
            ],
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed:
                  (_embedding != null && !_processing) ? _save : null,
              icon: const Icon(Icons.save),
              label: const Text('Save to database'),
              style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16)),
            ),
          ],
        ),
      ),
    );
  }
}
