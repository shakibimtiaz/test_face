import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../services/database_service.dart';
import '../services/face_recognition_service.dart';
import '../utils/image_utils.dart';

/// Real-time authentication.
///
/// On EVERY camera frame, in parallel:
///   • Face recognition — the face is matched against the registered faces.
///   • Liveness — eye blinking is tracked (open → closed → open).
///
/// Both must happen on the same tracked face (ML Kit trackingId). As soon as
/// the face matches AND one blink has been seen, the user is authenticated.
/// A printed photo / photo on a screen can't blink, so it never passes.
enum _Stage { loading, scanning, success, failed }

class AuthenticateScreen extends StatefulWidget {
  const AuthenticateScreen({super.key});

  @override
  State<AuthenticateScreen> createState() => _AuthenticateScreenState();
}

class _AuthenticateScreenState extends State<AuthenticateScreen>
    with WidgetsBindingObserver {
  // Blink thresholds on ML Kit's eye-open probability (0 = closed, 1 = open).
  static const _eyeOpen = 0.6;
  static const _eyeClosed = 0.3;
  // Frames without a face before the current session is reset.
  static const _maxMissedFrames = 8;

  CameraController? _controller;
  CameraDescription? _camera;
  List<CameraDescription> _cameras = [];
  CameraLensDirection _lens = CameraLensDirection.front; // front or back
  bool _switching = false;
  bool _cameraStarting = false;
  bool _pausedByLifecycle = false;
  int _cameraGeneration = 0; // bumped on every start/stop to cancel stale starts
  bool _disposing = false;
  final _detector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      enableClassification: true, // eye-open probabilities (blink)
      enableTracking: true, // same trackingId across frames
      minFaceSize: 0.25,
    ),
  );

  List<RegisteredFace> _registered = [];
  bool _busy = false;
  _Stage _stage = _Stage.loading;
  String _message = 'Starting camera…';
  String? _error;

  // Per-face session state (reset when the tracked face changes / is lost).
  int? _trackingId;
  int _missedFrames = 0;
  FaceMatch? _match; // set once the face is recognised
  double? _lastScore;
  bool _eyesWereOpen = false;
  bool _eyesClosed = false;
  bool _blinkDetected = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  @override
  void dispose() {
    _disposing = true;
    WidgetsBinding.instance.removeObserver(this);
    _stopCamera();
    _detector.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Use `paused` (app really in background), NOT `inactive`: the camera
    // permission dialog makes the app inactive while initialize() is running,
    // and disposing the controller then caused the "uninitialized" error.
    if (state == AppLifecycleState.paused) {
      if (_controller != null || _cameraStarting) {
        _pausedByLifecycle = true;
        _stopCamera();
      }
    } else if (state == AppLifecycleState.resumed && _pausedByLifecycle) {
      _pausedByLifecycle = false;
      if (_stage != _Stage.success && _error == null) _startCamera();
    }
  }

  Future<void> _init() async {
    _registered = await DatabaseService.instance.getAllFaces();
    if (_registered.isEmpty) {
      setState(() {
        _stage = _Stage.failed;
        _error = 'No faces registered yet. Register a face first.';
      });
      return;
    }
    await _startCamera();
  }

  Future<void> _startCamera() async {
    if (_cameraStarting || _controller != null) return; // already starting/running
    _cameraStarting = true;
    final generation = ++_cameraGeneration;
    CameraController? controller;
    try {
      if (_cameras.isEmpty) _cameras = await availableCameras();
      final cameras = _cameras;
      if (cameras.isEmpty) throw CameraException('noCamera', 'No camera found');
      _camera = cameras.firstWhere(
        (c) => c.lensDirection == _lens,
        orElse: () => cameras.first,
      );
      _lens = _camera!.lensDirection;
      controller = CameraController(
        _camera!,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();

      // The screen was closed or the camera was stopped while initializing:
      // throw this controller away instead of streaming from it.
      if (!mounted ||
          generation != _cameraGeneration ||
          !controller.value.isInitialized) {
        await controller.dispose();
        return;
      }

      await controller.startImageStream(_onFrame);
      _controller = controller;
      _resetSession(message: 'Look at the camera and blink');
    } on CameraException catch (e) {
      await controller?.dispose();
      if (!mounted || generation != _cameraGeneration) return;
      setState(() {
        _stage = _Stage.failed;
        _error = e.code == 'CameraAccessDenied'
            ? 'Camera permission denied. Allow camera access in Settings.'
            : 'Camera error: ${e.description ?? e.code}';
      });
    } finally {
      if (generation == _cameraGeneration) _cameraStarting = false;
    }
  }

  Future<void> _stopCamera() async {
    _cameraGeneration++; // cancels any start that is still in progress
    _cameraStarting = false;
    final c = _controller;
    _controller = null;
    if (mounted && !_disposing) setState(() {});
    if (c == null) return;
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
    } catch (_) {}
    await c.dispose();
  }

  /// Clears recognition + blink state (new person / face lost).
  void _resetSession({String? message}) {
    if (!mounted) return;
    setState(() {
      _stage = _Stage.scanning;
      _trackingId = null;
      _missedFrames = 0;
      _match = null;
      _eyesWereOpen = false;
      _eyesClosed = false;
      _blinkDetected = false;
      if (message != null) _message = message;
    });
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_busy || _camera == null || _stage != _Stage.scanning) return;
    _busy = true;
    try {
      final frame = ImageUtils.frameFromCameraImage(image, _camera!);
      if (frame == null) return;
      final faces =
          await _detector.processImage(ImageUtils.inputImageFromFrame(frame));
      if (!mounted || _stage != _Stage.scanning) return;
      await _process(faces, frame);
    } catch (e) {
      debugPrint('Frame error: $e');
    } finally {
      _busy = false;
    }
  }

  Future<void> _process(List<Face> faces, FrameData frame) async {
    // ---- Face presence -----------------------------------------------------
    if (faces.isEmpty) {
      if (++_missedFrames > _maxMissedFrames && _trackingId != null) {
        _resetSession(message: 'No face detected');
      } else if (_trackingId == null) {
        setState(() => _message = 'No face detected');
      }
      return;
    }
    if (faces.length > 1) {
      _resetSession(message: 'Only one face allowed');
      return;
    }
    _missedFrames = 0;
    final face = faces.first;

    // A different face appeared → start over for the new face.
    if (_trackingId != null &&
        face.trackingId != null &&
        face.trackingId != _trackingId) {
      _resetSession(message: 'Look at the camera and blink');
    }
    _trackingId ??= face.trackingId;

    // ---- 1) Liveness: blink detection (every frame) ------------------------
    _updateBlink(face);

    // ---- 2) Recognition (every frame until matched) ------------------------
    if (_match == null) _recognize(face, frame);

    // ---- Both done? --------------------------------------------------------
    if (_match != null && _blinkDetected) {
      await _onAuthenticated();
      return;
    }

    setState(() {
      if (_match == null) {
        _message = _lastScore == null
            ? 'Look straight at the camera'
            : 'Face not recognized';
      } else {
        _message = 'Blink your eyes';
      }
    });
  }

  void _updateBlink(Face face) {
    if (_blinkDetected) return;
    final l = face.leftEyeOpenProbability;
    final r = face.rightEyeOpenProbability;
    if (l == null || r == null) return;

    final open = l > _eyeOpen && r > _eyeOpen;
    // Counts as closed when either eye closes, so a single blink — even of
    // one eye — is enough.
    final closed = l < _eyeClosed || r < _eyeClosed;

    if (open) {
      if (_eyesClosed && _eyesWereOpen) {
        _blinkDetected = true; // open → closed → open
      }
      _eyesWereOpen = true;
      _eyesClosed = false;
    } else if (closed && _eyesWereOpen) {
      _eyesClosed = true;
    }
  }

  void _recognize(Face face, FrameData frame) {
    // Skip frames that would give a poor embedding.
    final yaw = (face.headEulerAngleY ?? 0).abs();
    final roll = (face.headEulerAngleZ ?? 0).abs();
    final l = face.leftEyeOpenProbability ?? 1;
    final r = face.rightEyeOpenProbability ?? 1;
    if (yaw > 20 || roll > 20 || l < 0.4 || r < 0.4) return;

    final service = FaceRecognitionService.instance;
    final crop = cropFaceFromFrameFast(frame, face.boundingBox, service.inputSize);
    final emb = service.getEmbedding(crop);
    final best = FaceRecognitionService.findBestMatch(emb, _registered);
    if (best == null) return;
    _lastScore = best.similarity;
    if (best.similarity >= FaceRecognitionService.matchThreshold) {
      _match = best;
    }
  }

  bool get _hasBothCameras =>
      _cameras.any((c) => c.lensDirection == CameraLensDirection.front) &&
      _cameras.any((c) => c.lensDirection == CameraLensDirection.back);

  /// Switches between the front and back camera. The rotation of each camera
  /// is handled by its own sensorOrientation, so recognition + blink detection
  /// work the same on both. The session restarts because tracking IDs differ.
  Future<void> _switchCamera() async {
    if (_switching || _stage == _Stage.success) return;
    _switching = true;
    try {
      await _stopCamera();
      _lens = _lens == CameraLensDirection.front
          ? CameraLensDirection.back
          : CameraLensDirection.front;
      await _startCamera();
    } finally {
      _switching = false;
    }
  }

  Future<void> _onAuthenticated() async {
    final m = _match!;
    setState(() {
      _stage = _Stage.success;
      _message = 'Authenticated successfully';
    });
    _stopCamera();
    await DatabaseService.instance
        .insertAttendance(m.face.id!, m.face.name, m.similarity);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Authenticate'),
        actions: [
          if (_hasBothCameras && _stage != _Stage.success && _error == null)
            IconButton(
              tooltip: _lens == CameraLensDirection.front
                  ? 'Switch to back camera'
                  : 'Switch to front camera',
              icon: Icon(_lens == CameraLensDirection.front
                  ? Icons.camera_rear
                  : Icons.camera_front),
              onPressed: _switching ? null : _switchCamera,
            ),
        ],
      ),
      body: _error != null
          ? _buildError()
          : _stage == _Stage.success
              ? _buildSuccess()
              : _buildCamera(),
    );
  }

  Widget _buildError() => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _error!,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white, fontSize: 18),
          ),
        ),
      );

  Widget _buildSuccess() {
    final m = _match!;
    return Container(
      color: Colors.green.shade700,
      width: double.infinity,
      child: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.verified, color: Colors.white, size: 96),
            const SizedBox(height: 16),
            const Text(
              'Authenticated successfully',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: 32,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            CircleAvatar(
              radius: 48,
              backgroundImage: MemoryImage(m.face.faceJpeg),
            ),
            const SizedBox(height: 12),
            Text(
              m.face.name,
              style: const TextStyle(color: Colors.white, fontSize: 24),
            ),
            Text(
              'Match score ${m.similarity.toStringAsFixed(2)} • Liveness detected (blink)',
              style: const TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 32),
            FilledButton.tonal(
              onPressed: () => Navigator.pop(context),
              child: const Text('Done'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _statusChip(String label, bool ok) => Chip(
        avatar: Icon(
          ok ? Icons.check_circle : Icons.hourglass_empty,
          color: ok ? Colors.green : Colors.orange,
          size: 18,
        ),
        label: Text(label),
        visualDensity: VisualDensity.compact,
      );

  Widget _buildCamera() {
    final c = _controller;
    if (c == null || !c.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    final recognized = _match != null;
    final color = recognized && _blinkDetected
        ? Colors.lightGreenAccent
        : recognized || _blinkDetected
            ? Colors.amberAccent
            : Colors.white;

    return Stack(
      fit: StackFit.expand,
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: 1 / c.value.aspectRatio,
            child: CameraPreview(c),
          ),
        ),
        // Face guide — oval sized to the screen (85% of the width, 4:3 tall).
        Builder(builder: (context) {
          final w = MediaQuery.sizeOf(context).width * 0.85;
          final h = w * 4 / 3;
          return Center(
            child: Container(
              width: w,
              height: h,
              decoration: BoxDecoration(
                border: Border.all(color: color, width: 4),
                borderRadius:
                    BorderRadius.all(Radius.elliptical(w / 2, h / 2)),
              ),
            ),
          );
        }),
        Positioned(
          left: 16,
          right: 16,
          bottom: 32,
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.65),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  _message,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: color,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  alignment: WrapAlignment.center,
                  children: [
                    _statusChip(
                      recognized ? 'Face: ${_match!.face.name}' : 'Face',
                      recognized,
                    ),
                    _statusChip('Liveness (blink)', _blinkDetected),
                  ],
                ),
                if (_lastScore != null)
                  Text(
                    'score ${_lastScore!.toStringAsFixed(2)} / '
                    'threshold ${FaceRecognitionService.matchThreshold}',
                    style: const TextStyle(color: Colors.white38, fontSize: 12),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
