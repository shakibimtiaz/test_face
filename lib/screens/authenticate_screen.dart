import 'dart:async';
import 'dart:io';
// [LIVENESS] import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../services/database_service.dart';
import '../services/face_recognition_service.dart';
import '../utils/image_utils.dart';

// ---------------------------------------------------------------------------
// LIVENESS IS CURRENTLY DISABLED.
// Every liveness-related line is commented out and marked with `[LIVENESS]`.
// To re-enable it: uncomment the `[LIVENESS]` lines/blocks and remove the
// `[AUTH-ONLY]` lines that replaced them.
// ---------------------------------------------------------------------------

// [LIVENESS] Liveness challenges. A printed photo or a photo on a screen cannot
// blink, smile on demand, or turn its head in 3D while ML Kit keeps tracking
// the same face, so passing these proves a live person is in front of the camera.
// enum Challenge { blink, smile, turnHead }

// [LIVENESS] enum _Stage { loading, identify, liveness, finalCheck, success, failed }
enum _Stage { loading, identify, success, failed } // [AUTH-ONLY]

class AuthenticateScreen extends StatefulWidget {
  const AuthenticateScreen({super.key});

  @override
  State<AuthenticateScreen> createState() => _AuthenticateScreenState();
}

class _AuthenticateScreenState extends State<AuthenticateScreen>
    with WidgetsBindingObserver {
  // [LIVENESS] static const _sessionTimeout = Duration(seconds: 20);

  CameraController? _controller;
  CameraDescription? _camera;
  bool _cameraStarting = false;
  bool _pausedByLifecycle = false;
  int _cameraGeneration = 0; // bumped on every start/stop to cancel stale starts
  bool _disposing = false;
  final _detector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      // [LIVENESS] enableClassification: true, // eye-open & smiling probabilities
      enableLandmarks: true,
      // [LIVENESS] enableTracking: true, // same trackingId across frames
      minFaceSize: 0.25,
    ),
  );

  List<RegisteredFace> _registered = [];
  bool _busy = false;
  _Stage _stage = _Stage.loading;
  String _message = 'Starting camera…';
  String? _error;

  // Identity / session state
  FaceMatch? _candidate;
  // [LIVENESS] int? _trackingId;
  // [LIVENESS] DateTime? _sessionStart;
  // [LIVENESS] int _missedFrames = 0;
  double? _lastScore;

  // [LIVENESS] Liveness state
  // List<Challenge> _challenges = [];
  // int _challengeIndex = 0;
  // bool _eyesWereOpen = false;
  // bool _eyesWereClosed = false;
  // bool _wasNeutral = false;
  // bool _headTurned = false;

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
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw CameraException('noCamera', 'No camera found');
      _camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
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
      _resetSession(message: 'Look straight at the camera');
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

  void _resetSession({String? message}) {
    if (!mounted) return;
    setState(() {
      _stage = _Stage.identify;
      _candidate = null;
      // [LIVENESS] _trackingId = null;
      // [LIVENESS] _sessionStart = null;
      // [LIVENESS] _missedFrames = 0;
      // [LIVENESS] _challengeIndex = 0;
      // [LIVENESS] _challenges = [];
      _message = message ?? 'Look straight at the camera';
    });
  }

  /* [LIVENESS] only used by the liveness flow
  void _fail(String reason) {
    setState(() {
      _stage = _Stage.failed;
      _message = reason;
    });
    // Automatically try again after a short pause.
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted && _stage == _Stage.failed && _error == null) {
        _resetSession();
      }
    });
  }
  */

  Future<void> _onFrame(CameraImage image) async {
    if (_busy || _camera == null) return;
    // [LIVENESS] if (_stage != _Stage.identify &&
    // [LIVENESS]     _stage != _Stage.liveness &&
    // [LIVENESS]     _stage != _Stage.finalCheck) {
    // [LIVENESS]   return;
    // [LIVENESS] }
    if (_stage != _Stage.identify) return; // [AUTH-ONLY]
    _busy = true;
    try {
      final frame = ImageUtils.frameFromCameraImage(image, _camera!);
      if (frame == null) return;
      final faces =
          await _detector.processImage(ImageUtils.inputImageFromFrame(frame));
      if (!mounted) return;
      await _handleFaces(faces, frame);
    } catch (e) {
      debugPrint('Frame error: $e');
    } finally {
      _busy = false;
    }
  }

  bool _isFrontal(Face f) =>
      (f.headEulerAngleY ?? 0).abs() < 12 &&
      (f.headEulerAngleZ ?? 0).abs() < 12 &&
      (f.headEulerAngleX ?? 0).abs() < 15;

  Future<FaceMatch?> _match(Face face, FrameData frame) async {
    final crop =
        await compute(cropFaceFromFrame, CropRequest(frame, face.boundingBox));
    final emb = FaceRecognitionService.instance.getEmbedding(crop);
    return FaceRecognitionService.findBestMatch(emb, _registered);
  }

  Future<void> _handleFaces(List<Face> faces, FrameData frame) async {
    /* [LIVENESS] Session timeout
    if (_sessionStart != null &&
        DateTime.now().difference(_sessionStart!) > _sessionTimeout) {
      _fail('Timed out. Liveness check not completed.');
      return;
    }
    */

    if (faces.isEmpty) {
      setState(() => _message = 'No face detected'); // [AUTH-ONLY]
      /* [LIVENESS]
      if (_stage == _Stage.identify) {
        setState(() => _message = 'No face detected');
      } else if (++_missedFrames > 15) {
        _fail('Face lost. Keep your face inside the frame.');
      }
      */
      return;
    }
    if (faces.length > 1) {
      setState(() => _message = 'Only one face allowed'); // [AUTH-ONLY]
      /* [LIVENESS]
      if (_stage == _Stage.identify) {
        setState(() => _message = 'Only one face allowed');
      } else {
        _fail('Multiple faces detected.');
      }
      */
      return;
    }
    // [LIVENESS] _missedFrames = 0;
    final face = faces.first;

    /* [LIVENESS] Once identified, the very same tracked face must complete every step.
    if (_stage != _Stage.identify &&
        _trackingId != null &&
        face.trackingId != null &&
        face.trackingId != _trackingId) {
      _fail('A different face appeared. Starting over.');
      return;
    }
    */

    await _identify(face, frame); // [AUTH-ONLY]
    /* [LIVENESS]
    switch (_stage) {
      case _Stage.identify:
        await _identify(face, frame);
      case _Stage.liveness:
        _checkLiveness(face);
      case _Stage.finalCheck:
        await _finalCheck(face, frame);
      default:
        break;
    }
    */
  }

  Future<void> _identify(Face face, FrameData frame) async {
    if (!_isFrontal(face)) {
      setState(() => _message = 'Look straight at the camera');
      return;
    }
    final best = await _match(face, frame);
    if (!mounted || best == null) return;
    _lastScore = best.similarity;
    if (best.similarity < FaceRecognitionService.matchThreshold) {
      setState(() => _message =
          'Face not recognized (score ${best.similarity.toStringAsFixed(2)})');
      return;
    }

    // [AUTH-ONLY] Face matched → authenticated immediately (no liveness).
    setState(() {
      _stage = _Stage.success;
      _candidate = best;
      _message = 'Authenticated successfully';
    });
    await DatabaseService.instance
        .insertAttendance(best.face.id!, best.face.name, best.similarity);
    _stopCamera();

    /* [LIVENESS] Identity candidate found — now prove liveness. Blink is always
    // required; one more random challenge makes pre-recorded videos less useful.
    final rnd = math.Random();
    final extra = [Challenge.smile, Challenge.turnHead][rnd.nextInt(2)];
    final challenges = rnd.nextBool()
        ? [Challenge.blink, extra]
        : [extra, Challenge.blink];
    setState(() {
      _candidate = best;
      _trackingId = face.trackingId;
      _sessionStart = DateTime.now();
      _challenges = challenges;
      _challengeIndex = 0;
      _stage = _Stage.liveness;
      _resetChallengeFlags();
      _message = _challengeText(challenges.first);
    });
    */
  }

  /* [LIVENESS] ---------- liveness helpers (disabled) ----------
  void _resetChallengeFlags() {
    _eyesWereOpen = false;
    _eyesWereClosed = false;
    _wasNeutral = false;
    _headTurned = false;
  }

  String _challengeText(Challenge c) => switch (c) {
        Challenge.blink => 'Blink your eyes',
        Challenge.smile => 'Smile',
        Challenge.turnHead => 'Turn your head to one side, then back',
      };

  void _checkLiveness(Face face) {
    final challenge = _challenges[_challengeIndex];
    var done = false;

    switch (challenge) {
      case Challenge.blink:
        final l = face.leftEyeOpenProbability;
        final r = face.rightEyeOpenProbability;
        if (l == null || r == null) break;
        if (l > 0.7 && r > 0.7) {
          if (_eyesWereClosed) done = true; // open → closed → open
          _eyesWereOpen = true;
        } else if (l < 0.25 && r < 0.25 && _eyesWereOpen) {
          _eyesWereClosed = true;
        }
      case Challenge.smile:
        final s = face.smilingProbability;
        if (s == null) break;
        if (s < 0.3) _wasNeutral = true; // must go neutral → smiling
        if (s > 0.8 && _wasNeutral) done = true;
      case Challenge.turnHead:
        final yaw = (face.headEulerAngleY ?? 0).abs();
        if (yaw > 25) _headTurned = true;
        if (_headTurned && yaw < 10) done = true; // turned → back to centre
    }

    if (!done) return;
    _challengeIndex++;
    _resetChallengeFlags();
    setState(() {
      if (_challengeIndex >= _challenges.length) {
        _stage = _Stage.finalCheck;
        _message = 'Look straight at the camera';
      } else {
        _message = _challengeText(_challenges[_challengeIndex]);
      }
    });
  }

  Future<void> _finalCheck(Face face, FrameData frame) async {
    if (!_isFrontal(face)) return;
    // Re-verify identity after liveness so no one can swap faces mid-way.
    final best = await _match(face, frame);
    if (!mounted || best == null) return;
    _lastScore = best.similarity;
    if (best.similarity >= FaceRecognitionService.matchThreshold &&
        best.face.id == _candidate!.face.id) {
      setState(() {
        _stage = _Stage.success;
        _candidate = best;
        _message = 'Authenticated';
      });
      await DatabaseService.instance
          .insertAttendance(best.face.id!, best.face.name, best.similarity);
      _stopCamera();
    } else {
      _fail('Face did not match after liveness check.');
    }
  }
  ---------- end of liveness helpers ---------- */

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(title: const Text('Authenticate')),
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
    final m = _candidate!;
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
                fontSize: 36,
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
              // [LIVENESS] 'Match score ${m.similarity.toStringAsFixed(2)} • Live face verified',
              'Match score ${m.similarity.toStringAsFixed(2)}', // [AUTH-ONLY]
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

  Widget _buildCamera() {
    final c = _controller;
    if (c == null || !c.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    final failed = _stage == _Stage.failed;
    // [LIVENESS] final color = failed
    // [LIVENESS]     ? Colors.redAccent
    // [LIVENESS]     : _stage == _Stage.identify
    // [LIVENESS]         ? Colors.white
    // [LIVENESS]         : Colors.lightGreenAccent;
    final color = failed ? Colors.redAccent : Colors.white; // [AUTH-ONLY]

    return Stack(
      fit: StackFit.expand,
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: 1 / c.value.aspectRatio,
            child: CameraPreview(c),
          ),
        ),
        // Face guide
        Center(
          child: Container(
            width: 240,
            height: 320,
            decoration: BoxDecoration(
              border: Border.all(color: color, width: 4),
              borderRadius: const BorderRadius.all(Radius.elliptical(120, 160)),
            ),
          ),
        ),
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
                // [LIVENESS] if (_candidate != null && _stage != _Stage.failed)
                // [LIVENESS]   Text(
                // [LIVENESS]     'Hello ${_candidate!.face.name} — liveness check',
                // [LIVENESS]     style: const TextStyle(color: Colors.white70),
                // [LIVENESS]   ),
                Text(
                  _message,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: color,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                /* [LIVENESS] challenge progress dots
                if (_challenges.isNotEmpty && _stage == _Stage.liveness) ...[
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(
                      _challenges.length,
                      (i) => Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: Icon(
                          i < _challengeIndex
                              ? Icons.check_circle
                              : Icons.radio_button_unchecked,
                          color: Colors.lightGreenAccent,
                        ),
                      ),
                    ),
                  ),
                ],
                */
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
