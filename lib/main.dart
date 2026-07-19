import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const FasciaApp());
}

class FasciaApp extends StatelessWidget {
  const FasciaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LIFTNWITTIM Holistic Wellness',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
      ),
      home: const WelcomeScreen(),
    );
  }
}

// ── WELCOME SCREEN ────────────────────────────────────────────
class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              const Text(
                'LIFTNWITTIM',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 3),
              ),
              const Text(
                'HOLISTIC WELLNESS APP',
                style: TextStyle(
                    color: Colors.white54,
                    fontSize: 13,
                    letterSpacing: 2),
              ),
              const SizedBox(height: 16),
              const Text(
                'Welcome to the\nPath of Holistic Wellness.',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 26,
                    fontWeight: FontWeight.w300,
                    height: 1.4),
              ),
              const SizedBox(height: 20),
              const Divider(color: Colors.white12),
              const SizedBox(height: 12),
              const Text(
                'WHAT WE MEASURE',
                style: TextStyle(
                    color: Colors.white38,
                    fontSize: 11,
                    letterSpacing: 2),
              ),
              const SizedBox(height: 12),
              _infoCard(
                'M1  Shearing Force Algorithm',
                'Measures how well your body\'s internal layers slide past each other like slick pancakes — preventing stiff knots and ensuring pain-free movement.',
              ),
              _infoCard(
                'M2  Foot-to-Glute Connection',
                'Checks if your feet are acting as awake "antennas" that talk directly to your glutes for a strong, connected foundation.',
              ),
              _infoCard(
                'M3  Movement Bandwidth',
                'Tracks whether your body moves as one integrated, springy unit or in separate, clunky pieces that slow you down and limit your power.',
              ),
              _infoCard(
                'M4  Hydraulic Indicator',
                'Acts as a safety gate to confirm your "body oil" is thin and warm enough for you to move fast without the risk of injury.',
              ),
              _infoCard(
                'M5  Stability Map',
                'Measures your ability to spread out pressure during "organized chaos," protecting your joints from being overloaded during sudden or unpredictable movements.',
              ),
              const Spacer(),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                          builder: (_) => const HomeScreen()),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    minimumSize: const Size(double.infinity, 52),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(26)),
                  ),
                  child: const Text(
                    'Begin Assessment',
                    style: TextStyle(
                        color: Colors.black,
                        fontSize: 16,
                        fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _infoCard(String title, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(
            width: 4,
            child: DecoratedBox(
              decoration: BoxDecoration(color: Colors.white24),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold)),
                const SizedBox(height: 4),
                Text(description,
                    style: const TextStyle(
                        color: Colors.white54,
                        fontSize: 11,
                        height: 1.5)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── HOME SCREEN ───────────────────────────────────────────────
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late CameraController _controller;
  bool _isRearCamera = true;
  bool _isInitialized = false;
  bool _isAnalyzing = false;
  bool _showResults = false;
  String _archEngaged = 'neutral';

  double _score = 0;
  String _tier = '--';
  Map<String, dynamic> _debug = {};
  List<dynamic> _flags = [];

  final String backendUrl =
      'https://fascia-around-n-find-out-production.up.railway.app';

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final camera = _isRearCamera
        ? cameras.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.back,
            orElse: () => cameras[0])
        : cameras.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.front,
            orElse: () => cameras[0]);

    if (_isInitialized) {
      await _controller.dispose();
    }

    _controller = CameraController(camera, ResolutionPreset.high);
    await _controller.initialize();
    setState(() => _isInitialized = true);
  }

  Future<void> _toggleCamera() async {
    setState(() {
      _isRearCamera = !_isRearCamera;
      _isInitialized = false;
    });
    await _initCamera();
  }

  Color _tierColor(String tier) {
    switch (tier) {
      case 'ELITE':
        return const Color(0xFF1D9E75);
      case 'FUNCTIONAL':
        return const Color(0xFF378ADD);
      case 'COMPENSATING':
        return const Color(0xFFBA7517);
      case 'NO_BUENO':
        return const Color(0xFFE24B4A);
      default:
        return Colors.white54;
    }
  }

  Future<void> _analyze() async {
    if (_isAnalyzing) return;
    setState(() {
      _isAnalyzing = true;
      _showResults = false;
    });

    for (int i = 3; i > 0; i--) {
      setState(() => _tier = '$i...');
      await Future.delayed(const Duration(seconds: 1));
    }
    setState(() => _tier = 'Analyzing...');

    try {
      final image = await _controller.takePicture();
      final bytes = await image.readAsBytes();

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/analyze'),
      );

      request.files.add(
        http.MultipartFile.fromBytes('frame', bytes, filename: 'frame.jpg'),
      );
      request.fields['arch_engaged'] = _archEngaged;

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();
      print('RAW RESPONSE: $responseBody');
      final data = jsonDecode(responseBody);

      setState(() {
        _score = (data['score'] ?? 0).toDouble();
        _tier = (data['tier'] ?? 'unknown').toString();
        _debug = data['debug'] ?? {};
        _flags = data['flags'] ?? [];
        _showResults = true;
      });

      if (data['frame_received'] == true) {
        print('Resolution: ${data['resolution']}');
        final debug = data['debug'];
        if (debug != null) {
          print('M1 Shear: ${debug['m1_shear']}');
          print('M2 Foot-Glute: ${debug['m2_foot_glute']}');
          print('M3 Tensegrity: ${debug['m3_tensegrity']}');
          print('M4 Hydro: ${debug['m4_hydro']}');
          print('M5 Stability: ${debug['m5_stability']}');
        }
        if (data['flags'] != null && data['flags'].isNotEmpty) {
          print('FLAGS: ${data['flags']}');
        }
      }
    } catch (e) {
      print('ANALYZE ERROR: $e');
      setState(() {
        _score = 0;
        _tier = 'Error';
        _showResults = false;
      });
    }

    setState(() => _isAnalyzing = false);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Widget _moduleBar(String label, String description, dynamic value) {
    final double v = (value ?? 0).toDouble();
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                      fontWeight: FontWeight.bold)),
              Text(v.toStringAsFixed(1),
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 2),
          Text(description,
              style: const TextStyle(
                  color: Colors.white38, fontSize: 10, height: 1.3)),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: v / 100,
              minHeight: 6,
              backgroundColor: Colors.white12,
              valueColor: AlwaysStoppedAnimation<Color>(
                v >= 85
                    ? const Color(0xFF1D9E75)
                    : v >= 60
                        ? const Color(0xFF378ADD)
                        : v >= 35
                            ? const Color(0xFFBA7517)
                            : const Color(0xFFE24B4A),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Column(
        children: [
          Expanded(
            flex: 3,
            child: Stack(
              children: [
                _isInitialized
                    ? CameraPreview(_controller)
                    : const Center(child: CircularProgressIndicator()),
                Positioned(
                  top: 48,
                  right: 16,
                  child: GestureDetector(
                    onTap: _isAnalyzing ? null : _toggleCamera,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(24),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.flip_camera_ios,
                              color: Colors.white, size: 20),
                          const SizedBox(width: 6),
                          Text(
                            _isRearCamera ? 'Rear' : 'Front',
                            style: const TextStyle(
                                color: Colors.white, fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            flex: 2,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  // Score and tier
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Expanded(
                        child: Column(
                          children: [
                            const Text('SCORE',
                                style: TextStyle(
                                    color: Colors.white54, fontSize: 11)),
                            Text(
                              _showResults
                                  ? _score.toStringAsFixed(1)
                                  : '--',
                              style: TextStyle(
                                  color: _showResults
                                      ? _tierColor(_tier)
                                      : Colors.white,
                                  fontSize: 38,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          children: [
                            const Text('TIER',
                                style: TextStyle(
                                    color: Colors.white54, fontSize: 11)),
                            Text(
                              _tier,
                              style: TextStyle(
                                  color: _tierColor(_tier),
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),

                  // Module breakdown with descriptions
                  if (_showResults && _debug.isNotEmpty) ...[
                    _moduleBar(
                        'M1  Shearing Force',
                        'How freely your internal layers slide past each other',
                        _debug['m1_shear']),
                    _moduleBar(
                        'M2  Foot-to-Glute',
                        'Antenna signal from feet to glutes',
                        _debug['m2_foot_glute']),
                    _moduleBar(
                        'M3  Movement Bandwidth',
                        'Integrated springy unit vs clunky separate pieces',
                        _debug['m3_tensegrity']),
                    _moduleBar(
                        'M4  Hydraulic',
                        'Body oil warmth — safety gate for explosive movement',
                        _debug['m4_hydro']),
                    _moduleBar(
                        'M5  Stability',
                        'Pressure distribution during organized chaos',
                        _debug['m5_stability']),
                    const SizedBox(height: 8),
                  ],

                  // Flags
                  if (_showResults && _flags.isNotEmpty) ...[
                    const Divider(color: Colors.white12),
                    ..._flags.map((flag) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 3),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(
                                flag['severity'] == 'HIGH'
                                    ? Icons.warning_rounded
                                    : Icons.info_outline,
                                color: flag['severity'] == 'HIGH'
                                    ? const Color(0xFFE24B4A)
                                    : const Color(0xFFBA7517),
                                size: 14,
                              ),
                              const SizedBox(width: 6),
                              Expanded(
                                child: Text(
                                  flag['message'],
                                  style: TextStyle(
                                    color: flag['severity'] == 'HIGH'
                                        ? const Color(0xFFE24B4A)
                                        : const Color(0xFFBA7517),
                                    fontSize: 11,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        )),
                    const SizedBox(height: 8),
                  ],

                  // Arch engagement toggle
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('Arch:',
                          style: TextStyle(
                              color: Colors.white54, fontSize: 12)),
                      const SizedBox(width: 8),
                      ToggleButtons(
                        isSelected: [
                          _archEngaged == 'false',
                          _archEngaged == 'neutral',
                          _archEngaged == 'true',
                        ],
                        onPressed: (index) {
                          setState(() {
                            _archEngaged =
                                ['false', 'neutral', 'true'][index];
                          });
                        },
                        borderRadius: BorderRadius.circular(8),
                        selectedColor: Colors.black,
                        fillColor: Colors.white,
                        color: Colors.white54,
                        textStyle: const TextStyle(fontSize: 11),
                        constraints: const BoxConstraints(
                            minWidth: 72, minHeight: 32),
                        children: const [
                          Text('Collapsed'),
                          Text('Neutral'),
                          Text('Engaged'),
                        ],
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),

                  // Analyze button
                  ElevatedButton(
                    onPressed: _isAnalyzing ? null : _analyze,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      minimumSize: const Size(200, 48),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(24)),
                    ),
                    child: Text(
                      _isAnalyzing ? 'Analyzing...' : 'Analyze',
                      style: const TextStyle(
                          color: Colors.black,
                          fontSize: 16,
                          fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}