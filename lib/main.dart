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
      title: 'Fascia App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late CameraController _controller;
  bool _isInitialized = false;
  bool _isAnalyzing = false;
  bool _showResults = false;

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
    _controller = CameraController(cameras[0], ResolutionPreset.high);
    await _controller.initialize();
    setState(() => _isInitialized = true);
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
      request.fields['arch_engaged'] = 'neutral';

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();
      final data = jsonDecode(responseBody);

      setState(() {
        _score = (data['score'] ?? 0).toDouble();
        _tier = (data['tier'] ?? 'unknown').toString();
        _debug = data['debug'] ?? {};
        _flags = data['flags'] ?? [];
        _showResults = true;
      });
    } catch (e) {
      setState(() {
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

  Widget _moduleBar(String label, dynamic value) {
    final double v = (value ?? 0).toDouble();
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(color: Colors.white70, fontSize: 11)),
              Text('${v.toStringAsFixed(1)}',
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 3),
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
            child: _isInitialized
                ? CameraPreview(_controller)
                : const Center(child: CircularProgressIndicator()),
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

                  // Module breakdown
                  if (_showResults && _debug.isNotEmpty) ...[
                    _moduleBar('M1  Shear', _debug['m1_shear']),
                    _moduleBar('M2  Foot-Glute', _debug['m2_foot_glute']),
                    _moduleBar('M3  Tensegrity', _debug['m3_tensegrity']),
                    _moduleBar('M4  Hydraulic', _debug['m4_hydro']),
                    _moduleBar('M5  Stability', _debug['m5_stability']),
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