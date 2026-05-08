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
  String _score = '--';
  String _tier = '--';
  bool _isAnalyzing = false;

  final String backendUrl = 'https://fascia-around-n-find-out-production.up.railway.app';

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      cameras[0],
      ResolutionPreset.high,
    );
    await _controller.initialize();
    setState(() => _isInitialized = true);
  }

  Future<void> _analyze() async {
    if (_isAnalyzing) return;
    setState(() => _isAnalyzing = true);

    try {
      final image = await _controller.takePicture();
      final bytes = await image.readAsBytes();

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$backendUrl/analyze'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'frame',
          bytes,
          filename: 'frame.jpg',
        ),
      );

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();
      final data = jsonDecode(responseBody);

      setState(() {
        _score = (data['score'] ?? 0).toString();
        _tier = (data['tier'] ?? 'unknown').toString();
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
print('Warmup adequate: ${debug['warmup_adequate']}');
print('Spike count: ${debug['spike_count']}');
print('Lag ms: ${debug['lag_ms']}');
        }
        if (data['flags'] != null && data['flags'].isNotEmpty) {
          print('FLAGS: ${data['flags']}');
        }
      }
    } catch (e) {
      setState(() {
        _score = 'Error';
        _tier = e.toString();
      });
    }

    setState(() => _isAnalyzing = false);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
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
            flex: 1,
            child: SingleChildScrollView(
              child: Container(
                padding: const EdgeInsets.all(16),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        Expanded(
                          child: Column(
                            children: [
                              const Text('SCORE',
                                  style: TextStyle(
                                      color: Colors.white54, fontSize: 12)),
                              Text(_score,
                                  style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 36,
                                      fontWeight: FontWeight.bold)),
                            ],
                          ),
                        ),
                        Expanded(
                          child: Column(
                            children: [
                              const Text('TIER',
                                  style: TextStyle(
                                      color: Colors.white54, fontSize: 12)),
                              Text(_tier,
                                  style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold)),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    ElevatedButton(
                      onPressed: _isAnalyzing ? null : _analyze,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        minimumSize: const Size(200, 50),
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
          ),
        ],
      ),
    );
  }
}