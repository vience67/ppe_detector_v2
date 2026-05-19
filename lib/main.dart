import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'dart:math';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PPE Detector v2',
      theme: ThemeData.dark(),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<Detection> _detections = [];
  String _status = "Loading...";
  bool _isReady = false;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // 1. Загрузка меток
      _labels = (await rootBundle.loadString('assets/labels.txt'))
          .split('\n')
          .where((s) => s.trim().isNotEmpty)
          .toList();

      // 2. Загрузка модели
      _interpreter = await Interpreter.fromAsset('assets/detect.tflite');
      
      // 3. Инициализация камеры
      _controller = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _controller.initialize();

      setState(() {
        _isReady = true;
        _status = "Ready";
      });

      // 4. Запуск потока
      await _controller.startImageStream(_processFrame);
    } catch (e) {
      setState(() {
        _status = "Error: $e";
      });
    }
  }

  void _processFrame(CameraImage image) {
    if (!_isReady || _interpreter == null) return;

    try {
      // Конвертация изображения
      final input = _preprocessImage(image);
      
      // Создание output буфера (SSD MobileNet: [1, 10, 6] = 60 float)
      final output = List.filled(60, 0.0);
      
      // Запуск инференса
      _interpreter!.run(input, output);
      
      // Парсинг детекций
      _parseDetections(output);
      
    } catch (e) {
      print("Error: $e");
    }
  }

  List<double> _preprocessImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int targetSize = 300; // SSD MobileNet использует 300x300
    
    final List<double> input = List.filled(targetSize * targetSize * 3, 0.0);
    
    final Uint8List y = image.planes[0].bytes;
    final Uint8List u = image.planes[1].bytes;
    final Uint8List v = image.planes[2].bytes;
    final int yRow = image.planes[0].bytesPerRow;
    final int uvRow = image.planes[1].bytesPerRow;
    final int uvPixel = uvRow ~/ (width ~/ 2);
    
    int idx = 0;
    for (int ty = 0; ty < targetSize; ty++) {
      final int sy = (ty * height / targetSize).floor();
      final int yOff = sy * yRow;
      final int uvOff = (sy ~/ 2) * uvRow;
      
      for (int tx = 0; tx < targetSize; tx++) {
        final int sx = (tx * width / targetSize).floor();
        final int uvIdx = uvOff + (sx ~/ 2) * uvPixel;
        
        final int yVal = y[yOff + sx];
        final int uVal = u[uvIdx];
        final int vVal = v[uvIdx];
        
        // YUV to RGB
        int r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
        int g = (yVal - 0.698001 * (uVal - 128) - 0.337633 * (vVal - 128)).round().clamp(0, 255);
        int b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);
        
        // Normalize to [-1, 1]
        input[idx++] = (r / 127.5) - 1.0;
        input[idx++] = (g / 127.5) - 1.0;
        input[idx++] = (b / 127.5) - 1.0;
      }
    }
    
    return input;
  }

  void _parseDetections(List<double> output) {
    List<Detection> newDetections = [];
    
    // SSD MobileNet output format: [1, 10, 6]
    // Для каждого из 10 детекций: [class_id, confidence, y_min, x_min, y_max, x_max]
    
    for (int i = 0; i < 10; i++) {
      final int baseIdx = i * 6;
      final int classId = output[baseIdx].round();
      final double confidence = output[baseIdx + 1];
      
      if (confidence > 0.5 && classId > 0 && classId < _labels.length) {
        newDetections.add(Detection(
          label: _labels[classId],
          confidence: confidence,
          bbox: Rect.fromLTRB(
            output[baseIdx + 3], // x_min
            output[baseIdx + 2], // y_min
            output[baseIdx + 5], // x_max
            output[baseIdx + 4], // y_max
          ),
        ));
      }
    }
    
    setState(() {
      _detections = newDetections;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_isReady) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(_status),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_controller),
          
          // Статус
          Positioned(
            top: 40,
            left: 10,
            right: 10,
            child: Container(
              padding: const EdgeInsets.all(8),
              color: Colors.black54,
              child: Text(
                _status,
                style: const TextStyle(color: Colors.yellow, fontSize: 14),
              ),
            ),
          ),
          
          // Детекции
          Positioned(
            bottom: 20,
            left: 10,
            right: 10,
            child: Container(
              padding: const EdgeInsets.all(10),
              color: Colors.black87,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: _detections.map((det) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(
                    '${det.label}: ${(det.confidence * 100).toInt()}%',
                    style: const TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                )).toList(),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    _interpreter?.close();
    super.dispose();
  }
}

class Detection {
  final String label;
  final double confidence;
  final Rect bbox;
  
  Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });
}