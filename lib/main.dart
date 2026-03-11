import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

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
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FacePage(),
    );
  }
}

class FacePage extends StatefulWidget {
  const FacePage({super.key});
  @override
  State<FacePage> createState() => _FacePageState();
}

class _FacePageState extends State<FacePage> {
  late CameraController controller;
  late Interpreter interpreter;

  final FaceDetector faceDetector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.fast),
  );

  List<double>? firstEmbedding;
  bool isProcessing = false;

  String resultText = "Capture first face 👤";

  @override
  void initState() {
    super.initState();
    initAll();
  }

  Future<void> initAll() async {
    controller = CameraController(
      cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.front,
      ),
      ResolutionPreset.medium,
    );

    await controller.initialize();

    interpreter = await Interpreter.fromAsset("models/mobilefacenet.tflite");

    setState(() {});
  }

  Future<void> captureAndProcess() async {
    if (isProcessing) return;
    isProcessing = true;

    setState(() {
      resultText = "Processing... ⏳";
    });

    try {
      final XFile file = await controller.takePicture();
      final inputImage = InputImage.fromFilePath(file.path);
      final faces = await faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        setState(() {
          resultText = "No face detected ❌";
        });
        isProcessing = false;
        return;
      }

      final imageBytes = await file.readAsBytes();
      img.Image? baseImage = img.decodeImage(imageBytes);
      if (baseImage == null) return;

      final face = faces.first.boundingBox;

      int x = face.left.toInt();
      int y = face.top.toInt();
      int w = face.width.toInt();
      int h = face.height.toInt();

      // Clamp start position
      x = x.clamp(0, baseImage.width - 1);
      y = y.clamp(0, baseImage.height - 1);

      // Make sure width/height stay inside bounds
      if (x + w > baseImage.width) {
        w = baseImage.width - x;
      }
      if (y + h > baseImage.height) {
        h = baseImage.height - y;
      }

      // Extra safety (avoid zero/negative size)
      if (w <= 0 || h <= 0) {
        print("Invalid crop dimensions");
        return;
      }

      img.Image cropped = img.copyCrop(
        baseImage,
        x: x,
        y: y,
        width: w,
        height: h,
      );

      var inputShape = interpreter.getInputTensor(0).shape;
      int inputSize = inputShape[1];

      img.Image resized = img.copyResize(
        cropped,
        width: inputSize,
        height: inputSize,
      );

      var input = imageToFloat32(resized, inputSize);
      print("majaa ${interpreter.getOutputTensor(0).shape}");

      int outputSize = interpreter.getOutputTensor(0).shape[1];
      var output = List.generate(1, (_) => List.filled(outputSize, 0.0));

      interpreter.run(input, output);

      List<double> embedding = output[0];

      if (firstEmbedding == null) {
        firstEmbedding = embedding;
        setState(() {
          resultText = "First face saved ✅\nCapture again to compare.";
        });
      } else {
        double similarity = cosineSimilarity(firstEmbedding!, embedding);

        if (similarity > 0.6) {
          setState(() {
            resultText =
                "Same Person 🎉\nSimilarity: ${similarity.toStringAsFixed(2)}";
          });
        } else {
          setState(() {
            resultText =
                "Different Person ❌\nSimilarity: ${similarity.toStringAsFixed(2)}";
          });
        }
      }
    } catch (e) {
      setState(() {
        resultText = "Error: $e";
      });
    }

    isProcessing = false;
  }

  List imageToFloat32(img.Image image, int size) {
    var convertedBytes = Float32List(1 * size * size * 3);
    int index = 0;

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        var pixel = image.getPixel(x, y);

        convertedBytes[index++] = (pixel.r - 128) / 128;
        convertedBytes[index++] = (pixel.g - 128) / 128;
        convertedBytes[index++] = (pixel.b - 128) / 128;
      }
    }

    return convertedBytes.reshape([1, size, size, 3]);
  }

  double cosineSimilarity(List<double> e1, List<double> e2) {
    double dot = 0, norm1 = 0, norm2 = 0;

    for (int i = 0; i < e1.length; i++) {
      dot += e1[i] * e2[i];
      norm1 += e1[i] * e1[i];
      norm2 += e2[i] * e2[i];
    }

    return dot / (sqrt(norm1) * sqrt(norm2));
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Face Recognition 🔐")),
      body: Column(
        children: [
          Expanded(child: CameraPreview(controller)),
          Container(
            padding: const EdgeInsets.all(16),
            child: Text(
              resultText,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
          ElevatedButton(
            onPressed: captureAndProcess,
            child: const Text("Capture Face 📸"),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}
