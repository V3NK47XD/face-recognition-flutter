Mobile On-Device Face Recognition 📱🧠

A Flutter-based mobile application that performs offline face recognition directly on the device using TensorFlow Lite.
The app captures a face using the camera and compares it with a previously captured face to determine if they match.

Everything runs locally on the device — no internet, no cloud APIs.

🚀 Features

📷 Capture face using Flutter Camera

🧠 Face detection using Google ML Kit

⚡ Face embedding generation using TensorFlow Lite

🔒 Fully offline face comparison

📱 Runs directly on mobile devices

🪶 Lightweight model option for smaller apps

🛠 Tech Stack

Flutter

TensorFlow Lite

tflite_flutter

google_mlkit_face_detection

🧠 How It Works

1️⃣ User captures the first face image
2️⃣ The app extracts face embeddings using a MobileFaceNet model
3️⃣ User captures another face
4️⃣ The app compares both embeddings
5️⃣ If the distance is below a threshold → faces match

All computations happen on-device.

📦 Models Used

Two MobileFaceNet models are included:

Model	Size	Notes
mobilefacenet-128output-800kb.tflite	~800 KB	Smaller model, faster but less accurate
mobilefacenet-512-big.tflite	~153 MB	Larger model, higher accuracy

You can choose depending on accuracy vs app size tradeoff.
