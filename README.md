# Real-Time Face Detection and Recognition System

This is a local Python computer-vision application that utilizes a webcam to capture live video frames. It actively detects human faces in real time and compares them against a single reference face to establish an identity match. The system operates entirely offline using deep learning facial feature extraction algorithms.

## Features

- Real-time webcam face detection
- Single-reference face recognition
- Multi-Face Detection & Recognition
- Bunny / Unknown labeling
- Euclidean distance display
- Configurable recognition tolerance
- Graceful error handling
- Webcam resource cleanup

## Technologies Used

- **Python**: Core programming language.
- **OpenCV (`cv2`)**: Used for hardware webcam interfacing, real-time video stream processing, and image rendering.
- **NumPy**: Used for matrix operations and numerical distance calculations.
- **face-recognition**: High-level facial feature extraction library that generates 128-dimensional encodings.
- **dlib**: Underlying C++ library containing the deep learning models for landmark estimation.
- **Pillow**: Used as a fallback dependency for robust image opening and validation.

## How It Works

The core workflow operates as follows:

```text
Webcam Frame
↓
BGR → RGB Conversion
↓
Face Detection
↓
Face Encoding
↓
128-Dimensional Face Embedding
↓
Euclidean Distance Comparison
↓
Distance <= 0.6 → Bunny
Distance > 0.6 → Unknown
```

To ensure smooth performance, the reference image (`images/bunny.jpeg`) is processed and encoded exactly once during the application's startup sequence. The resulting encoding is cached in memory and reused for all subsequent distance comparisons against live webcam frames. 

*Note: The calculated distance represents Euclidean distance in the 128-dimensional feature space, not a confidence percentage.*

## Project Structure

```text
Face Detection/
│
├── main.py                # Core application loop and configuration variables
├── requirements.txt       # List of required project dependencies
├── README.md              # Project documentation
├── pkg_resources.py       # Compatibility polyfill for environment stability
│
├── images/
│   └── bunny.jpeg         # The single reference identity image
│
└── src/
    ├── __init__.py
    ├── face_encoder.py    # Logic for validating and encoding the reference image
    └── face_detector.py   # Logic for processing webcam frames and comparing faces
```

## Installation

1. Install Python.
2. Install the required project dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Depending on your environment, installing `dlib` may require CMake and C++ build tools.)*

## Running the Project

To start the system, verify your webcam is connected and run:

```bash
python main.py
```

Press **Q** to safely release webcam resources and exit the application.
